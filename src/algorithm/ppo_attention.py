import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import network.actorcritic as AC
import datetime

class PPOAttention:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        model_name=None,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon=0.2,
        epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_heads=4,
        curriculum_threshold=0.6,  # 목표 달성 임계값
        curriculum_factor=0.95,    # 난이도 조정 계수
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_critic = AC.ActorCriticWithAttention(state_dim, action_dim, num_heads).to(device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.attention.parameters()},
            {'params': self.actor_critic.feature_extraction.parameters()},
            {'params': self.actor_critic.actor_direction.parameters()},
            {'params': self.actor_critic.actor_direction_std},            
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic}
        ], lr=lr_actor)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        self.memory = deque()
        
        # 어텐션 관련 하이퍼파라미터
        self.warmup_steps = 1000
        self.attention_weight = 0.0
        self.attention_weight_increment = 0.001
        self.total_steps = 0
        
        # 커리큘럼 학습 파라미터 추가
        self.curriculum_threshold = curriculum_threshold
        self.curriculum_factor = curriculum_factor
        self.current_difficulty = 1.0

        # 탐험 관련 파라미터 추가
        self.training = True
        self.exploration_noise = 1.0
        self.min_exploration_noise = 0.1
        self.exploration_decay = 0.995
        
        # 입실론 그리디 파라미터 추가
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.current_epsilon = self.epsilon_start

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 입실론 그리디 전략 구현
            if self.training and np.random.random() < self.current_epsilon:
                # 무작위 행동 선택 (discrete: -1, 0, 1)
                random_action = np.random.choice([-1.0, 0.0, 1.0])
                return (
                    np.array([random_action]),
                    0.0,  # 무작위 행동의 value
                    0.0   # 무작위 행동의 log_prob
                )
            
            # ActorCritic에서 이미 discrete한 direction_mean을 받음
            direction_mean, direction_std, value, _, action_probs = self.actor_critic(state)
            
            if self.training:
                # Categorical 분포에서 행동 선택
                action_probs = F.softmax(action_probs, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action_idx = action_dist.sample()
                
                # 인덱스를 실제 행동값으로 변환 (-1, 0, 1)
                action = action_idx.float() - 1.0  # [0,1,2] -> [-1,0,1]
                
                # log probability 계산
                log_prob = action_dist.log_prob(action_idx)
            else:
                # 평가 시에는 가장 높은 확률의 행동 선택
                action = direction_mean
                log_prob = torch.log(action_probs.max(dim=-1)[0])

            return (
                action.cpu().numpy(),
                value.cpu().numpy()[0],
                log_prob.cpu().numpy()
            )
        
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self, batch_size=64, success_rate=0.0):
        if len(self.memory) < batch_size:
            return 0
        
        # 커리큘럼 학습: 성공률에 따른 난이도 조정
        if success_rate > self.curriculum_threshold:
            self.current_difficulty *= self.curriculum_factor  # 난이도 증가
        else:
            self.current_difficulty = min(1.0, self.current_difficulty / self.curriculum_factor)  # 난이도 감소
        
        # 난이도에 따른 리워드 스케일링
        reward_batch = []
        for transition in self.memory:
            state, action, reward, next_state, log_prob, value, done = transition
            scaled_reward = reward * self.current_difficulty  # 난이도에 따른 리워드 조정
            reward_batch.append(scaled_reward)
        
        # 메모리에서 데이터 추출
        state_batch = []
        action_batch = []
        next_state_batch = []
        log_prob_batch = []
        value_batch = []
        done_batch = []
        
        for transition in self.memory:
            state, action, reward, next_state, log_prob, value, done = transition
            state_batch.append(state)
            action_batch.append(action)
            next_state_batch.append(next_state)
            log_prob_batch.append(log_prob)
            value_batch.append(value)
            done_batch.append(done)
        
        # 텐서로 변환 (어텐션을 위한 차원 추가)
        state_batch = torch.FloatTensor(np.array(state_batch)).unsqueeze(1).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).unsqueeze(1).to(self.device)
        old_log_prob_batch = torch.FloatTensor(np.array(log_prob_batch)).to(self.device)
        old_value_batch = torch.FloatTensor(np.array(value_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # GAE 계산
        advantages = []
        returns = []
        gae = 0
        
        with torch.no_grad():
            next_value = self.actor_critic(next_state_batch)[2]
            next_value = next_value.squeeze()
            
            for r, v, done, next_v in zip(
                reversed(reward_batch),
                reversed(old_value_batch),
                reversed(done_batch),
                reversed(next_value)
            ):
                if done:
                    delta = r - v
                    gae = delta
                else:
                    delta = r + self.gamma * next_v - v
                    gae = delta + self.gamma * 0.95 * gae
                
                returns.insert(0, gae + v)
                advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            indices = np.random.permutation(len(state_batch))
            for start_idx in range(0, len(state_batch), batch_size):
                idx = indices[start_idx:start_idx + batch_size]
                
                if len(idx) < batch_size:
                    break
                
                state = state_batch[idx]
                action = action_batch[idx]
                advantage = advantages[idx]
                return_ = returns[idx]
                old_log_prob = old_log_prob_batch[idx]
                
                # 값 검증 및 클리핑 추가
                direction_mean, direction_std, value, attention_weights, action_probs = self.actor_critic(state)
                
                # NaN 체크 및 처리
                if torch.isnan(direction_mean).any() or torch.isnan(direction_std).any():
                    print("Warning: NaN values detected in policy parameters")
                    continue
                
                try:
                    # Categorical 분포 사용
                    action_probs = F.softmax(action_probs, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    
                    # 행동 인덱스로 변환 ([-1,0,1] -> [0,1,2])
                    action_idx = (action + 1).long()
                    new_log_prob = action_dist.log_prob(action_idx)
                except ValueError as e:
                    print(f"Error in distribution: {e}")
                    continue
                
                # 차원 맞추기
                if len(new_log_prob.shape) < len(old_log_prob.shape):
                    new_log_prob = new_log_prob.unsqueeze(-1)
                if len(old_log_prob.shape) < len(new_log_prob.shape):
                    old_log_prob = old_log_prob.unsqueeze(-1)
                
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # advantage의 차원 확장이 필요한 경우
                if len(ratio.shape) > len(advantage.shape):
                    advantage = advantage.unsqueeze(-1)
                
                # 1. Actor Loss - KL 페널티 추가
                kl_div = 0.5 * ((new_log_prob - old_log_prob) ** 2).mean()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * kl_div
                
                # 2. Critic Loss - Huber Loss 사용
                value = value.squeeze(-1)
                critic_loss = nn.SmoothL1Loss()(value, return_)
                
                # 3. 엔트로피와 정규화 손실
                entropy_loss = -0.01 * action_dist.entropy().mean()
                
                # 4. 거래 관련 페널티
                trading_fee = 0.0003
                fee_penalty = trading_fee * torch.abs(action).mean()
                position_change_penalty = 0.005 * torch.abs(action[1:] - action[:-1]).mean()
                
                # 5. 리스크 관리 손실
                max_drawdown_penalty = 0.02 * torch.max(torch.cumsum(torch.min(action, torch.zeros_like(action)), dim=0))
                volatility_penalty = 0.015 * torch.std(action)
                
                # 6. 어텐션 정규화
                attention_regularization = 0.01 * torch.mean(torch.abs(attention_weights))
                
                # 전체 손실 함수 조합
                loss = (
                    2.0 * actor_loss +  # 액터 손실 가중치 증가
                    0.5 * critic_loss * self.current_difficulty +
                    0.02 * entropy_loss +  # 엔트로피 가중치 증가
                    0.0001 * fee_penalty * self.current_difficulty +  # 거래 비용 페널티 감소
                    0.001 * position_change_penalty * self.current_difficulty +  # 포지션 변경 페널티 감소
                    0.01 * max_drawdown_penalty * self.current_difficulty +
                    0.01 * volatility_penalty * self.current_difficulty +
                    self.attention_weight * attention_regularization
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        self.memory.clear()
        return 1
    
    def save_model(self, data, path):
        torch.save(data, path)

    def checkpoint(self, data, path):
        torch.save(data, path)

    def load_model(self):
        self.actor_critic.load_state_dict(torch.load(f'models/{self.model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))

    def load_checkpoint(self, path):
        return torch.load(path)