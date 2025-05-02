import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import datetime
import network.actorcritic as AC
import network.indicator as ID

class PPO3:
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
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.actor_critic = AC.ActorCritic2(state_dim, action_dim).to(device)
        self.indicator_distribution = ID.IndicatorDistribution(state_dim, action_dim).to(device)
        
        self.optimizer = optim.SGD([
            {'params': self.actor_critic.feature_extraction.parameters()},
            {'params': self.actor_critic.actor_direction.parameters()},
            {'params': self.actor_critic.actor_direction_std},            
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic},
            {'params': self.indicator_distribution.final_network.parameters(), 'lr': lr_critic}
        ], lr=lr_actor, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        self.memory = deque()
        
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state = state[:, 5:]  # 5번 인덱스부터 마지막까지의 데이터만 사용
        with torch.no_grad():
            value, action_probs, action_logits = self.actor_critic(state)
            pi_I = self.indicator_distribution(state)
            alpha = 0.3
            pi = alpha * action_probs + (1 - alpha) * pi_I
            
            # Categorical 분포에서 액션 샘플링
            action_dist = torch.distributions.Categorical(pi)
            action_idx = action_dist.sample()
            action = action_idx.float() - 1.0
            log_prob = action_dist.log_prob(action_idx)
        
        return (
            action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0]
        )
        
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self, batch_size=32, success_rate=None):
        if len(self.memory) < batch_size:
            return 0
            
        # 메모리에서 데이터 추출
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        log_prob_batch = []
        value_batch = []
        done_batch = []
        
        for transition in self.memory:
            state, action, reward, next_state, log_prob, value, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            log_prob_batch.append(log_prob)
            value_batch.append(value)
            done_batch.append(done)
        
        # 텐서로 변환
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        old_log_prob_batch = torch.FloatTensor(np.array(log_prob_batch)).to(self.device)
        old_value_batch = torch.FloatTensor(np.array(value_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # GAE 계산
        advantages = []
        returns = []
        gae = 0
        
        with torch.no_grad():
            next_state_batch = next_state_batch[:, 5:]
            next_value = self.actor_critic(next_state_batch)[0]  # value는 첫 번째 반환값
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
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트
        for _ in range(self.epochs):
            # 미니배치 생성 (섞지 않고 순차적으로)
            for start_idx in range(0, len(state_batch), batch_size):
                end_idx = min(start_idx + batch_size, len(state_batch))
                idx = range(start_idx, end_idx)
                
                if len(idx) < batch_size:
                    break
                
                # 현재 미니배치
                state = state_batch[idx]
                state = state[:, 5:]
                action = action_batch[idx]
                advantage = advantages[idx]
                return_ = returns[idx]
                old_log_prob = old_log_prob_batch[idx]
                
                # 현재 정책의 행동 분포
                value, action_probs, action_logits = self.actor_critic(state)
                pi_I = self.indicator_distribution(state)
                alpha = 0.3
                pi = alpha * action_probs + (1 - alpha) * pi_I
                
                # Categorical 분포에서 액션 샘플링
                action_dist = torch.distributions.Categorical(pi)
                action_idx = action_dist.sample()
                new_action = action_idx.float() - 1.0
                new_log_prob = action_dist.log_prob(action_idx)
                
                # PPO 비율 계산
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # 핵심 손실 함수들
                
                # 1. Actor Loss - PPO 클리핑 손실
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 2. Critic Loss - Huber Loss
                value = value.squeeze(-1)
                critic_loss = nn.SmoothL1Loss()(value, return_)
                
                # 3. 엔트로피 손실 (탐색을 위한)
                entropy_loss = -0.01 * action_dist.entropy().mean()

                
                # 전체 손실 함수
                loss = actor_loss + 0.5 * critic_loss + entropy_loss
                
                # 역전파 및 최적화
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        # 메모리 비우기
        self.memory.clear()
        return 1
    
    def save_model(self, data, path):
        # NumPy 스칼라를 Python 기본 타입으로 변환하는 함수
        def convert_numpy_scalars(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_scalars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_scalars(v) for v in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 데이터 변환
        converted_data = convert_numpy_scalars(data)
        
        # 변환된 데이터 저장
        torch.save(converted_data, path)
    
    def checkpoint(self, data, path):
        torch.save(data, path)
    
    def load_model(self):
        self.actor_critic.load_state_dict(torch.load(f'models/{self.model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))
    
    def load_checkpoint(self, path):
        return torch.load(path)