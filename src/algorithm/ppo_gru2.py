import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import network.actorcritic as AC
import datetime

class PPOGRU2:
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
        self.device = device
        
        # 3개의 actor_critic 생성 (각각 5개의 state_dim을 가짐)
        self.actor_critics = nn.ModuleList([
            AC.ActorCriticGRU(5, action_dim).to(device) for _ in range(3)
        ])
        
        # 학습률 스케줄링 파라미터 추가
        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic
        self.lr_decay = 0.995
        self.min_lr = 1e-5
        
        # 각 actor_critic에 대한 옵티마이저 생성
        self.optimizers = [
            optim.Adam([
                {'params': actor_critic.gru.parameters()},
                {'params': actor_critic.feature_extraction.parameters()},
                {'params': actor_critic.actor_direction.parameters()},
                {'params': actor_critic.actor_direction_std},            
                {'params': actor_critic.critic.parameters(), 'lr': lr_critic}
            ], lr=lr_actor) for actor_critic in self.actor_critics
        ]
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.model_name = model_name
        self.memory = deque()
        
        # 경험 리플레이 메모리 추가
        self.replay_buffer = deque(maxlen=10000)
        self.replay_batch_size = 32
        self.replay_ratio = 0.2  # 리플레이에서 샘플링할 비율
        
        # 어텐션 관련 하이퍼파라미터
        self.warmup_steps = 1000
        self.attention_weight = 0.0
        self.attention_weight_increment = 0.001
        self.total_steps = 0
        
        # GAE 파라미터 최적화
        self.gae_lambda = 0.95  # GAE 람다 파라미터
        self.gae_gamma = 0.99   # GAE 감마 파라미터
        
        # 클리핑 범위 동적 조정 파라미터
        self.initial_epsilon = epsilon
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995
        self.current_epsilon = epsilon
        
        # 배치 정규화 파라미터
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.01
        
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
        
        # 확신 기반 거래 파라미터 추가
        self.confidence_threshold = 0.85  # 0.75에서 0.85로 증가
        self.min_trade_interval = 10  # 5에서 10으로 증가
        self.last_trade_step = -self.min_trade_interval  # 마지막 거래 스텝
        self.confidence_history = deque(maxlen=10)  # 확신도 기록
        self.trade_count = 0  # 거래 횟수
        self.successful_trades = 0  # 성공한 거래 횟수
        
        # 멀티스텝 학습 파라미터
        self.n_step = 3  # n-step 리턴 계산
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # 성능 모니터링
        self.performance_history = []
        self.best_performance = float('-inf')
        self.patience = 20
        self.patience_counter = 0
        
        # 각 actor_critic의 가중치 (초기값은 동일하게 설정)
        self.actor_weights = [1.0/3, 1.0/3, 1.0/3]

    def select_action(self, state):
        # 상태를 3개의 부분으로 나누기
        state_parts = np.array_split(state, 3)
        state_parts = [torch.FloatTensor(part).unsqueeze(0).unsqueeze(0).to(self.device) for part in state_parts]
        
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
            
            # 각 actor_critic에서 행동과 가치 예측
            all_direction_means = []
            all_direction_stds = []
            all_values = []
            all_action_probs = []
            
            for i, actor_critic in enumerate(self.actor_critics):
                direction_mean, direction_std, value, _, action_probs = actor_critic(state_parts[i])
                all_direction_means.append(direction_mean)
                all_direction_stds.append(direction_std)
                all_values.append(value)
                all_action_probs.append(action_probs)
            
            # 가중 평균으로 최종 행동 결정
            weighted_direction_mean = sum(w * m for w, m in zip(self.actor_weights, all_direction_means))
            weighted_direction_std = sum(w * s for w, s in zip(self.actor_weights, all_direction_stds))
            weighted_value = sum(w * v for w, v in zip(self.actor_weights, all_values))
            
            # 액션 확률의 가중 평균 계산
            weighted_action_probs = sum(w * p for w, p in zip(self.actor_weights, all_action_probs))
            
            # 확신도 계산 (액션 확률의 최대값)
            confidence = torch.max(F.softmax(weighted_action_probs, dim=-1)).item()
            self.confidence_history.append(confidence)
            
            # 현재 스텝과 마지막 거래 스텝의 차이
            steps_since_last_trade = self.total_steps - self.last_trade_step
            
            # 거래 조건 확인
            should_trade = (
                confidence > self.confidence_threshold and  # 확신도가 임계값 이상
                steps_since_last_trade >= self.min_trade_interval and  # 최소 거래 간격 경과
                len(self.confidence_history) >= 5 and  # 충분한 확신도 기록
                np.mean(list(self.confidence_history)[-5:]) > self.confidence_threshold  # 최근 5개 확신도 평균이 임계값 이상
            )
            
            if self.training:
                # 거래 조건이 충족되면 행동 선택
                if should_trade:
                    # Categorical 분포에서 행동 선택
                    action_probs = F.softmax(weighted_action_probs, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action_idx = action_dist.sample()
                    
                    # 인덱스를 실제 행동값으로 변환 (-1, 0, 1)
                    action = action_idx.float() - 1.0  # [0,1,2] -> [-1,0,1]
                    
                    # log probability 계산
                    log_prob = action_dist.log_prob(action_idx)
                    
                    # 거래 기록 업데이트
                    self.last_trade_step = self.total_steps
                    self.trade_count += 1
                else:
                    # 거래 조건이 충족되지 않으면 HOLD(0) 선택
                    action = torch.tensor([0.0]).to(self.device)
                    log_prob = torch.log(torch.tensor([0.5])).to(self.device)  # 중립적인 log_prob
            else:
                # 평가 시에는 가장 높은 확률의 행동 선택
                action = weighted_direction_mean
                log_prob = torch.log(weighted_action_probs.max(dim=-1)[0])

            return (
                action.cpu().numpy(),
                weighted_value.cpu().numpy()[0],
                log_prob.cpu().numpy()
            )
    
    def store_transition(self, transition):
        self.memory.append(transition)
        
        # n-step 버퍼에 추가
        self.n_step_buffer.append(transition)
        
        # n-step 버퍼가 가득 차면 n-step 리턴 계산
        if len(self.n_step_buffer) == self.n_step:
            n_step_return = 0
            for i, (_, _, reward, _, _, _, _) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * reward
            
            # n-step 리턴으로 첫 번째 전환 업데이트
            state, action, _, next_state, log_prob, value, done = self.n_step_buffer[0]
            n_step_transition = (state, action, n_step_return, next_state, log_prob, value, done)
            
            # 리플레이 버퍼에 추가
            self.replay_buffer.append(n_step_transition)
            
            # n-step 버퍼에서 첫 번째 전환 제거
            self.n_step_buffer.popleft()
    
    def update(self, batch_size=64, success_rate=0.0):
        if len(self.memory) < batch_size:
            return 0
        
        # 성능 모니터링
        current_performance = success_rate
        self.performance_history.append(current_performance)
        
        # 조기 종료 검사
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            return 0
        
        # 학습률 스케줄링
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                if 'lr' in param_group:
                    param_group['lr'] = max(param_group['lr'] * self.lr_decay, self.min_lr)
        
        # 클리핑 범위 동적 조정
        self.current_epsilon = max(self.current_epsilon * self.epsilon_decay, self.min_epsilon)
        
        # 커리큘럼 학습: 성공률에 따른 난이도 조정
        if success_rate > self.curriculum_threshold:
            self.current_difficulty *= self.curriculum_factor  # 난이도 증가
        else:
            self.current_difficulty = min(1.0, self.current_difficulty / self.curriculum_factor)  # 난이도 감소
        
        # 거래 성공률에 따른 확신 임계값 조정
        if self.trade_count > 0:
            trade_success_rate = self.successful_trades / self.trade_count
            if trade_success_rate > 0.7:  # 높은 성공률
                self.confidence_threshold = max(0.75, self.confidence_threshold - 0.005)
            elif trade_success_rate < 0.4:  # 낮은 성공률
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.01)
        
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
        
        # 텐서로 변환
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        
        # inhomogeneous shape 문제 해결을 위한 개별 처리
        old_log_prob_batch = []
        for log_prob in log_prob_batch:
            if isinstance(log_prob, float):
                old_log_prob_batch.append([log_prob])
            else:
                old_log_prob_batch.append(log_prob)
        old_log_prob_batch = torch.FloatTensor(np.array(old_log_prob_batch)).to(self.device)
        
        old_value_batch = []
        for value in value_batch:
            if isinstance(value, float):
                old_value_batch.append([value])
            else:
                old_value_batch.append(value)
        old_value_batch = torch.FloatTensor(np.array(old_value_batch)).to(self.device)
        
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # GAE 계산 - 개선된 버전
        advantages = []
        returns = []
        gae = 0
        
        with torch.no_grad():
            # ppo_gru.py와 동일한 방식으로 next_value 계산
            next_state_parts = [torch.FloatTensor(part).unsqueeze(0).unsqueeze(0).to(self.device) 
                               for part in np.array_split(next_state_batch[-1].cpu().numpy(), 3)]
            
            # 각 actor_critic에서 next_value 계산
            next_values = [actor_critic(next_state_part)[2] for actor_critic, next_state_part in zip(self.actor_critics, next_state_parts)]
            next_value = sum(w * v for w, v in zip(self.actor_weights, next_values))
            next_value = next_value.squeeze()
            
            # 각 항목을 개별적으로 처리하여 반복
            for i in range(len(reward_batch)):
                r = reward_batch[i]
                v = old_value_batch[i]
                done = done_batch[i]
                
                if done:
                    delta = r - v
                    gae = delta
                else:
                    # 그래디언트 요구사항 제거
                    next_value_detached = next_value.detach()
                    delta = r + self.gae_gamma * next_value_detached - v
                    gae = delta + self.gae_gamma * self.gae_lambda * gae
                
                returns.insert(0, gae + v)
                advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 배치 정규화 적용
        if self.use_batch_norm and advantages.numel() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 리플레이 버퍼에서 샘플링
        replay_indices = []
        if len(self.replay_buffer) > self.replay_batch_size:
            replay_indices = np.random.choice(
                len(self.replay_buffer), 
                size=int(batch_size * self.replay_ratio), 
                replace=False
            )
            
            # 리플레이 데이터 추가
            for idx in replay_indices:
                state, action, reward, next_state, log_prob, value, done = self.replay_buffer[idx]
                state_batch = torch.cat([state_batch, torch.FloatTensor(state).unsqueeze(0).to(self.device)], dim=0)
                action_batch = torch.cat([action_batch, torch.FloatTensor(action).unsqueeze(0).to(self.device)], dim=0)
                reward_batch = torch.cat([reward_batch, torch.FloatTensor([reward]).to(self.device)], dim=0)
                next_state_batch = torch.cat([next_state_batch, torch.FloatTensor(next_state).unsqueeze(0).to(self.device)], dim=0)
                
                # log_prob이 float인 경우 처리
                if isinstance(log_prob, float):
                    log_prob_tensor = torch.FloatTensor([log_prob]).to(self.device)
                else:
                    log_prob_tensor = torch.FloatTensor(log_prob).unsqueeze(0).to(self.device)
                
                # 차원 확인 및 조정
                if len(old_log_prob_batch.shape) > len(log_prob_tensor.shape):
                    log_prob_tensor = log_prob_tensor.unsqueeze(-1)
                elif len(old_log_prob_batch.shape) < len(log_prob_tensor.shape):
                    old_log_prob_batch = old_log_prob_batch.unsqueeze(-1)
                
                old_log_prob_batch = torch.cat([old_log_prob_batch, log_prob_tensor], dim=0)
                
                # value가 float인 경우 처리 - NumPy 배열로 변환 후 텐서로 변환
                if isinstance(value, float):
                    value_tensor = torch.FloatTensor(np.array([value])).to(self.device)
                else:
                    value_tensor = torch.FloatTensor(np.array(value)).unsqueeze(0).to(self.device)
                
                # 차원 확인 및 조정
                if len(old_value_batch.shape) > len(value_tensor.shape):
                    value_tensor = value_tensor.unsqueeze(-1)
                elif len(old_value_batch.shape) < len(value_tensor.shape):
                    old_value_batch = old_value_batch.unsqueeze(-1)
                
                old_value_batch = torch.cat([old_value_batch, value_tensor], dim=0)
                
                done_batch = torch.cat([done_batch, torch.FloatTensor([done]).to(self.device)], dim=0)
                
                # GAE 재계산 - ppo_gru.py와 동일한 방식으로 next_value 계산
                next_state_parts = [torch.FloatTensor(part).unsqueeze(0).unsqueeze(0).to(self.device) 
                                   for part in np.array_split(next_state, 3)]
                
                # 각 actor_critic에서 next_value 계산
                next_values = [actor_critic(next_state_part)[2] for actor_critic, next_state_part in zip(self.actor_critics, next_state_parts)]
                next_value = sum(w * v for w, v in zip(self.actor_weights, next_values))
                next_value = next_value.squeeze()
                
                # reward와 value를 텐서로 변환 - NumPy 배열로 변환 후 텐서로 변환
                reward_tensor = torch.FloatTensor(np.array([reward])).to(self.device)
                value_tensor = torch.FloatTensor(np.array([value])).to(self.device)
                
                if done:
                    delta = reward_tensor - value_tensor
                    gae = delta
                else:
                    # 그래디언트 요구사항 제거
                    next_value_detached = next_value.detach()
                    delta = reward_tensor + self.gae_gamma * next_value_detached - value_tensor
                    gae = delta + self.gae_gamma * self.gae_lambda * gae
                
                # 차원 불일치 문제 해결
                # gae와 value를 텐서로 유지하고 계산 후 결과를 텐서로 변환
                gae_plus_value = gae + value_tensor
                
                # 차원 확인 및 조정
                if len(returns.shape) > len(gae_plus_value.shape):
                    gae_plus_value = gae_plus_value.unsqueeze(-1)
                elif len(returns.shape) < len(gae_plus_value.shape):
                    returns = returns.unsqueeze(-1)
                
                returns = torch.cat([returns, gae_plus_value], dim=0)
                
                # advantage의 차원도 확인 및 조정
                if len(advantages.shape) > len(gae.shape):
                    gae = gae.unsqueeze(-1)
                elif len(advantages.shape) < len(gae.shape):
                    advantages = advantages.unsqueeze(-1)
                
                advantages = torch.cat([advantages, gae], dim=0)
        
        # 각 actor_critic에 대해 학습 수행
        for actor_critic_idx, (actor_critic, optimizer) in enumerate(zip(self.actor_critics, self.optimizers)):
            for _ in range(self.epochs):
                indices = np.random.permutation(len(state_batch))
                for start_idx in range(0, len(state_batch), batch_size):
                    idx = indices[start_idx:start_idx + batch_size]
                    
                    if len(idx) < batch_size:
                        break
                    
                    # 상태를 3개의 부분으로 나누기
                    state_parts = [torch.FloatTensor(part).unsqueeze(1).to(self.device) 
                                  for part in np.array_split(state_batch[idx].cpu().numpy(), 3, axis=1)]
                    
                    action = action_batch[idx]
                    advantage = advantages[idx]
                    return_ = returns[idx]
                    old_log_prob = old_log_prob_batch[idx]
                    
                    # 현재 actor_critic에 해당하는 상태 부분만 사용
                    state = state_parts[actor_critic_idx]
                    
                    # 값 검증 및 클리핑 추가
                    direction_mean, direction_std, value, attention_weights, action_probs = actor_critic(state)
                    
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
                    surr2 = torch.clamp(ratio, 1-self.current_epsilon, 1+self.current_epsilon) * advantage
                    actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * kl_div
                    
                    # 2. Critic Loss - Huber Loss 사용
                    # 차원 불일치 문제 해결
                    value = value.squeeze(-1)
                    return_ = return_.squeeze(-1)  # return_의 차원도 맞춤
                    
                    # 차원 확인 및 조정
                    if len(value.shape) != len(return_.shape):
                        if len(value.shape) > len(return_.shape):
                            return_ = return_.unsqueeze(-1)
                        else:
                            value = value.unsqueeze(-1)
                    
                    critic_loss = nn.SmoothL1Loss()(value, return_)
                    
                    # 3. 엔트로피와 정규화 손실
                    entropy_loss = -0.01 * action_dist.entropy().mean()
                    
                    # 4. 거래 관련 페널티
                    trading_fee = 0.0005
                    fee_penalty = trading_fee * torch.abs(action).mean()
                    position_change_penalty = 0.01 * torch.abs(action[1:] - action[:-1]).mean()
                    
                    # 5. 리스크 관리 손실
                    max_drawdown_penalty = 0.05 * torch.max(torch.cumsum(torch.min(action, torch.zeros_like(action)), dim=0))
                    volatility_penalty = 0.03 * torch.std(action)
                    
                    # 6. 어텐션 정규화
                    attention_regularization = 0.01 * torch.mean(torch.abs(attention_weights))
                    
                    # 7. 확신 기반 거래 손실
                    confidence_loss = 0.1 * torch.mean((action_probs.max(dim=-1)[0] - 0.8).clamp(min=0))
                    
                    # 전체 손실 함수 조합
                    loss = (
                        2.0 * actor_loss +
                        0.5 * critic_loss * self.current_difficulty +
                        0.05 * entropy_loss +
                        0.0002 * fee_penalty * self.current_difficulty +
                        0.002 * position_change_penalty * self.current_difficulty +
                        0.03 * max_drawdown_penalty * self.current_difficulty +
                        0.02 * volatility_penalty * self.current_difficulty +
                        self.attention_weight * attention_regularization +
                        0.7 * confidence_loss
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
                    optimizer.step()
        
        # actor_weights 업데이트 (성능에 따라 가중치 조정)
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])
            if recent_performance > self.best_performance:
                # 성능이 향상되면 현재 가중치를 유지
                pass
            else:
                # 성능이 저하되면 가중치를 재조정
                # 각 actor_critic의 성능을 평가하고 가중치 조정
                # (실제 구현에서는 각 actor_critic의 성능을 별도로 평가해야 함)
                pass
        
        self.memory.clear()
        return 1
    
    def update_trade_result(self, success):
        """거래 결과 업데이트"""
        if success:
            self.successful_trades += 1

    def save_model(self, data, path):
        torch.save(data, path)

    def checkpoint(self, data, path):
        torch.save(data, path)

    def load_model(self):
        for i, actor_critic in enumerate(self.actor_critics):
            actor_critic.load_state_dict(torch.load(f'models/{self.model_name}_{i}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))

    def load_checkpoint(self, path):
        return torch.load(path)