import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import network.actorcritic as AC
import datetime

class PPO4:
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
        curriculum_threshold=0.6,
        curriculum_factor=0.95,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_critic = AC.ActorCriticGRU(state_dim, action_dim).to(device)
        
        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic
        self.lr_decay = 0.995
        self.min_lr = 1e-5
        
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.gru.parameters()},
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
        
        self.replay_buffer = deque(maxlen=10000)
        self.replay_batch_size = 32
        self.replay_ratio = 0.2
        
        self.warmup_steps = 1000
        self.attention_weight = 0.0
        self.attention_weight_increment = 0.001
        self.total_steps = 0
        
        self.gae_lambda = 0.95
        self.gae_gamma = 0.99
        
        self.initial_epsilon = epsilon
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995
        self.current_epsilon = epsilon
        
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.01
        
        self.curriculum_threshold = curriculum_threshold
        self.curriculum_factor = curriculum_factor
        self.current_difficulty = 1.0

        self.training = True
        self.exploration_noise = 1.0
        self.min_exploration_noise = 0.1
        self.exploration_decay = 0.995
        
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.current_epsilon = self.epsilon_start
        
        self.confidence_threshold = 0.85
        self.min_trade_interval = 10
        self.last_trade_step = -self.min_trade_interval
        self.confidence_history = deque(maxlen=10)
        self.trade_count = 0
        self.successful_trades = 0
        
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        self.performance_history = []
        self.best_performance = float('-inf')
        self.patience = 20
        self.patience_counter = 0

        # 캔들스틱 패턴 관련 설정
        self.lookback = 10
        self.features_per_candle = 6
        self.pattern_types = 5
        self.pattern_weights = np.array([1.0, 1.5, 1.3, 1.3, 1.4])
        self.pattern_threshold = 0.7
        self.volume_threshold = 1.2

    def _process_candle_state(self, state):
        """
        캔들스틱 패턴 상태 데이터 처리
        """
        candles = state.reshape(self.lookback, self.features_per_candle + self.pattern_types)
        
        processed_features = []
        
        for i in range(self.lookback):
            candle = candles[i]
            features = candle[:self.features_per_candle]
            pattern = candle[self.features_per_candle:]
            
            processed_features.extend(features)
            processed_features.extend(pattern)
        
        # 패턴 강도 계산 및 정규화
        pattern_strengths = []
        for i in range(0, len(processed_features), self.features_per_candle + self.pattern_types):
            pattern = processed_features[i + self.features_per_candle:i + self.features_per_candle + self.pattern_types]
            pattern_strength = np.sum(pattern * self.pattern_weights)
            pattern_strengths.append(pattern_strength)
        
        if pattern_strengths:
            max_strength = max(pattern_strengths)
            if max_strength > 0:
                pattern_strengths = [s / max_strength for s in pattern_strengths]
                processed_features.extend(pattern_strengths)
        
        # 입력 크기를 110로 맞추기 위해 필요한 경우 패딩
        target_size = 110
        if len(processed_features) < target_size:
            processed_features.extend([0.0] * (target_size - len(processed_features)))
        elif len(processed_features) > target_size:
            processed_features = processed_features[:target_size]
        
        return np.array(processed_features, dtype=np.float32)

    def _identify_pattern(self, candle):
        """
        캔들스틱 패턴 식별
        """
        if isinstance(candle, torch.Tensor):
            candle = candle.cpu().numpy()
        
        if len(candle.shape) > 1:
            candle = candle.squeeze()
        
        # 캔들 데이터가 충분한지 확인
        if len(candle) < self.features_per_candle + self.pattern_types:
            return 0, 0.0
            
        # 기본 특성 추출
        features = candle[:self.features_per_candle]
        pattern = candle[self.features_per_candle:self.features_per_candle + self.pattern_types]
        
        # 패턴 강도 계산
        pattern_strength = np.sum(pattern * self.pattern_weights)
        
        if pattern_strength > self.pattern_threshold:
            if pattern[0] > 0.8:  # 해머
                return 1, pattern_strength
            elif pattern[1] > 0.8:  # 강한 상승
                return 2, pattern_strength
            elif pattern[2] > 0.8:  # 슈팅스타
                return -1, pattern_strength
            elif pattern[3] > 0.8:  # 강한 하락
                return -2, pattern_strength
            elif pattern[4] > 0.8:  # 도지
                return 0, pattern_strength
        
        return 0, pattern_strength

    def select_action(self, state):
        processed_state = self._process_candle_state(state)
        state = torch.FloatTensor(processed_state).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.training and np.random.random() < self.current_epsilon:
                random_action = np.random.choice([-1.0, 0.0, 1.0])
                return (
                    np.array([random_action]),
                    0.0,
                    0.0
                )
            
            direction_mean, direction_std, value, _, action_probs = self.actor_critic(state)
            
            confidence = torch.max(F.softmax(action_probs, dim=-1)).item()
            self.confidence_history.append(confidence)
            
            steps_since_last_trade = self.total_steps - self.last_trade_step
            
            current_candle = state[-1, -self.features_per_candle-self.pattern_types:]
            pattern_type, pattern_strength = self._identify_pattern(current_candle)
            
            should_trade = (
                confidence > self.confidence_threshold and
                steps_since_last_trade >= self.min_trade_interval and
                len(self.confidence_history) >= 5 and
                np.mean(list(self.confidence_history)[-5:]) > self.confidence_threshold and
                pattern_strength > self.pattern_threshold
            )
            
            if self.training:
                if should_trade:
                    if pattern_type > 0:
                        action_probs[0] *= 1.2
                    elif pattern_type < 0:
                        action_probs[2] *= 1.2
                    
                    action_probs = F.softmax(action_probs, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action_idx = action_dist.sample()
                    action = action_idx.float() - 1.0
                    log_prob = action_dist.log_prob(action_idx)
                    
                    self.last_trade_step = self.total_steps
                    self.trade_count += 1
                else:
                    action = torch.tensor([0.0]).to(self.device)
                    log_prob = torch.log(torch.tensor([0.5])).to(self.device)
            else:
                action = direction_mean
                log_prob = torch.log(action_probs.max(dim=-1)[0])

            return (
                action.cpu().numpy(),
                value.cpu().numpy()[0],
                log_prob.cpu().numpy()
            )
        
    def store_transition(self, transition):
        state, action, reward, next_state, log_prob, value, done = transition
        
        # 텐서를 CPU로 이동하고 NumPy 배열로 변환
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.cpu().numpy()
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()
        
        transition = (state, action, reward, next_state, log_prob, value, done)
        self.memory.append(transition)
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) == self.n_step:
            n_step_return = 0
            for i, (_, _, reward, _, _, _, _) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * reward
            
            state, action, _, next_state, log_prob, value, done = self.n_step_buffer[0]
            n_step_transition = (state, action, n_step_return, next_state, log_prob, value, done)
            
            self.replay_buffer.append(n_step_transition)
            self.n_step_buffer.popleft()
    
    def update(self, batch_size=64, success_rate=0.0):
        if len(self.memory) < batch_size:
            return 0
        
        current_performance = success_rate
        self.performance_history.append(current_performance)
        
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            return 0
        
        for param_group in self.optimizer.param_groups:
            if 'lr' in param_group:
                param_group['lr'] = max(param_group['lr'] * self.lr_decay, self.min_lr)
        
        self.current_epsilon = max(self.current_epsilon * self.epsilon_decay, self.min_epsilon)
        
        if success_rate > self.curriculum_threshold:
            self.current_difficulty *= self.curriculum_factor
        else:
            self.current_difficulty = min(1.0, self.current_difficulty / self.curriculum_factor)
        
        if self.trade_count > 0:
            trade_success_rate = self.successful_trades / self.trade_count
            if trade_success_rate > 0.7:
                self.confidence_threshold = max(0.75, self.confidence_threshold - 0.005)
            elif trade_success_rate < 0.4:
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.01)
        
        # 배치 준비
        reward_batch = []
        state_batch = []
        action_batch = []
        next_state_batch = []
        log_prob_batch = []
        value_batch = []
        done_batch = []
        
        for transition in self.memory:
            state, action, reward, next_state, log_prob, value, done = transition
            scaled_reward = reward * self.current_difficulty
            reward_batch.append(scaled_reward)
            state_batch.append(state)
            action_batch.append(action)
            next_state_batch.append(next_state)
            log_prob_batch.append(log_prob)
            value_batch.append(value)
            done_batch.append(done)
        
        # 텐서 변환
        state_batch = torch.FloatTensor(np.array(state_batch)).unsqueeze(1).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).unsqueeze(1).to(self.device)
        
        # 로그 확률 처리
        old_log_prob_batch = []
        for log_prob in log_prob_batch:
            if isinstance(log_prob, float):
                old_log_prob_batch.append([log_prob])
            else:
                old_log_prob_batch.append(log_prob)
        old_log_prob_batch = torch.FloatTensor(np.array(old_log_prob_batch)).to(self.device)
        
        # 가치 처리
        old_value_batch = []
        for value in value_batch:
            if isinstance(value, float):
                old_value_batch.append([value])
            else:
                old_value_batch.append(value)
        old_value_batch = torch.FloatTensor(np.array(old_value_batch)).to(self.device)
        
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # GAE 계산
        advantages = []
        returns = []
        gae = 0
        
        with torch.no_grad():
            next_value = self.actor_critic(next_state_batch)[2]
            next_value = next_value.squeeze()
            
            # CUDA 텐서를 이용한 GAE 계산
            for r, v, done, next_v in zip(
                reversed(reward_batch),
                reversed(old_value_batch),
                reversed(done_batch),
                reversed(next_value)
            ):
                # 모든 계산은 동일 장치(device)에서 수행
                if done:
                    delta = r - v
                    gae = delta
                else:
                    next_value_detached = next_v.detach()
                    delta = r + self.gae_gamma * next_value_detached - v
                    gae = delta + self.gae_gamma * self.gae_lambda * gae
                
                # 텐서를 리스트에 추가
                returns.insert(0, (gae + v).clone().reshape(1))
                advantages.insert(0, gae.clone().reshape(1))
        
        # 빈 시퀀스 검사
        if not advantages:
            return 0
            
        # 리스트를 텐서로 변환하여 장치에 배치
        advantages = torch.cat(advantages).to(self.device)
        returns = torch.cat(returns).to(self.device)
        
        if self.use_batch_norm and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 리플레이 버퍼 처리
        if len(self.replay_buffer) > self.replay_batch_size:
            try:
                replay_indices = np.random.choice(
                    len(self.replay_buffer), 
                    size=min(int(batch_size * self.replay_ratio), len(self.replay_buffer)), 
                    replace=False
                )
                
                for idx in replay_indices:
                    state, action, reward, next_state, log_prob, value, done = self.replay_buffer[idx]
                    
                    # 상태 배치에 추가
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                    state_batch = torch.cat([state_batch, state_tensor], dim=0)
                    
                    # 액션 배치에 추가
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    action_batch = torch.cat([action_batch, action_tensor], dim=0)
                    
                    # 보상 배치에 추가
                    reward_tensor = torch.FloatTensor([reward]).to(self.device)
                    reward_batch = torch.cat([reward_batch, reward_tensor], dim=0)
                    
                    # 다음 상태 배치에 추가
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
                    next_state_batch = torch.cat([next_state_batch, next_state_tensor], dim=0)
                    
                    # 로그 확률 처리
                    if isinstance(log_prob, float):
                        log_prob_tensor = torch.FloatTensor([log_prob]).to(self.device)
                    else:
                        log_prob_tensor = torch.FloatTensor(log_prob).unsqueeze(0).to(self.device)
                    
                    # 차원 맞추기
                    if log_prob_tensor.dim() == 1 and old_log_prob_batch.dim() == 2:
                        log_prob_tensor = log_prob_tensor.unsqueeze(-1)
                    
                    old_log_prob_batch = torch.cat([old_log_prob_batch, log_prob_tensor], dim=0)
                    
                    # 가치 처리
                    if isinstance(value, float):
                        value_tensor = torch.FloatTensor([value]).to(self.device)
                    else:
                        value_tensor = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                    
                    # 차원 맞추기
                    if value_tensor.dim() == 1 and old_value_batch.dim() == 2:
                        value_tensor = value_tensor.unsqueeze(-1)
                    
                    old_value_batch = torch.cat([old_value_batch, value_tensor], dim=0)
                    
                    # 완료 배치에 추가
                    done_tensor = torch.FloatTensor([done]).to(self.device)
                    done_batch = torch.cat([done_batch, done_tensor], dim=0)
                    
                    # 다음 가치 계산
                    with torch.no_grad():
                        next_value = self.actor_critic(next_state_tensor)[2]
                        next_value = next_value.squeeze()
                        
                        # GAE 계산
                        # 모든 값이 동일한 디바이스에 있는지 확인
                        r_tensor = reward_tensor.to(self.device)
                        v_tensor = value_tensor.to(self.device)
                        next_v_tensor = next_value.to(self.device)
                        
                        if done:
                            delta = r_tensor - v_tensor
                            gae = delta
                        else:
                            next_value_detached = next_v_tensor.detach()
                            delta = r_tensor + self.gae_gamma * next_value_detached - v_tensor
                            gae = delta + self.gae_gamma * self.gae_lambda * gae
                        
                        # 리턴과 어드밴티지 추가 (모두 동일 디바이스에 있는지 확인)
                        returns = torch.cat([returns, (gae + v_tensor).view(1).to(self.device)], dim=0)
                        advantages = torch.cat([advantages, gae.view(1).to(self.device)], dim=0)
            except (ValueError, RuntimeError) as e:
                # 리플레이 버퍼 처리 오류는 무시하고 계속 진행
                pass
        
        # 훈련 반복
        for _ in range(self.epochs):
            indices = np.random.permutation(len(state_batch))
            for start_idx in range(0, len(state_batch), batch_size):
                end_idx = min(start_idx + batch_size, len(state_batch))
                idx = indices[start_idx:end_idx]
                
                if len(idx) < 2:  # 최소 배치 크기 확인
                    continue
                
                state = state_batch[idx]
                action = action_batch[idx]
                advantage = advantages[idx]
                return_ = returns[idx]
                old_log_prob = old_log_prob_batch[idx]
                
                direction_mean, direction_std, value, attention_weights, action_probs = self.actor_critic(state)
                
                if torch.isnan(direction_mean).any() or torch.isnan(direction_std).any():
                    continue
                
                try:
                    action_probs = F.softmax(action_probs, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    
                    action_idx = (action + 1).long()
                    new_log_prob = action_dist.log_prob(action_idx)
                except ValueError as e:
                    continue
                
                # 차원 맞추기
                if new_log_prob.dim() < old_log_prob.dim():
                    new_log_prob = new_log_prob.unsqueeze(-1)
                if old_log_prob.dim() < new_log_prob.dim():
                    old_log_prob = old_log_prob.unsqueeze(-1)
                
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # 차원 맞추기
                if ratio.dim() > advantage.dim():
                    advantage = advantage.unsqueeze(-1)
                
                kl_div = 0.5 * ((new_log_prob - old_log_prob) ** 2).mean()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.current_epsilon, 1+self.current_epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * kl_div
                
                value = value.squeeze(-1)
                critic_loss = nn.SmoothL1Loss()(value, return_)
                
                entropy_loss = -0.01 * action_dist.entropy().mean()
                
                trading_fee = 0.0005
                fee_penalty = trading_fee * torch.abs(action).mean()
                position_change_penalty = 0.01 * torch.abs(action[1:] - action[:-1]).mean() if len(action) > 1 else torch.tensor(0.0).to(self.device)
                
                max_drawdown_penalty = 0.05 * torch.max(torch.cumsum(torch.min(action, torch.zeros_like(action)), dim=0)) if len(action) > 0 else torch.tensor(0.0).to(self.device)
                volatility_penalty = 0.03 * torch.std(action) if len(action) > 1 else torch.tensor(0.0).to(self.device)
                
                attention_regularization = 0.01 * torch.mean(torch.abs(attention_weights))
                
                confidence_loss = 0.1 * torch.mean((action_probs.max(dim=-1)[0] - 0.8).clamp(min=0))
                
                pattern_loss = 0.05 * torch.mean((action_probs - 0.5).abs())
                
                loss = (
                    2.0 * actor_loss +
                    0.5 * critic_loss * self.current_difficulty +
                    0.05 * entropy_loss +
                    0.0002 * fee_penalty * self.current_difficulty +
                    0.002 * position_change_penalty * self.current_difficulty +
                    0.03 * max_drawdown_penalty * self.current_difficulty +
                    0.02 * volatility_penalty * self.current_difficulty +
                    self.attention_weight * attention_regularization +
                    0.7 * confidence_loss +
                    0.3 * pattern_loss
                )
                
                # 그래디언트 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        self.memory.clear()
        return 1
    
    def update_trade_result(self, success):
        if success:
            self.successful_trades += 1

    def save_model(self, data, path):
        torch.save(data, path)

    def checkpoint(self, data, path):
        torch.save(data, path)

    def load_model(self):
        self.actor_critic.load_state_dict(torch.load(f'models/{self.model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))

    def load_checkpoint(self, path):
        return torch.load(path)
