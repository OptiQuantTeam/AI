import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import datetime
import network.actorcritic as AC

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
        #self.actor_critic = ActorCriticLSTM(state_dim, action_dim, hidden_dim=128, lstm_layers=2).to(device)
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.feature_extraction.parameters()},
            {'params': self.actor_critic.actor_direction.parameters()},
            {'params': self.actor_critic.actor_direction_std},            
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic}
        ], lr=lr_actor)
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        self.memory = deque()
        
        # 이전 상태 저장 변수 추가
        self.prev_sma_5 = None
        self.prev_sma_20 = None
        self.prev_sma_60 = None
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            direction_mean, direction_std, value, action_probs, action_logits = self.actor_critic(state)
            
            # 기술적 지표 추출
            close_price = state[0, 0].item()
            volume = state[0, 4].item()
            sma_5 = state[0, 5].item()
            sma_20 = state[0, 6].item()
            sma_60 = state[0, 7].item()
            rsi = state[0, 8].item()
            macd = state[0, 9].item()
            macd_signal = state[0, 10].item()
            bb_upper = state[0, 12].item()
            bb_lower = state[0, 14].item()
            price_change = state[0, 15].item()  # 가격 변화율
            
            # 최소 확률 설정
            min_prob = 0.1  # 최소 확률 10%
            
            # 기술적 지표 기반 액션 조정
            # 1. RSI 기반 조정
            if rsi > 70:  # 과매수
                action_probs[0, 2] = max(action_probs[0, 2] * 0.5, min_prob)  # LONG 액션 확률 감소
                action_probs[0, 0] = min(action_probs[0, 0] * 1.2, 1.0 - 2 * min_prob)  # SHORT 액션 확률 증가
            elif rsi < 30:  # 과매도
                action_probs[0, 0] = max(action_probs[0, 0] * 0.5, min_prob)  # SHORT 액션 확률 감소
                action_probs[0, 2] = min(action_probs[0, 2] * 1.2, 1.0 - 2 * min_prob)  # LONG 액션 확률 증가
            
            # 2. MACD 기반 조정
            if macd > macd_signal:  # 상승 추세
                action_probs[0, 2] = min(action_probs[0, 2] * 1.2, 1.0 - 2 * min_prob)  # LONG 액션 확률 증가
                action_probs[0, 0] = max(action_probs[0, 0] * 0.8, min_prob)  # SHORT 액션 확률 감소
            else:  # 하락 추세
                action_probs[0, 0] = min(action_probs[0, 0] * 1.2, 1.0 - 2 * min_prob)  # SHORT 액션 확률 증가
                action_probs[0, 2] = max(action_probs[0, 2] * 0.8, min_prob)  # LONG 액션 확률 감소
            
            # 3. 볼린저 밴드 기반 조정
            if close_price > bb_upper:  # 상단 밴드 돌파
                action_probs[0, 2] = max(action_probs[0, 2] * 0.7, min_prob)  # LONG 액션 확률 감소
                action_probs[0, 0] = min(action_probs[0, 0] * 1.3, 1.0 - 2 * min_prob)  # SHORT 액션 확률 증가
            elif close_price < bb_lower:  # 하단 밴드 돌파
                action_probs[0, 0] = max(action_probs[0, 0] * 0.7, min_prob)  # SHORT 액션 확률 감소
                action_probs[0, 2] = min(action_probs[0, 2] * 1.3, 1.0 - 2 * min_prob)  # LONG 액션 확률 증가
            
            # 4. 이동평균선 기반 조정
            if sma_5 > sma_20:  # 골든 크로스
                action_probs[0, 2] = min(action_probs[0, 2] * 1.2, 1.0 - 2 * min_prob)  # LONG 액션 확률 증가
            elif sma_5 < sma_20:  # 데드 크로스
                action_probs[0, 0] = min(action_probs[0, 0] * 1.2, 1.0 - 2 * min_prob)  # SHORT 액션 확률 증가
            
            # 5. 가격 변화율 기반 조정
            if abs(price_change) > 0.02:  # 큰 가격 변동
                action_probs[0, 1] = min(action_probs[0, 1] * 1.5, 1.0 - 2 * min_prob)  # FLAT 액션 확률 증가
            
            # 6. SMA 기울기와 관계에 따른 추가 조정
            if self.prev_sma_5 is not None and self.prev_sma_20 is not None and self.prev_sma_60 is not None:
                sma_5_slope = (sma_5 - self.prev_sma_5) / sma_5
                sma_20_slope = (sma_20 - self.prev_sma_20) / sma_20
                sma_60_slope = (sma_60 - self.prev_sma_60) / sma_60
                
                # 상승 추세 확인 (모든 SMA가 상승)
                if sma_5_slope > 0 and sma_20_slope > 0 and sma_60_slope > 0:
                    action_probs[0, 2] = min(action_probs[0, 2] * 1.3, 1.0 - 2 * min_prob)  # LONG 액션 확률 증가
                    action_probs[0, 0] = max(action_probs[0, 0] * 0.7, min_prob)  # SHORT 액션 확률 감소
                
                # 하락 추세 확인 (모든 SMA가 하락)
                elif sma_5_slope < 0 and sma_20_slope < 0 and sma_60_slope < 0:
                    action_probs[0, 0] = min(action_probs[0, 0] * 1.3, 1.0 - 2 * min_prob)  # SHORT 액션 확률 증가
                    action_probs[0, 2] = max(action_probs[0, 2] * 0.7, min_prob)  # LONG 액션 확률 감소
            
            # SMA 정렬 상태 확인
            if sma_5 > sma_20 > sma_60:  # 강한 상승 추세
                action_probs[0, 2] = min(action_probs[0, 2] * 1.4, 1.0 - 2 * min_prob)  # LONG 액션 확률 증가
                action_probs[0, 0] = max(action_probs[0, 0] * 0.6, min_prob)  # SHORT 액션 확률 감소
            elif sma_5 < sma_20 < sma_60:  # 강한 하락 추세
                action_probs[0, 0] = min(action_probs[0, 0] * 1.4, 1.0 - 2 * min_prob)  # SHORT 액션 확률 증가
                action_probs[0, 2] = max(action_probs[0, 2] * 0.6, min_prob)  # LONG 액션 확률 감소
            elif sma_5 < sma_20 and sma_20 > sma_60:  # 혼조 상태
                action_probs[0, 1] = min(action_probs[0, 1] * 1.3, 1.0 - 2 * min_prob)  # FLAT 액션 확률 증가
                action_probs[0, 0] = max(action_probs[0, 0] * 0.8, min_prob)  # SHORT 액션 확률 감소
                action_probs[0, 2] = max(action_probs[0, 2] * 0.8, min_prob)  # LONG 액션 확률 감소
            
            # 현재 상태를 이전 상태로 저장
            self.prev_sma_5 = sma_5
            self.prev_sma_20 = sma_20
            self.prev_sma_60 = sma_60
            
            # 최소 확률 보장
            action_probs[0, 1] = max(action_probs[0, 1] + 0.9, min_prob)  # flat 액션(인덱스 1)의 확률을 1.5배 증가
            action_probs[0, 0] = max(action_probs[0, 0] * 0.5, min_prob)  # 매도 액션(인덱스 0)의 확률 감소
            action_probs[0, 2] = max(action_probs[0, 2] * 0.5, min_prob)  # 매수 액션(인덱스 2)의 확률 감소
            
            # 확률 정규화
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
            
            # Categorical 분포에서 액션 샘플링 (0, 1, 2)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            
            # 액션을 -1, 0, 1로 변환
            action = action_idx.float() - 1.0
            
            # 로그 확률 계산
            log_prob = action_dist.log_prob(action_idx)
        
        return (
            action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0]
        )
        
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self, batch_size=64, success_rate=None):
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
            next_value = self.actor_critic(next_state_batch)[2]  # value는 2번째 반환값
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
            # 미니배치 생성
            indices = np.random.permutation(len(state_batch))
            for start_idx in range(0, len(state_batch), batch_size):
                idx = indices[start_idx:start_idx + batch_size]
                
                if len(idx) < batch_size:
                    break
                
                # 현재 미니배치
                state = state_batch[idx]
                action = action_batch[idx]
                advantage = advantages[idx]
                return_ = returns[idx]
                old_log_prob = old_log_prob_batch[idx]
                
                # 현재 정책의 행동 분포
                direction_mean, direction_std, value, action_probs, action_logits = self.actor_critic(state)
                
                
                action_dist = torch.distributions.Categorical(action_probs)
                action_idx = action_dist.sample()
                action = action_idx.float() - 1.0

                new_log_prob = action_dist.log_prob(action_idx)
                
                # PPO 비율 계산
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # 수정된 손실 함수들
                
                # 1. Actor Loss - KL 페널티 추가
                kl_div = 0.5 * ((new_log_prob - old_log_prob) ** 2).mean()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * kl_div
                
                # 2. Critic Loss - Huber Loss 사용
                value = value.squeeze(-1)   # 차원 축소 추가
                critic_loss = nn.SmoothL1Loss()(value, return_)
                
                # 3. 엔트로피와 정규화 손실
                entropy_loss = -0.01 * action_dist.entropy().mean()
                std_loss = 0.01 * (direction_std ** 2).mean()  # 표준편차 페널티
                
                # 4. 거래 관련 페널티
                trading_fee = 0.0003
                fee_penalty = trading_fee * torch.abs(action).mean()
                position_change_penalty = 0.005 * torch.abs(action[1:] - action[:-1]).mean()  # 급격한 포지션 변화 페널티
                
                # 5. 리스크 관리 손실
                max_drawdown_penalty = 0.02 * torch.max(torch.cumsum(torch.min(action, torch.zeros_like(action)), dim=0))
                volatility_penalty = 0.015 * torch.std(action)
                
                # 전체 손실 함수 조합
                loss = (
                    actor_loss +
                    0.5 * critic_loss +
                    entropy_loss +
                    std_loss +
                    fee_penalty +
                    position_change_penalty +
                    max_drawdown_penalty +
                    volatility_penalty
                )
                
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