import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCriticGRU(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2):
        super(ActorCriticGRU, self).__init__()
        
        # GRU 레이어
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 특징 추출 레이어
        self.feature_extraction = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor 네트워크 (정책 네트워크)
        self.actor_direction = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 방향 표준편차 파라미터
        self.actor_direction_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic 네트워크 (가치 네트워크)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
            
    def forward(self, state, hidden=None):
        # state shape: (batch_size, sequence_length, state_dim)
        batch_size = state.size(0)
        
        # GRU 레이어 통과
        gru_out, hidden = self.gru(state, hidden)
        # 마지막 시퀀스의 출력만 사용
        gru_out = gru_out[:, -1, :]  # shape: (batch_size, hidden_dim)
        
        # 특징 추출
        features = self.feature_extraction(gru_out)
        
        # Actor 네트워크
        action_logits = self.actor_direction(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 각 행동에 대한 값 매핑 [-1, 0, 1]
        action_values = torch.tensor([-1.0, 0.0, 1.0]).to(state.device)
        
        # 가장 높은 확률을 가진 행동의 값을 direction_mean으로 사용
        max_prob_indices = torch.argmax(action_probs, dim=-1)
        direction_mean = action_values[max_prob_indices]
        
        # 방향 표준편차 계산
        direction_std = F.softplus(self.actor_direction_std) + 1e-6
        
        # Critic 네트워크
        value = self.critic(features)
        
        # 어텐션 가중치 대신 GRU의 마지막 은닉 상태를 반환
        attention_weights = gru_out
        #print(f'direction_mean: {direction_mean}, direction_std: {direction_std}, value: {value}, attention_weights: {attention_weights}, action_probs: {action_probs}', flush=True)
        return direction_mean, direction_std, value, attention_weights, action_probs

class ActorCriticGRUMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []