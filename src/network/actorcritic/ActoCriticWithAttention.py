import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # d_model을 num_heads로 나눌 수 있도록 조정
        self.d_model = (d_model // num_heads) * num_heads
        self.num_heads = num_heads
        self.head_dim = self.d_model // num_heads
        
        # 입력 차원을 d_model로 매핑하는 레이어 추가
        self.input_projection = nn.Linear(d_model, self.d_model)
        
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, d_model)
        
    def forward(self, x):
        # 입력 차원: [batch_size, seq_len, feature_dim]
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 입력을 적절한 차원으로 투영
        x = self.input_projection(x)
        
        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Concatenate heads and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out(context)
        
        return output, attention

class ActorCriticWithAttention(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=4):
        super().__init__()
        self.attention = MultiHeadAttention(state_dim, num_heads)
        
        self.feature_extraction = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.actor_direction = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        self.actor_direction_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state):
        # 어텐션 적용
        attended_state, attention_weights = self.attention(state)
        # 배치의 첫 번째 시퀀스만 사용
        features = self.feature_extraction(attended_state[:, 0])
        
        # actor_direction 출력을 softmax로 변환
        action_probs = F.softmax(self.actor_direction(features), dim=-1)
        
        # 각 행동에 대한 값 매핑 [-1, 0, 1]
        action_values = torch.tensor([-1.0, 0.0, 1.0]).to(state.device)
        
        # 가장 높은 확률을 가진 행동의 값을 direction_mean으로 사용
        max_prob_indices = torch.argmax(action_probs, dim=-1)
        direction_mean = action_values[max_prob_indices]
        
        direction_std = F.softplus(self.actor_direction_std) + 1e-6
        value = self.critic(features)
        
        return direction_mean, direction_std, value, attention_weights, action_probs  # action_probs도 반환