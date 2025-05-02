import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """시퀀스 데이터를 위한 위치 인코딩 모듈"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class ActorCriticTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1, max_seq_len=100):
        super(ActorCriticTransformer, self).__init__()
        
        # 시퀀스 길이 파라미터
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 입력 임베딩
        self.input_embedding = nn.Linear(state_dim, d_model)
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer 인코더 레이어
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dim_feedforward=d_model*4, 
                                                  dropout=dropout, 
                                                  activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 자기 주의 모듈 (Multi-Head Attention)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        
        # 액터 네트워크 (정책) - 포지션 방향
        self.actor_direction = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3개 액션: LONG, FLAT, SHORT
        )
        
        # 행동의 표준편차
        self.actor_direction_std = nn.Parameter(torch.zeros(1))
        
        # 크리틱 네트워크 (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 출력 헤드 (추가적인 정보 제공)
        self.action_probs = nn.Linear(d_model, 3)  # LONG, FLAT, SHORT에 대한 확률
        
    def _reshape_state(self, state):
        """상태 텐서를 시퀀셜 포맷으로 변환"""
        batch_size = state.size(0)
        
        # 시계열 데이터 또는 1D 벡터에 따라 처리 방식 변경
        if len(state.shape) == 3:  # [batch_size, seq_len, features]
            return state
        elif len(state.shape) == 2:  # [batch_size, features]
            # 특징 차원을 시퀀스와 특징으로 재구성
            seq_len = self.max_seq_len
            if state.size(1) % seq_len == 0:
                feature_dim = state.size(1) // seq_len
                return state.view(batch_size, seq_len, feature_dim)
            else:
                # 시퀀스로 해석할 수 없는 경우 임베딩 후 시퀀스 길이 1로 처리
                return state.unsqueeze(1)
        
    def forward(self, state):
        # 상태 텐서 재구성
        batch_size = state.size(0)
        reshaped_state = self._reshape_state(state)
        
        # 입력 임베딩 및 위치 인코딩
        x = self.input_embedding(reshaped_state)
        x = self.pos_encoder(x)
        
        # Transformer 인코더 처리 (B, S, D)에서 (S, B, D)로 변환 후 다시 (B, S, D)로
        x = x.permute(1, 0, 2)  # [S, B, D]
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.permute(1, 0, 2)  # [B, S, D]
        
        # 자기 주의 메커니즘
        query = transformer_output.mean(dim=1, keepdim=True)  # 전체 시퀀스의 평균을 쿼리로 사용
        attn_output, attention_weights = self.multihead_attn(
            query.permute(1, 0, 2),
            transformer_output.permute(1, 0, 2),
            transformer_output.permute(1, 0, 2)
        )
        attn_output = attn_output.permute(1, 0, 2).squeeze(1)  # [B, D]
        
        # 액터: 행동 분포의 평균과 표준편차
        action_logits = self.actor_direction(attn_output)
        direction_mean = torch.tanh(action_logits[:, 0].unsqueeze(-1))  # -1 ~ 1 범위로 제한
        direction_std = torch.exp(self.actor_direction_std).expand_as(direction_mean)
        
        # 크리틱: 상태 가치
        value = self.critic(attn_output)
        
        # 추가 정보: 행동 확률
        action_probs = self.action_probs(attn_output)
        
        return direction_mean, direction_std, value, attention_weights, action_probs