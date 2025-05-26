import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 가중치 초기화 함수
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 더 극단적인 초기화
                gain = 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 공통 특징 추출 레이어 - 단순화된 구조
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 액터 네트워크 (정책) - 포지션 방향
        self.actor_direction = nn.Sequential(
            nn.Linear(16, action_dim)
        )
        
        # 액터의 표준편차 파라미터
        self.actor_direction_std = nn.Parameter(torch.ones(1) * 2.0)  # 더 큰 초기값
        
        # 크리틱 네트워크 (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(16, 1)
        )
        
        # 가중치 초기화 적용
        self.apply(init_weights)
        
    def forward(self, state):
        # 특징 추출
        features = self.feature_extraction(state)
        
        # 액터: 행동 분포
        action_logits = self.actor_direction(features)
        
        # 직접적인 softmax 적용
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 크리틱: 상태 가치
        value = self.critic(features)
        
        return value, action_probs, action_logits