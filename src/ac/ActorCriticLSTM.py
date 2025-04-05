import torch
import torch.nn as nn

class ActorCriticLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lstm_layers=1):
        super(ActorCriticLSTM, self).__init__()
        
        # LSTM 레이어
        self.feature_extraction = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # 액터 네트워크 (정책) - 포지션 방향
        self.actor_direction = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # -1 ~ 1 범위로 제한
        )
        
        # 행동의 표준편차
        self.actor_direction_std = nn.Parameter(torch.zeros(1))
        
        # 크리틱 네트워크 (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # LSTM의 hidden state와 cell state를ㅇ
        
    def init_hidden(self, batch_size=1):
        # LSTM의 상태 초기화
        return (torch.zeros(1, batch_size, self.feature_extraction.hidden_size),
                torch.zeros(1, batch_size, self.feature_extraction.hidden_size))
    
    def forward(self, x, hidden=None):
        # 입력 형태 변환 (if needed)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # 배치 차원 추가
            
        # LSTM 통과
        features, self.hidden = self.feature_extraction(x, hidden)
        
        # 액터: 행동 분포의 평균과 표준편차
        direction_mean = self.actor_direction(features)
        direction_std = torch.exp(self.actor_direction_std).expand_as(direction_mean)
        
        # 크리틱: 상태 가치
        value = self.critic(features)
        
        return direction_mean, direction_std, value