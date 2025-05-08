import torch.nn as nn
import torch.nn.functional as F

class RNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # 정책 헤드
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 가치 헤드
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 정책과 가치 계산
        policy = self.policy_head(lstm_out)
        value = self.value_head(lstm_out)
        
        return policy, value, hidden