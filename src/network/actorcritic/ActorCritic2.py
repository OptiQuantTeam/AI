import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + self.layers(x)

class PriceNormalization(nn.Module):
    def __init__(self, num_features, window_size=20):
        super(PriceNormalization, self).__init__()
        self.window_size = window_size
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        self.momentum = 0.1
        self.eps = 1e-6  # 더 큰 epsilon 값 사용

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            std = x.std(dim=0) + self.eps  # epsilon 추가
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std
            x_norm = (x - mean) / std
        else:
            x_norm = (x - self.running_mean) / (self.running_std + self.eps)
        return torch.clamp(x_norm, -3, 3)

class VolumeNormalization(nn.Module):
    def __init__(self, num_features, window_size=20):
        super(VolumeNormalization, self).__init__()
        self.window_size = window_size
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_std', torch.ones(num_features))
        self.momentum = 0.1
        self.eps = 1e-6

    def forward(self, x):
        # 음수 값 처리
        x = torch.abs(x)
        
        # 로그 변환 (0 값 처리)
        x = torch.log1p(x + self.eps)
        
        if self.training:
            mean = x.mean(dim=0)
            std = x.std(dim=0) + self.eps
            
            # NaN 체크 및 처리
            mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
            std = torch.where(torch.isnan(std), torch.ones_like(std), std)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std
            
            x_norm = (x - mean) / std
        else:
            x_norm = (x - self.running_mean) / (self.running_std + self.eps)
        
        # NaN 체크 및 처리
        x_norm = torch.where(torch.isnan(x_norm), torch.zeros_like(x_norm), x_norm)
        
        return torch.clamp(x_norm, -3, 3)

class TechnicalIndicatorNormalization(nn.Module):
    def __init__(self, num_features):
        super(TechnicalIndicatorNormalization, self).__init__()
        self.register_buffer('min_val', torch.zeros(num_features))
        self.register_buffer('max_val', torch.ones(num_features))
        self.momentum = 0.1
        self.eps = 1e-6  # 더 큰 epsilon 값 사용

    def forward(self, x):
        if self.training:
            min_val = x.min(dim=0)[0]
            max_val = x.max(dim=0)[0]
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val
            x_norm = (x - min_val) / (max_val - min_val + self.eps)
        else:
            x_norm = (x - self.min_val) / (self.max_val - self.min_val + self.eps)
        return torch.clamp(x_norm, 0, 1)

class SignalNormalization(nn.Module):
    def forward(self, x):
        return torch.clamp(x, -1, 1)

class ActorCritic2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic2, self).__init__()
        
        # 가중치 초기화 함수
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if 'feature_extraction' in str(m):
                    # 특징 추출 레이어: Kaiming 초기화
                    nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.uniform_(m.bias, -0.1, 0.1)
                elif 'actor_direction' in str(m):
                    # 액터 레이어: 균형잡힌 초기화
                    if 'Linear' in str(m) and str(m.weight.shape[1]) == str(action_dim):
                        # 마지막 레이어: 균등 분포로 초기화
                        nn.init.uniform_(m.weight, -0.01, 0.01)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)  # 바이어스를 0으로 초기화하여 균등한 확률 분포
                    else:
                        # 중간 레이어: Xavier 초기화
                        nn.init.xavier_uniform_(m.weight, gain=0.01)
                        if m.bias is not None:
                            nn.init.uniform_(m.bias, -0.05, 0.05)
                elif 'critic' in str(m):
                    # 크리틱 레이어: Xavier 초기화
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias, -0.2, 0.2)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm 초기화
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # 정규화 레이어 정의
        # [1,4,12,16,17,18,19,24,25,26,29] 인덱스에 맞춰 정규화 레이어 설정
        self.price_norm = PriceNormalization(2)  # 인덱스 1,4 (가격 관련)
        self.volume_norm = VolumeNormalization(1)  # 인덱스 4 (거래량)
        self.technical_norm = TechnicalIndicatorNormalization(4)  # 인덱스 16,17,18,19 (기술적 지표)
        self.signal_norm = SignalNormalization()  # 인덱스 24,25,26,29 (신호 지표)
        
        # 공통 특징 추출 레이어
        self.feature_extraction = nn.Sequential(
            nn.Linear(11, 256),  # 입력 차원을 11로 수정
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )
        
        # 첫 번째 액터 네트워크
        self.actor_direction = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(64),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, action_dim),
            nn.LeakyReLU()
        )
        
        # 첫 번째 크리틱 네트워크
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(64),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # Temperature 파라미터
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # 가중치 초기화 적용
        self.apply(init_weights)
    
    def forward(self, state):
        # 입력 상태를 각 그룹별로 분리
        price_data = state[:, :2]  # 인덱스 1,4 (가격 관련)
        volume_data = state[:, 2:3]  # 인덱스 4 (거래량)
        technical_data = state[:, 3:7]  # 인덱스 16,17,18,19 (기술적 지표)
        signal_data = state[:, 7:]  # 인덱스 24,25,26,29 (신호 지표)
        
        # 각 그룹별 정규화
        normalized_price = self.price_norm(price_data)
        normalized_volume = self.volume_norm(volume_data)
        normalized_technical = self.technical_norm(technical_data)
        normalized_signal = self.signal_norm(signal_data)
        
        # 정규화된 데이터 결합
        normalized_state = torch.cat([
            normalized_price,
            normalized_volume,
            normalized_technical,
            normalized_signal
        ], dim=1)
        #print(f'normalized_state: {normalized_state}')
        # 특징 추출
        features = self.feature_extraction(normalized_state)
        
        # 첫 번째 액터: 행동 분포
        action_logits = self.actor_direction(features)
        scaled_logits = action_logits / self.temperature
        action_probs = F.softmax(scaled_logits, dim=-1)

        # 첫 번째 크리틱: 상태 가치
        value = self.critic(features)
        
        return value, action_probs, action_logits