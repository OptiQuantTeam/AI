import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IndicatorDistribution(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(IndicatorDistribution, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 각 지표별 확률 분포를 위한 네트워크
        self.ema_network = nn.Sequential(
            nn.Linear(3, 32),  # EMA_4, EMA_12, EMA_24 기울기
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        self.stoch_rsi_network = nn.Sequential(
            nn.Linear(1, 16),  # stochRSI 값
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
        
        self.macd_network = nn.Sequential(
            nn.Linear(5, 32),  # MACD, MACD Signal, Cross Signal
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        self.bollinger_network = nn.Sequential(
            nn.Linear(2, 32),  # 상단밴드, 중간밴드, 하단밴드
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 최종 확률 분포를 위한 네트워크
        self.final_network = nn.Sequential(
            nn.Linear(action_dim * 7, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        # 상태 벡터에서 각 지표 추출
        # state는 [batch_size, state_dim] 형태
        batch_size = state.shape[0]
        
        # 1) 기본 확률 정의
        default_probs = torch.tensor([0.0, 1.0, 0.0], device=state.device)
        default_logits = default_probs.unsqueeze(0).expand(batch_size, -1)
        
        # 2) 각 지표별 신호 생성
        combined_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 2-1) EMA 기울기 분석
        ema_4_slope = state[:, 0]  # EMA_4 기울기
        ema_12_slope = state[:, 1]  # EMA_12 기울기
        ema_24_slope = state[:, 2]  # EMA_24 기울기
        
        # EMA 기울기 방향 확인
        ema_4_direction = torch.sign(ema_4_slope)
        ema_12_direction = torch.sign(ema_12_slope)
        ema_24_direction = torch.sign(ema_24_slope)
        
        # EMA 기울기 강도 계산
        ema_4_strength = torch.abs(ema_4_slope)
        ema_12_strength = torch.abs(ema_12_slope)
        ema_24_strength = torch.abs(ema_24_slope)
        
        # EMA 신호 생성
        ema_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 모든 EMA가 상승 추세일 때
        all_up = torch.logical_and(
            torch.logical_and(ema_4_direction > 0, ema_12_direction > 0),
            ema_24_direction > 0
        )
        ema_signal[all_up, 2] = 0.7  # LONG 액션 고정 가중치
        
        # 모든 EMA가 하락 추세일 때
        all_down = torch.logical_and(
            torch.logical_and(ema_4_direction < 0, ema_12_direction < 0),
            ema_24_direction < 0
        )
        ema_signal[all_down, 0] = 0.7  # SHORT 액션 고정 가중치
        
        # 단기 EMA가 중장기 EMA를 상향 돌파할 때
        short_up_break = torch.logical_and(
            ema_4_direction > 0,
            torch.logical_and(
                ema_4_strength > ema_12_strength,
                ema_4_strength > ema_24_strength
            )
        )
        ema_signal[short_up_break, 2] += 0.3  # LONG 액션 추가 고정 가중치
        
        # 단기 EMA가 중장기 EMA를 하향 돌파할 때
        short_down_break = torch.logical_and(
            ema_4_direction < 0,
            torch.logical_and(
                ema_4_strength > ema_12_strength,
                ema_4_strength > ema_24_strength
            )
        )
        ema_signal[short_down_break, 0] += 0.3  # SHORT 액션 추가 고정 가중치
        
        # 중기 EMA가 장기 EMA를 상향 돌파할 때
        mid_up_break = torch.logical_and(
            ema_12_direction > 0,
            torch.logical_and(
                ema_12_strength > ema_4_strength,
                ema_12_strength > ema_24_strength
            )
        )
        ema_signal[mid_up_break, 2] += 0.2  # LONG 액션 추가 고정 가중치
        
        # 중기 EMA가 장기 EMA를 하향 돌파할 때
        mid_down_break = torch.logical_and(
            ema_12_direction < 0,
            torch.logical_and(
                ema_12_strength > ema_4_strength,
                ema_12_strength > ema_24_strength
            )
        )
        ema_signal[mid_down_break, 0] += 0.2  # SHORT 액션 추가 고정 가중치
        
        # 2-2) Stochastic RSI 분석
        stoch_rsi = state[:, 3:4]  # stochRSI 값
        stoch_rsi_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 과매수 상태 (80 이상)
        stoch_rsi_signal[:, 1] = 0.4

        overbought = stoch_rsi.squeeze(-1) > 80
        stoch_rsi_signal[overbought, 0] = 0.6  # SHORT 액션 고정 가중치
        
        # 과매도 상태 (20 이하)
        oversold = stoch_rsi.squeeze(-1) < 20
        stoch_rsi_signal[oversold, 2] = 0.6  # LONG 액션 고정 가중치
        
        # 2-3) MACD 분석
        macd_values = state[:, 4:9]  # MACD, MACD Signal, Cross Signal
        macd_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        macd_signal[:, 1] = 0.4

        # MACD 상향 돌파 시점
        macd_cross_up = macd_values[:, 4] == 1
        macd_signal[macd_cross_up, 2] = 0.5  # LONG 액션 고정 가중치
        
        # MACD 하향 돌파 시점
        macd_cross_down = macd_values[:, 4] == -1
        macd_signal[macd_cross_down, 0] = 0.5  # SHORT 액션 고정 가중치
        
        # MACD Divergence 분석
        bullish_divergence = (macd_values[:, 3] < 0) & (macd_values[:, 2] > 0) & (macd_values[:, 0] > macd_values[:, 1])
        macd_signal[bullish_divergence, 2] += 0.3  # LONG 액션 추가 고정 가중치
        
        bearish_divergence = (macd_values[:, 3] > 0) & (macd_values[:, 2] < 0) & (macd_values[:, 0] < macd_values[:, 1])
        macd_signal[bearish_divergence, 0] += 0.3  # SHORT 액션 추가 고정 가중치
        
        # MACD Trade Signal 분석
        strong_bullish = (macd_values[:, 0] > 0) & (macd_values[:, 2] > 0) & (macd_values[:, 0] > macd_values[:, 1])
        macd_signal[strong_bullish, 2] += 0.4  # LONG 액션 추가 고정 가중치
        
        strong_bearish = (macd_values[:, 0] < 0) & (macd_values[:, 2] < 0) & (macd_values[:, 0] < macd_values[:, 1])
        macd_signal[strong_bearish, 0] += 0.4  # SHORT 액션 추가 고정 가중치
        
        # 2-4) 볼린저 밴드 분석
        bollinger_values = state[:, 9:11]  # 볼린저 밴드 값
        bollinger_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 밴드 폭이 넓어질 때
        band_width_increase = bollinger_values[:, 1] > 0
        bollinger_signal[band_width_increase, 1] = 0.5  # HOLD 액션 고정 가중치
        
        # 밴드 폭이 좁아질 때
        band_width_decrease = bollinger_values[:, 1] < 0
        bollinger_signal[band_width_decrease, 1] = 0.5  # HOLD 액션 고정 가중치
        
        # 2-5) Stochastic RSI와 볼린저 밴드 통합 분석
        stoch_bollinger_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 과매수 + 상단밴드 돌파
        overbought_upper_break = (stoch_rsi.squeeze(-1) > 80) & (bollinger_values[:, 0] > 0)
        stoch_bollinger_signal[overbought_upper_break, 0] = 0.8  # SHORT 신호 고정 가중치
        
        # 과매도 + 하단밴드 돌파
        oversold_lower_break = (stoch_rsi.squeeze(-1) < 20) & (bollinger_values[:, 0] < 0)
        stoch_bollinger_signal[oversold_lower_break, 2] = 0.8  # LONG 신호 고정 가중치
        
        # 중립 구간 + 밴드 폭 증가
        neutral_band_width = (stoch_rsi.squeeze(-1) >= 20) & (stoch_rsi.squeeze(-1) <= 80) & (bollinger_values[:, 1] > 0)
        stoch_bollinger_signal[neutral_band_width, 1] = 0.9  # HOLD 신호 고정 가중치
        
        # 2-6) EMA와 MACD 통합 분석
        ema_macd_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # EMA 상승 + MACD 상향 돌파
        ema_up_macd_up = torch.logical_and(
            torch.logical_and(ema_4_direction > 0, ema_12_direction > 0),
            macd_cross_up
        )
        ema_macd_signal[ema_up_macd_up, 2] = 0.7  # LONG 신호 고정 가중치
        
        # EMA 하락 + MACD 하향 돌파
        ema_down_macd_down = torch.logical_and(
            torch.logical_and(ema_4_direction < 0, ema_12_direction < 0),
            macd_cross_down
        )
        ema_macd_signal[ema_down_macd_down, 0] = 0.7  # SHORT 신호 고정 가중치
        
        # 2-7) MACD와 볼린저 밴드 통합 분석
        macd_bollinger_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # MACD 상향 돌파 + 밴드 폭 감소
        macd_up_band_narrow = torch.logical_and(macd_cross_up, band_width_decrease)
        macd_bollinger_signal[macd_up_band_narrow, 2] = 0.6  # LONG 신호 고정 가중치
        
        # MACD 하향 돌파 + 밴드 폭 감소
        macd_down_band_narrow = torch.logical_and(macd_cross_down, band_width_decrease)
        macd_bollinger_signal[macd_down_band_narrow, 0] = 0.6  # SHORT 신호 고정 가중치
        
        # 3) 모든 신호 통합
        combined_signal = (
            ema_signal + 
            stoch_rsi_signal + 
            macd_signal + 
            bollinger_signal + 
            stoch_bollinger_signal +
            ema_macd_signal +
            macd_bollinger_signal
        )
        
        '''
        # 4) 각 지표의 신경망 출력 계산
        ema_slopes = torch.stack([ema_4_slope, ema_12_slope, ema_24_slope], dim=1)
        ema_probs = F.softmax(self.ema_network(ema_slopes), dim=-1)
        stoch_rsi_probs = F.softmax(self.stoch_rsi_network(stoch_rsi), dim=-1)
        macd_probs = F.softmax(self.macd_network(macd_values), dim=-1)
        bollinger_probs = F.softmax(self.bollinger_network(bollinger_values), dim=-1)
        '''

        # 5) 모든 지표의 확률 분포를 결합
        combined_feats = torch.cat([
            ema_signal,
            stoch_rsi_signal,
            macd_signal,
            bollinger_signal,
            stoch_bollinger_signal,
            ema_macd_signal,
            macd_bollinger_signal,
        ], dim=-1)
        
        # 6) 최종 네트워크에서 나온 로짓에 더하기
        #raw_logits = self.final_network(combined_feats)
        mixed_logits = default_logits + combined_signal
        
        # 7) 모든 값을 더해서 나누어 확률 분포 계산
        #sum_logits = mixed_logits.sum(dim=1, keepdim=True)
        #final_probs = mixed_logits / sum_logits
        final_probs = F.softmax(mixed_logits, dim=-1)

        return final_probs
        