import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IndicatorDistribution(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(IndicatorDistribution, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 하이킨 아시 캔들 분석을 위한 네트워크
        self.ha_network = nn.Sequential(
            nn.Linear(7, 32),  # ha_close, ha_open, ha_high, ha_low, ha_body, ha_lower_wick, ha_upper_wick
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 200 MA 분석을 위한 네트워크
        self.ma_network = nn.Sequential(
            nn.Linear(2, 16),  # ma_200, ma_200_signal
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
        
        # Stochastic RSI 분석을 위한 네트워크
        self.stoch_network = nn.Sequential(
            nn.Linear(2, 16),  # stoch_rsi, stoch_signal
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )

    def forward(self, state):
        batch_size = state.shape[0]
        
        # 1) 기본 확률 정의 - 더 균형있는 분포로 수정
        default_probs = torch.tensor([0.2, 0.6, 0.2], device=state.device)  # SHORT, HOLD, LONG
        default_logits = default_probs.unsqueeze(0).expand(batch_size, -1)
        
        # 2) 하이킨 아시 캔들 분석
        ha_open = state[:, 0]     # ha_open
        ha_close = state[:, 1]    # ha_close
        ha_high = state[:, 2]     # ha_high
        ha_low = state[:, 3]      # ha_low
        ha_body = state[:, 4]     # ha_body
        ha_lower_wick = state[:, 5]  # ha_lower_wick
        ha_upper_wick = state[:, 6]  # ha_upper_wick
        
        # 캔들 패턴 분석 - 더 세분화된 패턴 추가
        is_bullish = ha_close > ha_open  # 양봉
        is_bearish = ha_close < ha_open  # 음봉
        body_size = torch.abs(ha_close - ha_open)  # 몸통 크기
        has_small_lower_wick = ha_lower_wick < 1e-6  # 작은 아래꼬리
        has_small_upper_wick = ha_upper_wick < 1e-6  # 작은 위꼬리
        
        ha_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 강한 상승 신호 - 조건 완화
        strong_bullish = torch.logical_and(
            is_bullish,
            body_size > 0.3  # 몸통 크기 기준 완화
        )
        ha_signal_tensor[strong_bullish, 2] = 0.6  # LONG 가중치 조정
        
        # 강한 하락 신호 - 조건 완화
        strong_bearish = torch.logical_and(
            is_bearish,
            body_size > 0.3  # 몸통 크기 기준 완화
        )
        ha_signal_tensor[strong_bearish, 0] = 0.6  # SHORT 가중치 조정
        
        # 3) 200 MA 분석 - 신호 가중치 조정
        ma_200 = state[:, 7]        # ma_200
        ma_200_signal = state[:, 8]  # ma_200_signal
        
        # 4) Stochastic RSI 분석 - 과매수/과매도 구간 확장
        stoch_rsi = state[:, 9]      # stoch_rsi
        stoch_signal = state[:, 10]   # stoch_signal
        
        stoch_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # Stochastic RSI 과매수/과매도 구간 확장
        overbought = (stoch_rsi > 0.75)  # 75% 이상
        oversold = (stoch_rsi < 0.25)    # 25% 이하
        
        # 기본 Stochastic RSI 신호 - 가중치 증가
        stoch_signal_tensor[oversold, 2] = 0.6  # 과매도 -> LONG
        stoch_signal_tensor[overbought, 0] = 0.6  # 과매수 -> SHORT
        
        # 200선 신호 - 가중치 조정
        ma_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        ma_signal_tensor[ma_200_signal > 0, 2] = 0.5  # 상단 -> LONG
        ma_signal_tensor[ma_200_signal < 0, 0] = 0.5  # 하단 -> SHORT
        
        # 5) 통합 신호 생성 - 진입 조건 완화
        ha_ma_stoch_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 롱 진입 신호 - 2개 이상의 지표가 일치하면 진입
        long_signal = (
            ((ha_signal_tensor[:, 2] > 0) & (ma_200_signal > 0)) |  # 하이킨 아시 + MA
            ((ha_signal_tensor[:, 2] > 0) & (stoch_signal < 0)) |   # 하이킨 아시 + RSI
            ((ma_200_signal > 0) & (stoch_signal < 0))              # MA + RSI
        )
        
        # 숏 진입 신호 - 2개 이상의 지표가 일치하면 진입
        short_signal = (
            ((ha_signal_tensor[:, 0] > 0) & (ma_200_signal < 0)) |  # 하이킨 아시 + MA
            ((ha_signal_tensor[:, 0] > 0) & (stoch_signal > 0)) |   # 하이킨 아시 + RSI
            ((ma_200_signal < 0) & (stoch_signal > 0))              # MA + RSI
        )
        
        ha_ma_stoch_signal[long_signal, 2] = 0.8  # LONG 가중치 조정
        ha_ma_stoch_signal[short_signal, 0] = 0.8  # SHORT 가중치 조정
        
        # 6) 모든 신호 통합 - 가중치 조정
        combined_signal = (
            ha_signal_tensor * 1.2 +  # 하이킨 아시 가중치 증가
            ma_signal_tensor * 1.0 +  # MA 기본 가중치
            stoch_signal_tensor * 1.0 +  # RSI 기본 가중치
            ha_ma_stoch_signal * 1.5  # 통합 신호 가중치 증가
        )
        
        # 7) 최종 확률 분포 계산
        mixed_logits = default_logits + combined_signal
        final_probs = F.softmax(mixed_logits, dim=-1)
        
        return final_probs
        