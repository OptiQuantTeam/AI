import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IndicatorDistribution2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(IndicatorDistribution2, self).__init__()
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
        
        # 1) 기본 확률 정의
        default_probs = torch.tensor([0.0, 1.5, 0.0], device=state.device)   #거래횟수 조정
        default_logits = default_probs.unsqueeze(0).expand(batch_size, -1)
        
        # 2) 하이킨 아시 캔들 분석
        ha_open = state[:, 0]     # ha_open
        ha_close = state[:, 1]    # ha_close
        ha_high = state[:, 2]     # ha_high
        ha_low = state[:, 3]      # ha_low
        ha_body = state[:, 4]     # ha_body
        ha_lower_wick = state[:, 5]  # ha_lower_wick
        ha_upper_wick = state[:, 6]  # ha_upper_wick
        ha_signal = state[:, 7]     # ha_signal
        # 캔들 패턴 분석
        is_bullish = ha_close > ha_open  # 양봉
        is_bearish = ha_close < ha_open  # 음봉
        body_size = torch.abs(ha_close - ha_open)  # 몸통 크기
        has_small_lower_wick = ha_lower_wick < 1e-6  # 작은 아래꼬리
        has_small_upper_wick = ha_upper_wick < 1e-6  # 작은 위꼬리
        
        ha_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 강한 상승 신호
        strong_bullish = torch.logical_and(
            torch.logical_and(is_bullish, has_small_lower_wick),
            body_size > 0.5
        )
        ha_signal_tensor[strong_bullish, 2] = 0.7  # LONG
        
        # 강한 하락 신호
        strong_bearish = torch.logical_and(
            torch.logical_and(is_bearish, has_small_upper_wick),
            body_size > 0.5
        )
        ha_signal_tensor[strong_bearish, 0] = 0.7  # SHORT
        
        # 3) 200 MA 분석
        ma_200 = state[:, 8]        # ma_200
        ma_200_signal = state[:, 9]  # ma_200_signal
        
        # 4) Stochastic RSI 분석
        stoch_rsi = state[:, 10]      # stoch_rsi
        stoch_signal = state[:, 11]   # stoch_signal
        
        stoch_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # Stochastic RSI 과매수/과매도 구간
        overbought = (stoch_rsi > 0.8)  # 80% 이상
        oversold = (stoch_rsi < 0.2)    # 20% 이하
        
        # 기본 Stochastic RSI 신호
        stoch_signal_tensor[oversold, 2] = 0.5  # 과매도 -> LONG
        stoch_signal_tensor[overbought, 0] = 0.5  # 과매수 -> SHORT
        
        # 200선 아래일때 지속 상승/200선 위일대 지속 하락
        ma_signal_tensor = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 기본 200일선 전략
        ma_signal_tensor[ma_200_signal > 0, 2] = 0.6  # 상단 -> LONG
        ma_signal_tensor[ma_200_signal < 0, 0] = 0.6  # 하단 -> SHORT
        
        # 추세 전환 감지
        # 1. 200일선 위에서 하락 반전
        bearish_reversal = (
            (ma_200_signal > 0) &  # 200일선 위
            (ha_signal_tensor[:, 0] > 0) &  # 하이킨 아시 하락
            (overbought)  # Stochastic RSI 과매수
        )
        
        # 2. 200일선 아래에서 상승 반전
        bullish_reversal = (
            (ma_200_signal < 0) &  # 200일선 아래
            (ha_signal_tensor[:, 2] > 0) &  # 하이킨 아시 상승
            (oversold)  # Stochastic RSI 과매도
        )
        
        # 추세 전환 신호 적용
        ma_signal_tensor[bearish_reversal, 0] = 0.8  # 하락 반전 -> SHORT
        ma_signal_tensor[bullish_reversal, 2] = 0.8  # 상승 반전 -> LONG
        
        # 5) 통합 신호 생성
        # 하이킨 아시 + 200 MA + Stochastic RSI 전략
        ha_ma_stoch_signal = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # 롱 진입 신호
        long_signal = (
            (ha_signal_tensor[:, 2] > 0) &  # 하이킨 아시 상승
            (ma_200_signal > 0) &           # 200 MA 상단
            (stoch_signal < 0)              # Stochastic RSI 과매도
        )
        
        # 숏 진입 신호
        short_signal = (
            (ha_signal_tensor[:, 0] > 0) &  # 하이킨 아시 하락
            (ma_200_signal < 0) &           # 200 MA 하단
            (stoch_signal > 0)              # Stochastic RSI 과매수
        )
        
        ha_ma_stoch_signal[long_signal, 2] = 1.0  # LONG
        ha_ma_stoch_signal[short_signal, 0] = 1.0  # SHORT
        
        # 6) 모든 신호 통합
        combined_signal = (
            ha_signal_tensor +
            ma_signal_tensor +
            stoch_signal_tensor +
            ha_ma_stoch_signal
        )
        
        # 7) 최종 확률 분포 계산
        mixed_logits = default_logits + combined_signal

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(mixed_logits)))
        final_probs = F.softmax((mixed_logits + gumbel_noise) / 0.5, dim=-1)

        #final_probs = F.softmax(mixed_logits, dim=-1)
        
        return final_probs
        