import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import os

@dataclass
class CandleState:
    """캔들스틱 상태 데이터 클래스"""
    body_size: float  # 몸통 크기
    upper_shadow: float  # 윗꼬리 길이
    lower_shadow: float  # 아랫꼬리 길이
    is_bullish: bool  # 상승 캔들 여부
    pattern: str  # 패턴 이름
    volume: float  # 거래량
    price_change: float  # 가격 변화율

class CandleStateGenerator:
    def __init__(self, data: pd.DataFrame, lookback: int = 10):
        """
        캔들스틱 상태 데이터 생성기
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV 데이터 (Open, High, Low, Close, Volume)
        lookback : int
            상태 생성에 사용할 과거 봉 수
        """
        self.data = data
        self.lookback = lookback
        
    def _calculate_body(self, row: pd.Series) -> float:
        """캔들 몸통 크기 계산"""
        return abs(row['Close'] - row['Open'])
    
    def _calculate_upper_shadow(self, row: pd.Series) -> float:
        """윗꼬리 길이 계산"""
        return row['High'] - max(row['Open'], row['Close'])
    
    def _calculate_lower_shadow(self, row: pd.Series) -> float:
        """아랫꼬리 길이 계산"""
        return min(row['Open'], row['Close']) - row['Low']
    
    def _is_bullish(self, row: pd.Series) -> bool:
        """상승 캔들 여부 확인"""
        return row['Close'] > row['Open']
    
    def _identify_pattern(self, current: pd.Series, previous: pd.Series = None) -> str:
        """캔들스틱 패턴 식별"""
        body = self._calculate_body(current)
        upper = self._calculate_upper_shadow(current)
        lower = self._calculate_lower_shadow(current)
        
        # 도지 패턴
        if body <= (upper + lower) * 0.1:
            return 'doji'
            
        # 해머 패턴
        if (lower >= 2 * body) and (upper <= body * 0.1):
            return 'hammer'
            
        # 샛별 패턴
        if (upper >= 2 * body) and (lower <= body * 0.1):
            return 'shooting_star'
            
        # 잉여 패턴
        if previous is not None:
            if self._is_bullish(current) != self._is_bullish(previous):
                current_body = self._calculate_body(current)
                previous_body = self._calculate_body(previous)
                if current_body > previous_body:
                    return 'bullish_engulfing' if self._is_bullish(current) else 'bearish_engulfing'
        
        return 'normal'
    
    def _calculate_price_change(self, current: pd.Series, previous: pd.Series) -> float:
        """가격 변화율 계산"""
        return (current['Close'] - previous['Close']) / previous['Close']
    
    def generate_state(self, current_time: pd.Timestamp) -> List[CandleState]:
        """
        현재 시점의 캔들스틱 상태 데이터 생성
        
        Parameters:
        -----------
        current_time : pd.Timestamp
            현재 시점
            
        Returns:
        --------
        List[CandleState]
            과거 lookback 기간의 캔들스틱 상태 리스트
        """
        # 현재 시점까지의 데이터
        data_until_now = self.data.loc[:current_time]
        
        # 최근 lookback 기간의 데이터
        recent_data = data_until_now.tail(self.lookback)
        
        states = []
        for i in range(len(recent_data)):
            current = recent_data.iloc[i]
            previous = recent_data.iloc[i-1] if i > 0 else None
            
            # 캔들스틱 상태 생성
            state = CandleState(
                body_size=self._calculate_body(current),
                upper_shadow=self._calculate_upper_shadow(current),
                lower_shadow=self._calculate_lower_shadow(current),
                is_bullish=self._is_bullish(current),
                pattern=self._identify_pattern(current, previous),
                volume=current['Volume'],
                price_change=self._calculate_price_change(current, previous) if previous is not None else 0.0
            )
            states.append(state)
        
        return states
    
    def get_state_vector(self):
        # 상태 벡터 생성
        state_vector = np.array([
            self.ha_open,
            self.ha_close,
            self.ha_high,
            self.ha_low,
            self.ha_body,
            self.ha_lower_wick,
            self.ha_upper_wick,
            self.ma_200,
            self.ma_200_signal,
            self.stoch_rsi,
            self.stoch_signal
        ], dtype=np.float32)
        
        return state_vector

    def save_state_to_csv(self, current_time: pd.Timestamp, output_dir: str = 'data/states'):
        """
        현재 시점의 상태 데이터를 CSV 파일로 저장
        
        Parameters:
        -----------
        current_time : pd.Timestamp
            현재 시점
        output_dir : str
            출력 디렉토리 경로
        """
        # 상태 데이터 생성
        states = self.generate_state(current_time)
        
        # 데이터프레임 생성
        state_data = []
        for state in states:
            state_dict = {
                'body_size': state.body_size,
                'upper_shadow': state.upper_shadow,
                'lower_shadow': state.lower_shadow,
                'is_bullish': state.is_bullish,
                'pattern': state.pattern,
                'volume': state.volume,
                'price_change': state.price_change
            }
            state_data.append(state_dict)
        
        df = pd.DataFrame(state_data)
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명 생성 (타임스탬프 포함)
        filename = f"candle_state_{current_time.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # CSV 파일로 저장
        df.to_csv(filepath, index=False)

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    data = pd.read_csv('/workspace/data/preprocess/BTCUSDT/BTCUSDT-1h-new.csv', index_col=0)
    
    # 상태 생성기 초기화
    state_generator = CandleStateGenerator(data, lookback=10)
    
    # 현재 시점의 상태 벡터 생성
    current_time = data.index[-1]
    state_vector = state_generator.get_state_vector()
    
    # 상태 데이터 CSV로 저장
    state_generator.save_state_to_csv(current_time)