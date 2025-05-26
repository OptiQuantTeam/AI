import pandas as pd
import numpy as np

def normalize_to_100(x, min_val=None, max_val=None):
    """Normalize values to 0-100 range using Min-Max normalization"""
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    return ((x - min_val) / (max_val - min_val)) * 100

def EMA(data, window=20):
    # Calculate EMA
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    
    # Calculate EMA slope (rate of change)
    ema_slope = ema.diff()
    
    # Normalize values to 0-100 range
    normalized_ema = normalize_to_100(ema)
    normalized_slope = normalize_to_100(ema_slope)
    
    return {
        'Normalized_EMA': normalized_ema,
        'Normalized_Slope': normalized_slope
    }

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # EMA 계산
    ema_result = EMA(data, window=10)
    print(ema_result)