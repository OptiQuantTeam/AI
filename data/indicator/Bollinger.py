import pandas as pd
import numpy as np

def normalize_to_100(x, min_val=None, max_val=None):
    """Normalize values to 0-100 range using Min-Max normalization"""
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    return ((x - min_val) / (max_val - min_val)) * 100

def Bollinger(data, window=20, num_std_dev=2):
    # 이동 평균 계산 (중앙 밴드)
    middle_band = data['Close'].rolling(window=window).mean()
    
    # 이동 표준 편차 계산
    std_dev = data['Close'].rolling(window=window).std()
    
    # 상단 밴드와 하단 밴드 계산
    upper_band = middle_band + (num_std_dev * std_dev)
    lower_band = middle_band - (num_std_dev * std_dev)
    
    # 밴드 폭 계산
    band_width = (upper_band - lower_band) / middle_band
    
    # Calculate band width change
    band_width_change = band_width.diff()
    
    # Normalize values to 0-100 range
    normalized_width = normalize_to_100(band_width)
    normalized_change = normalize_to_100(band_width_change)
    
    # Generate signals
    overbought_signal = (data['Close'] > upper_band).astype(int)
    oversold_signal = (data['Close'] < lower_band).astype(int)
    
    return {
        'Band Width': normalized_width,
        'Band Width Change': normalized_change,
        'Overbought Signal': overbought_signal,
        'Oversold Signal': oversold_signal
    }

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # 볼린저 밴드 계산
    bollinger_bands = Bollinger(data)
    print(bollinger_bands)