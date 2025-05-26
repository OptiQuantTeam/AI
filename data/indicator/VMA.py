import pandas as pd
import numpy as np

def normalize_to_100(x, min_val=None, max_val=None):
    """Normalize values to 0-100 range using Min-Max normalization"""
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    return ((x - min_val) / (max_val - min_val)) * 100

def VMA(data, period=20):
    # Calculate volume moving average
    vma = data['Volume'].rolling(window=period).mean()
    
    # Normalize to 0-100 range
    normalized_vma = normalize_to_100(vma)
    
    return normalized_vma  # 전체 시리즈 반환

# 예제 데이터 사용
if __name__ == "__main__":
    # 거래량 데이터 생성
    data = pd.read_csv('/workspace/data/raw/BTCUSDT-1m-2018.csv', index_col=0)
    
    # 거래량 이동평균선 계산
    vma = VMA(data, period=5)
    
    print("VMA:\n", vma)