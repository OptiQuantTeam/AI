import pandas as pd
import numpy as np

def sigmoid(x):
    """Apply sigmoid function to normalize values to 0-1 range"""
    return 1 / (1 + np.exp(-x))

def PriceChange(data, period=1):
    # Calculate price change
    price_change = data['Close'].pct_change(periods=period)
    
    # Normalize using sigmoid function and scale to 0-100
    normalized_change = sigmoid(price_change) * 100
    
    return normalized_change.iloc[-1]

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/data/raw/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # EMA 계산
    chg = PriceChange(data)
    print(chg)