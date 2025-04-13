import pandas as pd

def preprocess_data(df):
    """데이터 전처리: 결측치 처리 및 정규화"""
    # 필요한 컬럼만 선택
    #df = df[['Open', 'Close', 'Volume', 'CHG', 'stocRSI', 'MACD']]
    df = df[['Close', 'High', 'Low', 'Volume', 'stocRSI', 'MACD']]
    # 결측치 처리
    #df = df.fillna(method='ffill')  # 앞의 값으로 채우기
    df = df.ffill()
    df = df.bfill()
    #df = df.fillna(method='bfill')  # 뒤의 값으로 채우기
    
    # 이상치 제거 (극단값 제거)
    #for column in ['Open', 'Close', 'Volume', 'CHG']:
    for column in ['Close', 'High', 'Low', 'Volume', 'stocRSI', 'MACD']:
        q1 = df[column].quantile(0.01)
        q3 = df[column].quantile(0.99)
        df[column] = df[column].clip(q1, q3)
    
    # 정규화
    '''
    for column in ['Open', 'Close', 'Volume', 'CHG']:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / (std + 1e-8)
    '''
    # stocRSI와 MACD는 이미 정규화된 형태이므로 극단값만 처리
    df['stocRSI'] = df['stocRSI'].clip(0, 100)
    df['MACD'] = df['MACD'].clip(-10, 10)  # 적절한 범위로 조정
    
    return df