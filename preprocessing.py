from binance.client import Client
import pandas as pd
from dotenv import load_dotenv
import os
from binance.client import Client
from binance.enums import *    

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler #sklearn을 사용하기 위해 라이브러리 설치해야함
#설치코드 pip install pandas numpy scikit-learn

load_dotenv()

# Binance 클라이언트 생성 (API 키 없이 사용 가능)
api_key = os.getenv('API_KEY')
secret_key = os.getenv('Secret_Key')

client = Client(api_key, secret_key)

# 데이터 로드
df = pd.read_csv("btc_usdt_data.csv", parse_dates=["timestamp"])

# 결측치 제거
df = df.dropna()

# 정규화 (가격과 거래량만 스케일링)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["open", "high", "low", "close", "volume"]])

# 시퀀스 데이터 생성 함수(시계열 데이터를 입력과 출력으로 변환하는 역할을 함)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])  # seq_length 길이의 입력 데이터
        y.append(data[i + seq_length, 3])   # close 가격 (타겟)
    return np.array(X), np.array(y)

# 하이퍼파라미터 설정
SEQ_LENGTH = 50  # 최근 50개 데이터를 사용하여 다음 값 예측

# 시퀀스 데이터 생성
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# 데이터 확인
print(f"입력 데이터 형태: {X.shape}")  # (샘플 개수, 시퀀스 길이, 특성 개수)
print(f"출력 데이터 형태: {y.shape}")  # (샘플 개수,)

# 데이터 저장 (LSTM 학습용)
np.save("X.npy", X)
np.save("y.npy", y)
np.save("scaler.npy", scaler)  # 스케일러 저장 (예측 후 복원할 때 필요)
