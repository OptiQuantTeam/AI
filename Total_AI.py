import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
from binance.client import Client

# .env 파일 로드 (환경 변수 설정)
load_dotenv()

# 환경 변수 가져오기 (Binance API Key)
API_KEY = os.getenv("API_key")
API_SECRET = os.getenv("secret_key")

# Binance API 클라이언트 초기화
client = Client(API_KEY, API_SECRET)

# 하이퍼파라미터 설정
input_size = 5   # 입력 차원 (OHLCV 데이터 사용)
hidden_size = 64  # LSTM 은닉층 크기
num_layers = 2  # LSTM 층 개수
output_size = 1  # 출력 차원 (예측할 가격 값)

sequence_length = 10  # LSTM 입력 시퀀스 길이
batch_size = 32  # 미니배치 크기
learning_rate = 0.001  # 학습률
epochs = 100  # 학습 반복 횟수
test_sample_size = 100  # 테스트 샘플 개수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Binance 데이터 가져오기
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=400):  # -> 1시간 단위로 400개 정보가져오기 
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df = df[["open", "high", "low", "close", "volume"]].astype(float)     # 5개 열(OHLCV: Open, High, Low, Close, Volume)
    df["target"] = df["close"].shift(-1)   # shift(-1): 데이터를 한 칸 아래로 이동시켜 다음 캔들의 종가를 현재 데이터의 목표값으로 설정 
    df.dropna(inplace=True)
    return df

# LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):   # 모델 초기화함수
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # 입력차원, 은닉층 등등
        self.fc = nn.Linear(hidden_size, output_size)  # 완전연결층 

    def forward(self, x):   # 순전파  (forward 의미 자체)
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # hidden state 사용하지 X
        return self.fc(out[:, -1, :])    # 최종 예측값 반환 

# 데이터셋 클래스
class CryptoDataset(Dataset):    # PyTorch의 Dataset을 상속받아 LSTM 모델 학습을 위한 시계열 데이터를 생성하는 역할
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length  # LSTM 이 학습할 데이터 시퀀스 길이 

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):  # 인덱스에 해당하는 데이터를 반환
        x = self.data[idx : idx + self.seq_length, :-1]  # 마지막 열을 제외한 나머지 OHLCV 데이터 사용
        y = self.data[idx + self.seq_length, -1]  # 마지막 위치의 열은 정답값으로 설정
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)   #  PyTorch 텐서로 변환하고 반환   --> ???

# 모델 학습 함수  --> 학습한 후 저장 
def train_model():
    data = get_binance_data()
    dataset = CryptoDataset(data.values, sequence_length)    # data.values 은 Pandas DataFrame을 NumPy 배열로 변환하는 것것
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()    # MSELoss(): 평균 제곱 오차(MSE) 손실 함수 사용  -- > ???
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    # 모델의 가중치 갱신

    model.train()   # 학습 모드 
    for epoch in range(epochs):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()  # 기울기 초기화 
            loss = criterion(model(X), y)  # 손실 계산 
            loss.backward()  # 역전파로 기울기 계산 
            optimizer.step()
            total_loss += loss.item()    # 현재 배치의 손실값을 total_loss에 추가하는 형태태
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")   # 현재 epoch에서의 평균 손실값 출력

    torch.save(model.state_dict(), "lstm_trading_model.pth")
    print("Model saved successfully.")
    return model

# 모델 평가 함수   -->    학습된 LSTM 모델을 로드하여 새로운 데이터를 평가하고 예측 성능을 측정
def evaluate_model():
    data = get_binance_data(limit=sequence_length + test_sample_size)
    dataset = CryptoDataset(data.values, sequence_length)   # CryptoDataset 클래스로 변환
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load("lstm_trading_model.pth"))   # 저장된 가중치 불러오기
    model.eval()   # 평가 모드

    actuals, predictions = [], []   # 실제 가격, 예측 가격격
    with torch.no_grad():  # -> 기울기 비활성화
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            predictions.append(model(X).item())
            actuals.append(y.item())

    mse = np.mean((np.array(actuals) - np.array(predictions)) ** 2)  # 예측값 - 실제값의 차이를 제곱한 후 평균을 구한 값   --> 낮을수록 좋은 것것
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
    print(f"\nModel Evaluation:\nMSE: {mse:.4f}\nMAPE: {mape:.2f}%")

# 실행 (메인 함수)
if __name__ == "__main__":    # --> 실제해보니 그냥 쓸 수도없는 성능 수준
    train_model()
    evaluate_model()
