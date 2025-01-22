import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.ML import LSTM, train, predict, LSTMDataset, create_targets
from data import getTrainData

# 모델 설정
# 현재 데이터 가져오기
# 현제 데이터로 판단하기
# 판단 결과를 request로 전달하기(AWS API Gateway 이용)

# Target을 계산하기 위한 파라미터 설정
threshold_up = 0.02  # 매수 기준 상승률 (2%)
threshold_down = -0.02  # 매도 기준 하락률 (-2%)
future_window = 5  # 미래 데이터 관찰 창 (5분)
sequence_length=10
batch_size=32

# Model의 하이퍼파라미터 및 데이터 설정
input_size = 6  # 예: OHLCV + RSI + MACD + EMA
hidden_size = 64
num_layers = 2
output_size = 3  # 매수, 매도, 대기
sequence_length = 10
learning_rate = 0.001
epochs = 20
batch_size = 32




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

data = getTrainData(ticker='BTCUSDT', startYear=2017, endYear=2022, interval='1h', raw=True)
# Target 열 생성
data = create_targets(data, threshold_up, threshold_down, future_window)

# 학습/검증 데이터 분할
train_size = int(0.8 * len(data))
train_df = data[:train_size]
val_df = data[train_size:]

# Dataset 및 DataLoader 생성
train_dataset = LSTMDataset(train_df, sequence_length)
val_dataset = LSTMDataset(val_df, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train(model, train_loader, val_loader, epochs, learning_rate, device)


new_data = getTrainData(ticker='BTCUSDT', startYear=2023, interval='1h', raw=True)
new_data = create_targets(new_data, threshold_up, threshold_down, future_window)

pred_dataset = LSTMDataset(new_data, sequence_length)
pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=True)

predict(model, pred_loader, device)


torch.save(model.state_dict(), 'LSTM-20250122.pt')