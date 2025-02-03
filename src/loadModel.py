import torch
from torch.utils.data import DataLoader

from models.ML import LSTM, train, LSTMDataset, create_targets
from data import getTrainData

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
epochs = 100
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
model.load_state_dict(torch.load('LSTM-20250122.pt'))
model.train()

new_data = getTrainData(ticker='BTCUSDT', startYear=2023, interval='1h', raw=True)
new_data = create_targets(new_data, threshold_up, threshold_down, future_window)

train_size = int(0.8 * len(new_data))
train_df = new_data[:train_size]
val_df = new_data[train_size:]


train_dataset = LSTMDataset(train_df, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


val_dataset = LSTMDataset(val_df, sequence_length)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



train(model, train_loader, val_loader, epochs, learning_rate, device)