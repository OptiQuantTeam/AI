import torch
from torch.utils.data import DataLoader

from models.ML import LSTM, test, LSTMDataset, create_targets
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
model.load_state_dict(torch.load('LSTM-20250122.pt'))

new_data = getTrainData(ticker='BTCUSDT', startYear=2023, interval='1h', raw=True)
new_data = create_targets(new_data, threshold_up, threshold_down, future_window)

pred_dataset = LSTMDataset(new_data, sequence_length)
pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=True)

test(model, pred_loader, device)