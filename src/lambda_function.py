import torch

#from models.ML import LSTM, predict, create_targets
#from data import getCurrentData
import json
from aws_utils import connet_S3, download_model
import importlib
import boto3

# Model의 하이퍼파라미터 및 데이터 설정
input_size = 6  # 예: OHLCV + RSI + MACD + EMA
hidden_size = 64
num_layers = 2
output_size = 3  # 매수, 매도, 대기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s3 = connet_S3()

model = download_model(s3)

def lambda_handler(event, context):
    module = importlib.import_module(f'src/models/{model}')
    '''
    if model == 'LSTM':
        model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        model.load_state_dict(torch.load('LSTM-20250122.pt'))

        new_data = getCurrentData("BTCUSDT", "1h", limit=12)

        side = predict(model, new_data, device)
    elif model == 'IQN':
        pass
    '''
    
    
    return {
        "side":"ddd"
    }