import torch

#from models.ML import LSTM, predict, create_targets

from getData import getCurrentData
from utils import predict
import json

#from AWS_Lambda.aws_utils import connet_S3, download_model
import importlib
import boto3
import os
import sys
sys.path.insert(0, 'AWS_Lambda/')
#from LSTM import LSTM
# Model의 하이퍼파라미터 및 데이터 설정
input_size = 6  # 예: OHLCV + RSI + MACD + EMA
hidden_size = 64
num_layers = 2
output_size = 3  # 매수, 매도, 대기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
s3 = connet_S3()

model_name, file_name = download_model(s3)
'''
def load_file(directory):
    """현재 디렉터리에서 가장 먼저 발견된 .pt 파일 반환"""
    for file in os.listdir(directory):
        if file.endswith(".pt"):
            model_name = file.split('-')[0] 
            return model_name, file  # 첫 번째 .pt 파일을 찾으면 즉시 반환
    return None  # 이 부분은 실행되지 않겠지만, 예외 처리를 위해 남김
model_name, file_name = load_file('.')

def lambda_handler(event, context):
    
    module = importlib.import_module(f'{model_name}')
    model_class = getattr(module, model_name)
    print(model_name)
    if model_name == 'LSTM':
        #model = torch.load(f'{file_name}', weights_only=True)
        
        #model.eval()
        
        
        model = model_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        model.load_state_dict(torch.load(file_name, weights_only=True))


        
        new_data = getCurrentData("BTCUSDT", "1h", limit=12)
        
        side = predict(model, new_data, device)
        
    elif model_name == 'IQN':
        pass
    else:
        return {
            "statusCode":400,
        }
    
    
    return {
        "statusCode":200,
        "side":"ddd"
    }


if __name__ == '__main__':
    event={}
    context={}
    lambda_handler(event, context)