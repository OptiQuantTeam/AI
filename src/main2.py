import torch
from Loader import Loader

'''
AWS Train 서버에서 일주일마다 추가 학습 진행을 위한 프로그램
'''

if __name__ == "__main__":
    env_path = 'data/preprocess/BTCUSDT/BTCUSDT-30m-technical4.csv' # 학습 데이터 경로 수정 예정 
    # 지난 일주일 데이터 로드
    
    loader = Loader(env_path, further=True, auto=True)
    loader.train_in_server()
    
