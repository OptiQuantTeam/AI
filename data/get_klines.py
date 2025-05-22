import requests
import pandas as pd
import time
from datetime import datetime
import math
import numpy as np
import ta  # Technical Analysis 라이브러리  pip install ta로 받아와야 함
import os

cnt=0

def calculate_heikin_ashi(df):
    # Heikin Ashi 계산
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    ha_high = df[['High', 'Open', 'Close']].max(axis=1)
    ha_low = df[['Low', 'Open', 'Close']].min(axis=1)
    
    # 신호 생성 (1: 상승, 0: 중립, -1: 하락)
    body = abs(ha_close - ha_open)
    lower_wick = np.minimum(ha_open, ha_close) - ha_low
    upper_wick = ha_high - np.maximum(ha_open, ha_close)
    
    signal = np.zeros(len(df))
    signal[(ha_close > ha_open) & (lower_wick < 1e-6) & (body > 0.5)] = 1  # 상승
    signal[(ha_close < ha_open) & (upper_wick < 1e-6) & (body > 0.5)] = -1  # 하락
    
    return signal

def calculate_ema_signal(df):
    # 200 EMA 계산 및 신호 생성
    ema_200 = df['Close'].ewm(span=200).mean()
    signal = np.zeros(len(df))
    signal[df['Close'] > ema_200] = 1  # 상단
    signal[df['Close'] < ema_200] = -1  # 하단
    return signal

def calculate_stoch_signal(df):
    # Stochastic RSI 계산 및 신호 생성
    rsi = ta.momentum.RSIIndicator(df['Close']).rsi()
    stoch_k = ta.momentum.StochRSIIndicator(rsi).stochrsi_k()
    
    signal = np.zeros(len(df))
    signal[stoch_k < 20] = -1  # 과매도
    signal[stoch_k > 80] = 1   # 과매수
    return signal

def get_klines(symbol, interval, start_time=None, end_time=None, limit=None):
    url = "https://api.binance.com/api/v3/klines"
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Base asset volume', 'Number of trades',\
                'Taker buy volume', 'Taker buy base asset volume', 'Ignore']
    df = pd.DataFrame(columns=columns)
    latest=-1
    global cnt
    while True:
        cnt+=1
        if cnt == 1000:
            print("wait")
            time.sleep(60)
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        res = requests.get(url, params=params)
        value = res.json()

        tmp = pd.DataFrame(value, columns=columns)
        
        latest = int(tmp.iat[-1,0])
        if start_time == latest:
            break
        start_time = latest
        df = pd.concat([df,tmp])
        time.sleep(0.01)
    
    # 데이터 타입 변환
    df['Open time'] = df['Open time'].astype('int')
    df['Open time'] = df['Open time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df['Close time'] = df['Close time'].astype('int')
    df['Close time'] = df['Close time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    
    # 숫자형 컬럼 변환
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Base asset volume', 
                      'Number of trades', 'Taker buy volume', 'Taker buy base asset volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    
    # 지표 계산 및 추가
    df['ha_signal'] = calculate_heikin_ashi(df)
    df['ema_200_signal'] = calculate_ema_signal(df)
    df['stoch_signal'] = calculate_stoch_signal(df)
    
    df = df.set_index('Open time')
    print(cnt)
    return df

#timestamp = 1685577600000
#23년 6월 1일 오전 9시의 타임스탬프
#timestamp = 1732792920000
if __name__ == '__main__':
    year = 2024
    cnt = 0

    while year == 2024:
        date_string = str(year)+'-01-01 00:00:00'
        timestamp = int(time.mktime(datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S').timetuple())*1000)
        date_string2 = str(year)+'-12-31 23:59:59'
        timestamp2 = int(time.mktime(datetime.strptime(date_string2, '%Y-%m-%d %H:%M:%S').timetuple())*1000)

        df = get_klines("BTCUSDT", "30m", start_time=timestamp, end_time=timestamp2, limit=1000)
        # 디렉토리가 없으면 생성
        os.makedirs("/workspace/data/raw/BTCUSDT", exist_ok=True)
        df.to_csv(f"/workspace/data/raw/BTCUSDT/BTCUSDT-30m-{year}.csv")
        year+=1