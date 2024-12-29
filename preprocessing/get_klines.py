import requests
import pandas as pd
import time
from datetime import datetime
import math


def get_klines(symbol, interval, start_time=None, end_time=None, limit=None):
    url = "https://api.binance.com/api/v3/klines"
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Base asset volume', 'Number of trades',\
                'Taker buy volume', 'Taker buy base asset volume', 'Ignore']
    df = pd.DataFrame(columns=columns)
    latest=-1

    while True:
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
        time.sleep(0.075)

    df['Open time'] = df['Open time'].astype('int')
    df['Open time'] = df['Open time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df['Close time'] = df['Close time'].astype('int')
    df['Close time'] = df['Close time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df = df.set_index('Open time')

    return df

#timestamp = 1685577600000
#23년 6월 1일 오전 9시의 타임스탬프
#timestamp = 1732792920000
if __name__ == '__main__':
    date_string = '2024-12-28 18:00:00'
    timestamp = int(time.mktime(datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S').timetuple())*1000)
    current = math.trunc(int((time.time()/100))*100000)

    df =  get_klines("BTCUSDT", "1h", start_time=timestamp, end_time=current)
    df.to_csv("./BTCUSDT.csv")
