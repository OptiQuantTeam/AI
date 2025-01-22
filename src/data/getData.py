import pandas as pd
from indicator import RSI, EMA
import requests
from datetime import datetime

def getTrainData(ticker='BTCUSDT', startYear=2017, endYear=2023, interval='1h', raw=True):
    
    data = pd.DataFrame()
    while startYear<=endYear:
        path=''
        if raw:
            path = f'/workspace/data/raw/{ticker}/{ticker}-{interval}-{startYear}.csv'
        else:
            path = f'/workspace/data/processed/{ticker}/{ticker}-{interval}-{startYear}.csv'

        tmp = pd.read_csv(path, index_col=0)
        tmp['RSI'] = RSI(tmp)
        tmp['EMAF'] = EMA(tmp, window=10)
        tmp = tmp[['Open','High','Low','Close','RSI','EMAF']]
        data = pd.concat([data, tmp])
        startYear += 1

    return data


# 1분, 5분, 30분, 1시간 등의 데이터(현재 데이터)를 거래소로부터 가져온다.
def getCurrentData(symbol, interval='1m', limit=None):
    url = "https://api.binance.com/api/v3/klines"
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Base asset volume', 'Number of trades',\
                'Taker buy volume', 'Taker buy base asset volume', 'Ignore']
    

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": None,
        "endTime": None,
        "limit": limit
    }
    res = requests.get(url, params=params)
    value = res.json()

    df = pd.DataFrame(value, columns=columns)
        
        

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
   
    df =  getCurrentData("BTCUSDT", "1h", limit=1)
    print(df)
    #df.to_csv("./BTCUSDT-15m-"+str(year)+".csv")
    
