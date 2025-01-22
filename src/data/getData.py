import pandas as pd
#from indicator.RSI import RSI
#from indicator.EMA import EMA
from indicator import RSI, EMA
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
def getCurrentData():
    pass