import pandas as pd
from indicator import CHG, StochasticRSI, MACD
#전처리 과정을 통해 학습 데이터에 맞게 정제한다.


def preprocess_historical(ticker='BTCUSDT', interval='1d'):
    data = pd.DataFrame()
    year=2017
    while year<=2023:
        
        path = f'/workspace/data/raw/{ticker}/{ticker}-{interval}-{year}.csv'

        tmp = pd.read_csv(path, index_col=0)
        tmp['CHG'] = CHG(tmp)
        tmp['stocRSI'] = StochasticRSI(tmp)
        tmp['MACD'] = MACD(tmp)
        tmp = tmp[['Open','Close','Volume','CHG','stocRSI','MACD']]
        data = pd.concat([data, tmp])
        year += 1
    data.to_csv(f'/workspace/data/preprocess/{ticker}/{ticker}-{interval}.csv')
    return data

if __name__ == '__main__':
    preprocess_historical(interval='5m')