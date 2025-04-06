import pandas as pd
from indicator import CHG, StochasticRSI, MACD
# 전처리 과정을 통해 학습 데이터에 맞게 정제한다.
import indicator.CHG
import indicator.StochasticRSI
import indicator.MACD


def preprocess_historical(ticker='BTCUSDT', interval='3m'):
    data = pd.DataFrame()
    year=2017
    while year<=2024:
        
        path = f'/workspace/BTC_src/{ticker}-{interval}-{year}.csv'

        tmp = pd.read_csv(path, index_col=0)
        tmp['CHG'] = CHG(tmp)
        tmp['stocRSI'] = StochasticRSI(tmp)
        tmp['MACD'] = MACD(tmp)
        tmp = tmp[['Open','Close','Volume','CHG','stocRSI','MACD']]
        data = pd.concat([data, tmp])
        year += 1
    data.to_csv(f'/workspace/BTC_src/{ticker}-{interval}-total.csv')
    return data

if __name__ == '__main__':
    preprocess_historical(interval='3m')