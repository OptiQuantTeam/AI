import pandas as pd

def CHG(data):
    chg = data['Close']/data['Open']
    chg = pd.DataFrame({'CHG':round((chg-1)*100, 2)}, index=data.index)

    return chg

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/BTC_src/BTCUSDT-1M-2017.csv', index_col=0)
    
    # EMA 계산
    chg = CHG(data)
    print(chg)