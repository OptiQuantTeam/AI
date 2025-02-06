from binance.client import Client
import pandas as pd

from dotenv import load_dotenv
import os
from binance.client import Client
from binance.enums import *    


# Binance 클라이언트 생성 (API 키 없이 사용 가능)
client = Client()

# 데이터 수집 설정
symbol = "BTCUSDT"  # 거래 페어
interval = Client.KLINE_INTERVAL_1HOUR  # 1시간봉 데이터
start_date = "2023-01-01"  # 시작 날짜
end_date = "2024-01-01"  # 종료 날짜

# 바이낸스에서 데이터 가져오기
klines = client.get_historical_klines(symbol, interval, start_date, end_date)

# 데이터프레임 변환
columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades",
           "taker_base_volume", "taker_quote_volume", "ignore"]
df = pd.DataFrame(klines, columns=columns)

# 숫자형 데이터 변환
df = df.astype(float)

# 타임스탬프 변환
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# 필요한 열만 선택
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# 데이터 저장
df.to_csv("btc_usdt_data.csv", index=False)

print(df.head())  # 데이터 확인
