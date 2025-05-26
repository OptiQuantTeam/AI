import pandas as pd
import indicator
import numpy as np
#전처리 과정을 통해 학습 데이터에 맞게 정제한다.


def preprocess_historical(ticker='BTCUSDT', interval='1d'):
    data = pd.DataFrame()
    year=2017
    while year<=2023:
        
        path = f'/workspace/data/raw/{ticker}/{ticker}-{interval}-{year}.csv'

        tmp = pd.read_csv(path, index_col=0)
        tmp['CHG'] = indicator.PriceChange(tmp)
        tmp['stocRSI'] = indicator.StochasticRSI(tmp)
        tmp['MACD'] = indicator.MACD(tmp)
        tmp = tmp[['Open','Close','High','Low','Volume']]
        data = pd.concat([data, tmp])
        year += 1
    data.to_csv(f'/workspace/data/preprocess/{ticker}/{ticker}-{interval}-raw.csv')
    return data

def create_technical_indicators(ticker='BTCUSDT', interval='1d'):
    data = pd.DataFrame()
    year=2017
    while year <= 2023:
        path = f'/workspace/data/raw/{ticker}/{ticker}-{interval}-{year}.csv'
        df = pd.read_csv(path, index_col=0)
        data = pd.concat([data, df])
        year += 1

    """기술적 지표 생성"""
    # EMA - 정규화된 값 사용
    EMA_5 = indicator.EMA(data, window=5)
    EMA_20 = indicator.EMA(data, window=20)
    EMA_60 = indicator.EMA(data, window=60)
    data['ema_5'] = EMA_5['Normalized_EMA']  # 정규화된 EMA 값 (0-100)
    data['ema_5_slope'] = EMA_5['Normalized_Slope']  # 정규화된 기울기 (0-100)
    data['ema_20'] = EMA_20['Normalized_EMA']
    data['ema_20_slope'] = EMA_20['Normalized_Slope']
    data['ema_60'] = EMA_60['Normalized_EMA']
    data['ema_60_slope'] = EMA_60['Normalized_Slope']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD - 정규화된 값 사용
    MACD_result = indicator.MACD(data, cross=False)
    data['MACD'] = MACD_result['MACD']  # 정규화된 MACD 값 (0-100)
    data['MACD_Signal'] = MACD_result['MACD_Signal']  # 정규화된 Signal Line 값 (0-100)
    data['Cross Signal'] = MACD_result['Cross Signal']  # 1: 상향돌파, -1: 하향돌파, 0: 그 외
    data['Divergence Signal'] = MACD_result['Divergence Signal']  # 1: 상승다이버전스, -1: 하락다이버전스, 0: 그 외
    data['Trade Signal'] = MACD_result['Trade Signal']  # 1: 매수신호, -1: 매도신호, 0: 그 외
    
    # 볼린저 밴드 - 정규화된 값 사용
    bollinger_bands = indicator.Bollinger(data, window=20, num_std_dev=2)
    data['bb_width'] = bollinger_bands['Band Width']  # 정규화된 밴드 폭 (0-100)
    data['bb_width_change'] = bollinger_bands['Band Width Change']  # 정규화된 밴드 폭 변화 (0-100)
    data['bb_overbought'] = bollinger_bands['Overbought Signal']  # 1: 과매수, 0: 그 외
    data['bb_oversold'] = bollinger_bands['Oversold Signal']  # 1: 과매도, 0: 그 외

    """변동성 지표 생성"""
    # 변동성 계산
    volatility_window = 20  # 변동성 계산 기간
    volatility_ma_period = 5  # 변동성 이동평균 기간
    volatility_std_period = 20  # 변동성 표준편차 기간
    
    data['volatility'] = data['Close'].pct_change().rolling(window=volatility_window).std()
    data['volatility_ma'] = data['volatility'].rolling(window=volatility_ma_period).mean()
    data['volatility_std'] = data['volatility'].rolling(window=volatility_std_period).std()
    
    """다중 시간프레임 데이터 생성"""
    timeframes = {
        'short': 5,    # 5분
        'medium': 15,  # 15분
        'long': 60     # 60분
    }
    
    for tf in timeframes:
        # 종가 이동평균
        data[f'close_{tf}'] = data['Close'].rolling(window=timeframes[tf]).mean()
        
        # 변동성 계산 (로그 수익률의 표준편차)
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        data[f'volatility_{tf}'] = log_returns.rolling(window=timeframes[tf]).std()
        
        # 거래량 이동평균 - 정규화된 값 사용
        data[f'volume_{tf}'] = indicator.VMA(data, period=timeframes[tf])  # 정규화된 VMA 값 (0-100)

    # 가격 변화율 추가 (로그 수익률 사용)
    data['price_change'] = np.log(data['Close'] / data['Close'].shift(1))

    # 필요한 컬럼만 선택
    data = data[['Open', 'Close', 'High', 'Low', 'Volume',
                'ema_5', 'ema_5_slope', 'ema_20', 'ema_20_slope', 'ema_60', 'ema_60_slope',
                'RSI', 'MACD', 'MACD_Signal',
                'Cross Signal', 'Divergence Signal', 'Trade Signal',
                'bb_width', 'bb_width_change', 'bb_overbought', 'bb_oversold',
                'volatility', 'volatility_ma', 'volatility_std',
                'close_short', 'volatility_short', 'volume_short',
                'close_medium', 'volatility_medium',
                'close_long', 'volatility_long',
                'price_change']]
    
    # NaN 값 제거
    data = data.dropna()
    
    data.to_csv(f'/workspace/data/preprocess/{ticker}/{ticker}-{interval}-technical2.csv')
    return True

def create_technical_indicators2(ticker='BTCUSDT', interval='1d'):
    data = pd.DataFrame()
    year=2017
    while year <= 2023:
        path = f'/workspace/data/raw/{ticker}/{ticker}-{interval}-{year}.csv'
        df = pd.read_csv(path, index_col=0)
        data = pd.concat([data, df])
        year += 1

    """기술적 지표 생성"""
    # EMA - 정규화된 값 사용
    EMA_4 = indicator.EMA(data, window=8)
    EMA_12 = indicator.EMA(data, window=24)
    EMA_24 = indicator.EMA(data, window=48)
    data['EMA_4'] = EMA_4['Normalized_EMA']  # 정규화된 EMA 값 (0-100)
    data['EMA_4_slope'] = EMA_4['Normalized_Slope']  # 정규화된 기울기 (0-100)
    data['EMA_12'] = EMA_12['Normalized_EMA']
    data['EMA_12_slope'] = EMA_12['Normalized_Slope']
    data['EMA_24'] = EMA_24['Normalized_EMA']
    data['EMA_24_slope'] = EMA_24['Normalized_Slope']

    data['stochRSI'] = indicator.StochasticRSI(data)
    
    # MACD - 정규화된 값 사용
    MACD_result = indicator.MACD(data, cross=False)
    data['MACD'] = MACD_result['MACD']  # 정규화된 MACD 값 (0-100)
    data['MACD_Signal'] = MACD_result['MACD_Signal']  # 정규화된 Signal Line 값 (0-100)
    data['Cross Signal'] = MACD_result['Cross Signal']  # 1: 상향돌파, -1: 하향돌파, 0: 그 외
    data['Divergence Signal'] = MACD_result['Divergence Signal']  # 1: 상승다이버전스, -1: 하락다이버전스, 0: 그 외
    data['Trade Signal'] = MACD_result['Trade Signal']  # 1: 매수신호, -1: 매도신호, 0: 그 외
    
    # 볼린저 밴드 - 정규화된 값 사용
    bollinger_bands = indicator.Bollinger(data, window=20, num_std_dev=2)
    data['bb_width'] = bollinger_bands['Band Width']  # 정규화된 밴드 폭 (0-100)
    data['bb_width_change'] = bollinger_bands['Band Width Change']  # 정규화된 밴드 폭 변화 (0-100)
    data['bb_overbought'] = bollinger_bands['Overbought Signal']  # 1: 과매수, 0: 그 외
    data['bb_oversold'] = bollinger_bands['Oversold Signal']  # 1: 과매도, 0: 그 외

    # 거래량 이동평균 - 정규화된 값 사용
    data['volume_ma'] = indicator.VMA(data, period=20)  # 정규화된 VMA 값 (0-100)

    # 가격 변화율 추가 (로그 수익률 사용)
    data['price_change'] = indicator.PriceChange(data)

    # 필요한 컬럼만 선택
    data = data[['Open', 'Close', 'High', 'Low', 'Volume',
                'EMA_4', 'EMA_4_slope', 'EMA_12', 'EMA_12_slope', 'EMA_24', 'EMA_24_slope',
                'stochRSI', 'MACD', 'MACD_Signal',
                'Cross Signal', 'Divergence Signal', 'Trade Signal',
                'bb_width', 'bb_width_change', 'bb_overbought', 'bb_oversold',
                'volume_ma', 'price_change']]
    
    # NaN 값 제거
    data = data.dropna()
    
    data.to_csv(f'/workspace/data/preprocess/{ticker}/{ticker}-{interval}-technical4.csv')
    return True

def create_technical_indicators3(ticker='BTCUSDT', interval='1d'):
    data = pd.DataFrame()
    year=2020
    while year <= 2023:
        path = f'/workspace/data/raw/{ticker}-{interval}-{year}.csv'
        df = pd.read_csv(path, index_col=0)
        data = pd.concat([data, df])
        year += 1

    """하이킨 아시 캔들 계산"""
    # Heikin Ashi 캔들 계산
    data['ha_close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    data['ha_open'] = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
    data['ha_high'] = data[['High', 'Open', 'Close']].max(axis=1)
    data['ha_low'] = data[['Low', 'Open', 'Close']].min(axis=1)
    
    # 캔들 특성 계산
    data['ha_body'] = abs(data['ha_close'] - data['ha_open'])
    data['ha_lower_wick'] = np.minimum(data['ha_open'], data['ha_close']) - data['ha_low']
    data['ha_upper_wick'] = data['ha_high'] - np.maximum(data['ha_open'], data['ha_close'])
    
    # 하이킨 아시 신호 생성 (1: 상승, 0: 중립, -1: 하락)
    data['ha_signal'] = 0
    data.loc[(data['ha_close'] > data['ha_open']) & 
             (data['ha_lower_wick'] < 1e-6) & 
             (data['ha_body'] > 0.5), 'ha_signal'] = 1
    data.loc[(data['ha_close'] < data['ha_open']) & 
             (data['ha_upper_wick'] < 1e-6) & 
             (data['ha_body'] > 0.5), 'ha_signal'] = -1

    data['ha_high_diff'] = data['ha_high'] - data['ha_high'].shift(1)
    data['ha_low_diff'] = data['ha_low'] - data['ha_low'].shift(1)
    data['ha_body_diff'] = data['ha_body'] - data['ha_body'].shift(1)

    """200 EMA 계산"""
    EMA_200 = indicator.EMA(data, window=9600)  # 30분봉 기준 200일 (200 * 48)
    data['ema_200'] = EMA_200['Normalized_EMA']  # 정규화된 EMA 값 (0-100)
    data['ema_200_slope'] = EMA_200['Normalized_Slope']  # 정규화된 기울기 (0-100)
    data['ema_200_signal'] = 0
    data.loc[data['Close'] > data['ema_200'], 'ema_200_signal'] = 1
    data.loc[data['Close'] < data['ema_200'], 'ema_200_signal'] = -1

    """Stochastic RSI 계산"""
    # RSI 계산
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic RSI 계산
    data['stoch_rsi'] = (data['RSI'] - data['RSI'].rolling(14).min()) / \
                        (data['RSI'].rolling(14).max() - data['RSI'].rolling(14).min())
    
    # Stochastic RSI 신호 생성 (1: 과매수, 0: 중립, -1: 과매도)
    data['stoch_signal'] = 0
    data.loc[data['stoch_rsi'] < 0.2, 'stoch_signal'] = -1  # 과매도
    data.loc[data['stoch_rsi'] > 0.8, 'stoch_signal'] = 1   # 과매수

    """볼린저 밴드 계산"""
    bollinger_bands = indicator.Bollinger(data, window=20, num_std_dev=2)
    data['bb_width'] = bollinger_bands['Band Width']  # 정규화된 밴드 폭 (0-100)
    data['bb_width_change'] = bollinger_bands['Band Width Change']  # 정규화된 밴드 폭 변화 (0-100)
    data['bb_overbought'] = bollinger_bands['Overbought Signal']  # 1: 과매수, 0: 그 외
    data['bb_oversold'] = bollinger_bands['Oversold Signal']  # 1: 과매도, 0: 그 외

    # MACD
    MACD = indicator.MACD(data, cross=False)
    data['MACD'] = MACD['MACD']  # 정규화된 MACD 값 (0-100)
    data['MACD_Signal'] = MACD['MACD_Signal']  # 정규화된 Signal Line 값 (0-100)
    data['Cross Signal'] = MACD['Cross Signal']  # 1: 상향돌파, -1: 하향돌파, 0: 그 외
    data['Divergence Signal'] = MACD['Divergence Signal']  # 1: 상승다이버전스, -1: 하락다이버전스, 0: 그 외
    data['Trade Signal'] = MACD['Trade Signal']  # 1: 매수신호, -1: 매도신호, 0: 그 외

    # 거래량 이동평균 - 정규화된 값 사용
    data['volume_ma'] = indicator.VMA(data, period=20)  # 정규화된 VMA 값 (0-100)

    # 필요한 컬럼만 선택
    data = data[['Open', 'Close', 'High', 'Low', 'Volume',
                'ha_close', 'ha_open', 'ha_high', 'ha_low',
                'ha_body', 'ha_lower_wick', 'ha_upper_wick',
                'ha_signal', 'ha_high_diff', 'ha_low_diff', 'ha_body_diff',
                'ema_200', 'ema_200_slope', 'ema_200_signal',
                'stoch_rsi', 'stoch_signal',
                'bb_width', 'bb_width_change', 'bb_overbought', 'bb_oversold',
                'MACD', 'MACD_Signal', 'Cross Signal', 'Divergence Signal', 'Trade Signal',
                'volume_ma']]
    
    # NaN 값 제거
    data = data.dropna()
    
    data.to_csv(f'/workspace/data/preprocess/{ticker}/{ticker}-{interval}-FINAL_NEW_2020.csv')
    return True


if __name__ == '__main__':
    #preprocess_historical(interval='30m')
    create_technical_indicators3(interval='30m')