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
    # 이동평균
    data['sma_5'] = data['Close'].rolling(window=5).mean()
    data['sma_20'] = data['Close'].rolling(window=20).mean()
    data['sma_60'] = data['Close'].rolling(window=60).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # 볼린저 밴드
    data['bb_middle'] = data['Close'].rolling(window=20).mean()
    data['bb_std'] = data['Close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']

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
        
        # 거래량 이동평균
        data[f'volume_{tf}'] = data['Volume'].rolling(window=timeframes[tf]).mean()

    # 가격 변화율 추가 (로그 수익률 사용)
    data['price_change'] = np.log(data['Close'] / data['Close'].shift(1))

    # 필요한 컬럼만 선택
    data = data[['Open', 'Close', 'High', 'Low', 'Volume',
                'sma_5', 'sma_20', 'sma_60',
                'RSI', 'MACD', 'MACD_Signal',
                'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
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
    EMA_4 = indicator.EMA(data, window=8)
    EMA_12 = indicator.EMA(data, window=24)
    EMA_24 = indicator.EMA(data, window=48)
    data['EMA_4_slope'] = EMA_4['Normalized_Slope']
    data['EMA_12_slope'] = EMA_12['Normalized_Slope']
    data['EMA_24_slope'] = EMA_24['Normalized_Slope']

    data['stochRSI'] = indicator.StochasticRSI(data)
    
    # MACD
    MACD = indicator.MACD(data, cross=False)
    data['MACD'] = MACD['Histogram']
    data['MACD_Signal'] = MACD['Signal Line']
    data['Cross Signal'] = MACD['Cross Signal']
    data['Divergence Signal'] = MACD['Divergence Signal']
    data['Trade Signal'] = MACD['Trade Signal']
    
    # 볼린저 밴드
    bollinger_bands = indicator.Bollinger(data, window=20, num_std_dev=2)
    data['bb_width'] = bollinger_bands['Band Width']
    data['bb_width_change'] = bollinger_bands['Band Width Change']



    # 가격 변화율 추가 (로그 수익률 사용)
    data['price_change'] = indicator.PriceChange(data)

    # 필요한 컬럼만 선택
    data = data[['Open', 'Close', 'High', 'Low', 'Volume',
                'EMA_4_slope', 'EMA_12_slope', 'EMA_24_slope',
                'stochRSI', 'MACD', 'MACD_Signal',
                'Cross Signal', 'Divergence Signal', 'Trade Signal',
                'bb_width', 'bb_width_change',
                'price_change']]
    
    # NaN 값 제거
    data = data.dropna()
    
    data.to_csv(f'/workspace/data/preprocess/{ticker}/{ticker}-{interval}-technical4.csv')
    return True

def create_technical_indicators3(ticker='BTCUSDT', interval='1d'):
    data = pd.DataFrame()
    year=2024
    while year <= 2024:
        path = f'/workspace/data/raw/{ticker}/{ticker}-{interval}-{year}.csv'
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

    """200 EMA 계산"""
    data['ema_200'] = data['Close'].ewm(span=9600).mean()    # 30분봉 기준 200일 (200 * 48)
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
    data['bb_middle'] = data['Close'].rolling(window=20).mean()
    data['bb_std'] = data['Close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_width_change'] = data['bb_width'].diff()

    # 필요한 컬럼만 선택
    data = data[['Open', 'Close', 'High', 'Low', 'Volume',
                'ha_close', 'ha_open', 'ha_high', 'ha_low',
                'ha_body', 'ha_lower_wick', 'ha_upper_wick',
                'ha_signal', 'ema_200', 'ema_200_signal',
                'stoch_rsi', 'stoch_signal',
                'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
                'bb_width', 'bb_width_change']]
    
    # NaN 값 제거
    data = data.dropna()
    
    data.to_csv(f'/workspace/data/preprocess/{ticker}/{ticker}-{interval}-HEIKIN_ASHI_200EMA_test.csv')
    return True


if __name__ == '__main__':
    #preprocess_historical(interval='30m')
    create_technical_indicators3(interval='30m')