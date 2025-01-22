import pandas as pd

# Target 계산 함수
def _calculate_target(row, future_prices, threshold_up, threshold_down):
    current_price = row['Close']
    max_future_price = future_prices.max()
    min_future_price = future_prices.min()
    
    if max_future_price >= current_price * (1 + threshold_up):
        return 0  # 매수
    elif min_future_price <= current_price * (1 + threshold_down):
        return 1  # 매도
    else:
        return 2  # 대기

# Target 생성
def create_targets(df, threshold_up, threshold_down, future_window):
    targets = []
    for i in range(len(df)):
        if i + future_window < len(df):
            # future_window 만큼의 미래 Close 가격을 가져옴
            future_prices = df['Close'].iloc[i + 1:i + 1 + future_window]
            targets.append(_calculate_target(df.iloc[i], future_prices, threshold_up, threshold_down))
        else:
            targets.append(2)  # 미래 데이터 부족 시 대기
    df['Target'] = targets
    df.dropna(inplace=True)
    return df