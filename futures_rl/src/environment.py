import numpy as np
import pandas as pd
import gym
from gym import spaces
from preprocess import preprocess_data

class FuturesTradeEnv(gym.Env):
    def __init__(self, df, initial_balance=100000000, transaction_fee_percent=0.0002, max_leverage=10):
        super(FuturesTradeEnv, self).__init__()
        
        # 데이터 설정
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.max_leverage = max_leverage  # 최대 레버리지
        
        # 현재 스텝 (데이터 포인트)
        self.current_step = 0
        self.max_steps = len(df) - 1  # 데이터의 최대 길이
        # 수익률 기록
        self.returns_history = []
        
        # Action Space: 2차원 연속 공간
        # 첫 번째 차원: [-1.0, 1.0] - 포지션 방향 (-1: 최대 매도, 0: 홀드, 1: 최대 매수)
        # 두 번째 차원: [0.0, 1.0] - 거래량 비율 (가능한 최대 거래량의 비율)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation Space: 
        # [현재 잔고, 포지션 크기, Open, Close, Volume, CHG, stocRSI, MACD]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.current_price = self.df.iloc[self.current_step]['Close']
        self.trades = []
        self.returns_history = [0]  # 초기 수익률은 0%
        return self._get_observation()
    
    def _get_observation(self):
        # 현재 시점의 데이터 가져오기
        current_data = self.df.iloc[self.current_step]
        
        # 관측값 생성
        obs = np.array([
            self.balance,
            self.position,
            current_data['Open'],
            current_data['Close'],
            current_data['Volume'],
            current_data['CHG'],
            current_data['stocRSI'],
            current_data['MACD']
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_position_size(self, direction, volume_ratio):
        """포지션 크기(계약 수량) 계산
        
        Args:
            direction: -1 ~ 1 사이의 값 (포지션 방향)
            volume_ratio: 0 ~ 1 사이의 값 (최대 거래량 대비 실제 거래량 비율)
        """
        # 사용 가능한 증거금 계산 (초기 자본의 일정 비율)
        available_margin = self.balance * 0.9  # 90%만 사용
        
        # 현재 가격 기준 최대 계약 수량 계산
        max_contracts = (available_margin * self.max_leverage) / self.current_price
        
        # volume_ratio를 적용하여 실제 계약 수량 계산
        contracts = max_contracts * volume_ratio * abs(direction)
        
        # 방향 설정 (매수/매도)
        if direction < 0:
            contracts = -contracts
            
        return contracts
    
    def step(self, action):
        # 현재 가격과 다음 가격 가져오기
        current_price = self.df.iloc[self.current_step]['Close']
        
        # 마지막 스텝인 경우 현재 가격을 다음 가격으로 사용
        if self.current_step == self.max_steps:
            next_price = current_price
            done = True
        else:
            next_step = self.current_step + 1
            next_price = self.df.iloc[next_step]['Close']
            done = False
        
        liquidated = False
        
        # 포지션 진입 전 자산 저장
        pre_trade_assets = self.balance
        if self.position != 0:
            position_value = abs(self.position * self.position_size * current_price)
            pre_trade_assets += position_value
        
        # 거래 규모 계산 (action[0]은 방향, action[1]은 거래 규모 비율)
        trade_ratio = abs(action[1])  # 거래 규모 비율 (0~1)
        position_size = self.balance * trade_ratio / current_price  # 실제 거래 수량
        
        # 포지션 방향 결정
        if action[0] > 0:
            position_direction = 1  # 롱 포지션
        elif action[0] < 0:
            position_direction = -1  # 숏 포지션
        else:
            position_direction = 0  # 포지션 없음
        
        # 포지션 진입 또는 변경
        if self.position != position_direction:
            # 기존 포지션 청산
            if self.position != 0:
                # 청산 비용 계산
                liquidation_cost = abs(self.position) * current_price * self.transaction_fee_percent
                self.balance -= liquidation_cost
            
            # 새로운 포지션 진입
            if position_direction != 0:
                # 진입 비용 계산
                entry_cost = position_size * current_price * self.transaction_fee_percent
                self.balance -= entry_cost
                self.position = position_direction
                self.position_size = position_size  # 포지션 크기 저장
        
        # 다음 가격으로 포지션 가치 업데이트
        price_change = (next_price - current_price) / current_price
        if self.position != 0:
            # 포지션 수익/손실 계산
            position_profit = self.position * self.position_size * (next_price - current_price)
            self.balance += position_profit
            
            # 현재 총 자산 계산
            current_position_value = abs(self.position * self.position_size * next_price)
            current_total_assets = self.balance + current_position_value
            
            # 포지션 진입 전 자산 대비 10% 이상 하락 시 강제 청산
            if current_total_assets < pre_trade_assets * 0.9:
                liquidation_cost = self.position_size * next_price * self.transaction_fee_percent
                self.balance -= liquidation_cost
                self.position = 0
                self.position_size = 0
                reward = -100  # 큰 손실에 대한 페널티
                #liquidated = True
                #done = True
        
        # 수익률 계산
        profit_rate = (self.balance - self.initial_balance) / self.initial_balance
        
        # 총 자산 계산 (현재 잔고 + 포지션 가치)
        position_value = 0
        if self.position != 0:
            position_value = abs(self.position * self.position_size * next_price)
        total_assets = self.balance + position_value
        
        # 초기 자산 대비 30% 이상 하락 시 에피소드 종료
        if total_assets < self.initial_balance * 0.7:
            done = True
            reward = -1000000 + self.current_step * 10  # 큰 손실에 대한 페널티
            liquidated = True
        
        # 보상 계산 (수익률과 동일한 비율)
        if not done:
            reward = profit_rate * 100
        
        # 다음 상태로 이동
        if not done:
            self.current_step = next_step
        else:
            # 마지막 스텝에서 포지션 청산
            if self.position != 0:
                liquidation_cost = self.position_size * next_price * self.transaction_fee_percent
                self.balance -= liquidation_cost
                self.position = 0
                self.position_size = 0
        
        # 수익률 기록 업데이트
        self.returns_history.append(profit_rate * 100)
        
        # 다음 상태 반환
        next_state = self._get_observation()
        info = {
            'liquidated': liquidated,
            'total_assets': total_assets,
            'position_value': position_value,
            'position_size': self.position_size if self.position != 0 else 0,
            'current_price': current_price,
            'next_price': next_price,
            'current_step': self.current_step,
            'pre_trade_assets': pre_trade_assets
        }
        
        return next_state, reward, done, info

    def render(self):
            # Render the environment to the screen
            profit = self.balance - self.initial_balance
            profit_rate = (self.balance - self.initial_balance) / self.initial_balance
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Profit: {profit}, Profit Rate: {profit_rate}')


from gym.utils import seeding

# position constant
LONG = 0
SHORT = 1
FLAT = 2

# action constant
BUY = 0
SELL = 1
HOLD = 2

class FuturesEnv(gym.Env):
    def __init__(self, path=None):
        self.path = path
        self.initial_balance = 100000000
        self.balance = self.initial_balance
        self.actions = ['LONG', 'SHORT', 'FLAT']
        self.position = FLAT
        self.action = HOLD
        self.size = 0
        self.leverage = 2
        self.current_step = 0
        self.entry_price = 0
        self.trade_fee = 0.0002
        self.load_data()
        self.returns_history = []

        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)


    def load_data(self):
        data = pd.read_csv(self.path)
        self.data = preprocess_data(data)
        
    
    def reset(self):
        self.history = []
        self.action = HOLD
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._next_observation()
    
    def _next_observation(self):
        return np.array([float(self.balance), float(self.position), self.data.iloc[self.current_step]['Open'],
                         self.data.iloc[self.current_step]['Close'], self.data.iloc[self.current_step]['Volume'], self.data.iloc[self.current_step]['CHG'],
                         self.data.iloc[self.current_step]['stocRSI'], self.data.iloc[self.current_step]['MACD']], dtype=np.float32)
        
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        done = self.current_step >= len(self.data) - 1
        liquidated = False
        reward = -1
        

        if action > 0:
            position_direction = 1
        elif action < 0:
            position_direction = -1
        else:
            position_direction = 0

        if self.position != position_direction:
            # 기존 포지션 청산
            if self.position != 0:
                profit = -position_direction * (current_price - self.entry_price) * self.size
                self.balance += profit
                if (current_price - self.entry_price) / self.entry_price > 0.3:
                    reward = profit * 100
                else:
                    reward = profit
            
            profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
            self.returns_history.append(profit_rate * 100)   
            if self.balance < self.initial_balance * 0.7:
                done = True
                reward = -10000 + self.current_step * 10
                liquidated = True

            # 새로운 포지션 진입
            if position_direction != 0 and not done:
                # 진입 비용 계산
                trade_ratio = abs(action)
                position_size = self.balance * self.leverage * trade_ratio / current_price
                entry_cost = position_size * current_price * self.trade_fee
                self.balance -= entry_cost
                self.position = position_direction
                self.size = position_size
                self.entry_price = current_price
        
        if done:
            self.returns_history.append(self.current_step)
        info = {
            'liquidated': liquidated,
        }
        # 다음 가격으로 포지션 가치 업데이트
        return self._next_observation(), float(reward), done, info

    def render(self):
        # Render the environment to the screen
        profit = self.balance - self.initial_balance
        profit_rate = (self.balance - self.initial_balance) / self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Profit: {profit}, Profit Rate: {profit_rate}')


MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

import random

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, ), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.iloc[self.current_step - 6 : self.current_step -1]['Open'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - 6: self.current_step -1]['Close'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - 6: self.current_step -1]['Volume'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - 6: self.current_step -1]['CHG'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - 6: self.current_step -1]['stocRSI'].values / MAX_NUM_SHARES,
            self.df.iloc[self.current_step - 6: self.current_step -1]['MACD'].values/ MAX_NUM_SHARES     
        ])
        # Append additional data and scale each value to between 0-1
        # print(self.current_step)
        # print(frame.shape)
        obs1 = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)
        obs =obs1
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "Close"]

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            6, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')