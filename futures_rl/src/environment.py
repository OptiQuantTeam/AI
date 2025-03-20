import numpy as np
import pandas as pd
import gym
from gym import spaces
from preprocess import preprocess_data


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
        self.entry_price = 1

        self.trade_fee = 0.0002
        self.load_data()
        self.returns_history = []
        self.max_steps = 1000
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)


    def load_data(self):
        data = pd.read_csv(self.path)
        self.data = preprocess_data(data)
        
    
    def reset(self):
        # 전체 데이터 길이에서 랜덤한 시작 위치 선택
        self.current_step = np.random.randint(0, len(self.data) - self.max_steps)
        # 초기 상태 설정
        self.balance = self.initial_balance
        self.position = FLAT
        self.action = HOLD
        self.size = 0
        self.entry_price = 0
        self.position_size = 0
        self.returns_history = [0]
        self.num = 0
        
        print(f"학습 시작 위치: {self.current_step} (전체 데이터 중 {self.current_step/len(self.data)*100:.2f}%)")
        print(f"시작 시간: {self.data.index[self.current_step]}")
        
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
                if self.position_size != 0 and -position_direction * (current_price - self.entry_price) / self.entry_price > 0.7:
                    reward = profit * 100
                else:
                    reward = profit
            
            profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
            self.returns_history.append(profit_rate * 100)   
            if self.balance < self.initial_balance * 0.3:
                done = True
                reward = -100000 + self.num * 10
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
        else:
            self.current_step += 1

        info = {
            'liquidated': liquidated,
        }
        # 다음 가격으로 포지션 가치 업데이트
        return self._next_observation(), float(reward), done, info

    def render(self):
        # Render the environment to the screen
        profit = self.balance - self.initial_balance
        profit_rate = (self.balance - self.initial_balance) / self.initial_balance
        print(f'Last Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Profit: {profit}, Profit Rate: {profit_rate}')
