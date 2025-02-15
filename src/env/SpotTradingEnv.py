import gym
import numpy as np
import pandas as pd

class SpotTradingEnv(gym.Env):
    def __init__(self, data):
        super(SpotTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = 10000
        self.holdings = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.avg_price = 0

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.holdings = 0

        return self._next_observation()
    
    def _next_observation(self):
        return np.array([self.balance, self.holdings, self.data.iloc[self.current_step]['Open'],
                         self.data.iloc[self.current_step]['Close'], self.data.iloc[self.current_step]['High'],
                         self.data.iloc[self.current_step]['Low']], dtype=np.float32)
    '''
    action 0 : 매수
    action 1 : 관망
    action 2 : 매수

    마지막 step에선 현재 보유 중인 코인을 전부 매도
    '''
    def step(self, action):
        price = self.data.iloc[self.current_step]['Close']
        reward = 0
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        if action == 0 and self.balance > 0:
            self.avg_price = price
            self.holdings += self.balance/price
            self.balance = 0
        elif action == 2 and self.holdings > 0:
            self.balance += self.holdings*price
            reward = self.holdings*price - self.holdings*self.avg_price
            self.holdings = 0
            self.avg_price = 0
        
        if done and self.holdings > 0:
            reward = self.holdings*price - self.avg_price
            self.balance += self.holdings*price
            self.holdings = 0
        
        return self._next_observation(), reward, done, {}



if __name__ == '__main__':
    data = pd.read_csv(f'/workspace/data/raw/BTCUSDT/BTCUSDT-1h-2021.csv', index_col=0)
    data = data[['Open','High','Low','Close']]
    env = SpotTradingEnv(data)

    env.step(1)