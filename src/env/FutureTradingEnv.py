import gym
import numpy as np
import pandas as pd

LONG = 1
NOTHING = 0
SHORT = -1
class FutureTradingEnv(gym.Env):
    def __init__(self, data):
        super(FutureTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = 10000
        self.holdings = 0
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.position = NOTHING
        self.avg_price = 0
        self.fee_rate = 0.02
        self.leverage = 1
    
    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.holdings = 0
        self.position = NOTHING
        self.avg_price = 0

        return self._next_observation()
    
    def _next_observation(self):
        return np.array([self.balance, self.holdings, self.data.iloc[self.current_step]['Open'],
                         self.data.iloc[self.current_step]['Close'], self.data.iloc[self.current_step]['High'],
                         self.data.iloc[self.current_step]['Low']], dtype=np.float32)
    
    '''
    action 0 : BUY LONG
    action 1 : SELL LONG
    action 2 : HOLD
    action 3 : BUY SHORT
    action 4 : SELL SHORT
    '''
    def step(self, action):
        price = self.data.iloc[self.current_step]['Close']
        reward = 0
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        if action == 0 and self.position == NOTHING and self.balance > 0:
            self.avg_price = price
            self.holdings += (self.balance*self.leverage)/price
            self.balance = 0
            self.position = LONG
            
        elif action == 0 and self.position == SHORT and self.holdings > 0:
            #sell short
            self.balance += self.holdings*(2*self.avg_price-price)
            reward = self.holdings*(self.avg_price-price)
            self.holdings = 0
            #buy long
            self.avg_price = price
            self.holdings += (self.balance*self.leverage)/price
            self.balance = 0
            self.position = LONG
            
        elif action == 1 and self.position == LONG and self.holdings > 0:
            self.balance += self.holdings*price
            reward = self.holdings*(price-self.avg_price)
            self.holdings = 0
            self.avg_price = 0
            self.position = NOTHING
            
        elif action == 3 and self.position == NOTHING and self.balance > 0:
            self.avg_price = price
            self.holdings += (self.balance*self.leverage)/price
            self.balance = 0
            self.position = SHORT
            
        elif action == 3 and self.position == LONG and self.holdings > 0:
            #sell long
            self.balance += self.holdings*price
            reward = self.holdings*(price-self.avg_price)
            self.holdings = 0
            self.avg_price = 0
            #buy short
            self.avg_price = price
            self.holdings += (self.balance*self.leverage)/price
            self.balance = 0
            self.position = SHORT
            
        elif action == 4 and self.position == SHORT and self.holdings > 0:
            self.balance += self.holdings*(2*self.avg_price-price)
            reward = self.holdings*(self.avg_price-price)
            self.holdings = 0
            self.position = NOTHING
            

        if done and self.position == SHORT and self.holdings > 0:
            reward = self.holdings*(self.avg_price-price)
            self.balance += self.holdings*price
            self.holdings = 0
        elif done and self.position == LONG and self.holdings > 0:
            reward = self.holdings*(price-self.avg_price)
            self.balance += self.holdings*price
            self.holdings = 0

        if reward > 0:
            #print(f'{reward}')
            reward *= 5

        return self._next_observation(), reward, done, {}