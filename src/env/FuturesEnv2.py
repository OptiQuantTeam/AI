import numpy as np
import pandas as pd
import gym
from gym import spaces
from Logger import Logger

# position constant
LONG = 1
SHORT = -1
FLAT = 0

# action constant
BUY = 1
SELL = -1
HOLD = 0

class FuturesEnv2(gym.Env):
    def __init__(self, path=None, logger=None):
        self.logger = logger
        self.path = path
        self.initial_balance = 100000000
        self.actions = ['LONG', 'SHORT', 'FLAT']
        self.leverage = 2

        self.trade_fee = 0.0002
        self.load_data()
        self.returns_history = []
        self.max_steps = 1000
        self.action_space = spaces.Discrete(3, start=-1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.position_history = []  # 포지션 기록 추가

    def _load_data(self):
        data = pd.read_csv(self.path)
        self.data = self._preprocess_data(data)

    def _preprocess_data(self, df):
        """데이터 전처리: 결측치 처리 및 정규화"""
        # 필요한 컬럼만 선택
        #df = df[['Open', 'Close', 'Volume', 'CHG', 'stocRSI', 'MACD']]
        df = df[['Open', 'Close', 'High', 'Low', 'Volume', 'EMA_4_slope', \
                 'EMA_12_slope', 'EMA_24_slope', 'stochRSI', 'MACD', 'MACD_Signal', \
                    'Divergence Signal', 'Trade Signal', 'Cross Signal', 'bb_width', \
                        'bb_width_change', 'price_change']]

        
        # 결측치 처리
        #df = df.fillna(method='ffill')  # 앞의 값으로 채우기
        df = df.ffill()
        df = df.bfill()
        #df = df.fillna(method='bfill')  # 뒤의 값으로 채우기
        
        # 이상치 제거 (극단값 제거)
        #for column in ['Open', 'Close', 'Volume', 'CHG']:
        for column in ['Open', 'Close', 'High', 'Low', 'Volume', 'EMA_4_slope', \
                       'EMA_12_slope', 'EMA_24_slope', 'stochRSI', 'MACD', 'MACD_Signal', \
                        'Divergence Signal', 'Trade Signal', 'Cross Signal', 'bb_width', \
                            'bb_width_change', 'price_change']:
            
            q1 = df[column].quantile(0.01)
            q3 = df[column].quantile(0.99)
            df[column] = df[column].clip(q1, q3)
        
        # 정규화
        '''
        for column in ['Open', 'Close', 'Volume', 'CHG']:
            mean = df[column].mean()
            std = df[column].std()
            df[column] = (df[column] - mean) / (std + 1e-8)
        '''
        # stocRSI와 MACD는 이미 정규화된 형태이므로 극단값만 처리
        #df['stocRSI'] = df['stocRSI'].clip(0, 100)
        #df['MACD'] = df['MACD'].clip(-10, 10)  # 적절한 범위로 조정
            
        return df
        
    
    def reset(self):
        # 전체 데이터 길이에서 랜덤한 시작 위치 선택
        self.current_step = np.random.randint(36, len(self.data) - self.max_steps)
        # 초기 상태 설정
        self.balance = self.initial_balance
        self.position = FLAT
        self.action = HOLD
        self.size = 0
        self.entry_price = 1
        self.position_size = 0
        self.returns_history = [0]
        self.num = 0
        self.liquidated = False
        self.clear = False
        self.profit_rate_history = [0]
        self.position_history = []  # 포지션 기록 초기화
        self.logger.render(f"학습 시작 위치: {self.current_step} (전체 데이터 중 {self.current_step/len(self.data)*100:.2f}%)")
        #print(f"Start Step: {self.data.index[self.current_step]}\n")
        
        return self._next_observation()
    
    def _next_observation(self):

        def get_composite_change_score(data):
            short_term = np.mean(np.diff(data, n=1))

            mid_term = np.mean(np.diff(data, n=5))

            long_term = np.mean(np.diff(data, n=21))
            
            weights = [0.5, 0.3, 0.2]
            conposite_score = np.average([short_term, mid_term, long_term], weights=weights)
            return np.tanh(conposite_score)
        

        state = np.array([self.data.iloc[self.current_step]['Close'], get_composite_change_score([self.data.iloc[self.current_step-35:self.current_step+1]['Volume']]),
                          get_composite_change_score([self.data.iloc[self.current_step-35:self.current_step+1]['CHG']]), get_composite_change_score([self.data.iloc[self.current_step-35:self.current_step+1]['stocRSI']]), 
                          get_composite_change_score([self.data.iloc[self.current_step-35:self.current_step+1]['MACD']])], dtype=np.float32)
        return state
        
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        done = self.current_step >= len(self.data) - 1
        profit = 0
        reward = 0
        action = action[-1]
        #reward = -self.num
        
        
        if action > 0.3:
            position_direction = LONG
        elif action < -0.3:
            position_direction = SHORT
        else:
            position_direction = FLAT

        if self.position != position_direction:
            # 기존 포지션 청산
            if self.position != HOLD:
                profit = self.position * (current_price - self.entry_price) * self.size
                self.balance += profit
                '''
                if self.position_size != 0 and -position_direction * (current_price - self.entry_price) / self.entry_price > 0.7:
                    reward = profit * 100
                else:
                    reward = profit
                '''
                reward = self.position * (current_price - self.entry_price) / (self.entry_price * self.num * 10)
            
            profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
            self.profit_rate_history.append(profit_rate * 100)   
            

            if profit / self.entry_price < -0.1:
                #reward += np.exp((profit + 0.1)) - current_price
                reward += np.exp((profit/self.entry_price + 0.1)) - profit / self.entry_price / 10
            elif profit / self.entry_price > 0.2:
                #reward += np.exp(-(profit - 0.2)) + current_price
                reward += -np.exp(-(profit/self.entry_price - 0.2)) + profit / self.entry_price / 10

                
            if self.balance < self.initial_balance * 0.7:
                done = True
                reward += -100000
                self.liquidated = True
            elif self.balance > self.initial_balance * 1.5:
                done = True
                #reward += 10000 - self.num
                self.clear = True
            
            
            
            # 새로운 포지션 진입
            if position_direction != FLAT and not done:
                # 진입 비용 계산
                #trade_ratio = abs(action)
                trade_ratio = 1     # 전량
                position_size = self.balance * self.leverage * trade_ratio / current_price
                entry_cost = position_size * current_price * self.trade_fee
                self.balance -= entry_cost
                self.position = position_direction
                self.size = position_size
                self.entry_price = current_price
        self.position_history.append(self.position)  # 포지션 기록
        if done:
            self.returns_history.append(self.current_step)
        else:
            self.current_step += 1

        info = {
            'liquidated': self.liquidated,
            'clear': self.clear,
            'balance': self.balance,
            'profit_rate': float((self.balance - self.initial_balance) * 100 / self.initial_balance)
        }

        self.logger.render_step_state(f'    - num: {self.num}')
        self.logger.render_step_state(f'    - current_step: {self.current_step}')
        self.logger.render_step_state(f'    - action: {action}')
        self.logger.render_step_state(f'    - current_price: {current_price}')
        self.logger.render_step_state(f'    - balance: {self.balance}')  
        self.logger.render_step_state(f'    - profit: {profit}')
        self.logger.render_step_state(f'    - position: {self.position}')
        self.logger.render_step_state(f'    - position_direction: {position_direction}')
        self.logger.render_step_state(f'    - size: {self.size}')
        self.logger.render_step_state(f'    - entry_price: {self.entry_price}')
        self.logger.render_step_state(f'    - reward: {reward}')
        self.logger.render_step_state(f'    - done: {done}\n')

        
        # 다음 가격으로 포지션 가치 업데이트
        return self._next_observation(), float(reward), done, info

    def render(self):
        if self.logger is None:
            return
        # Render the environment to the screen    
        if self.liquidated:
            self.logger.error(f"  청산 여부: {'청산됨' if self.liquidated else '정상 종료'}")
        elif self.clear:
            self.logger.error(f"  목표 달성 : {'목표 달성' if self.clear else '달성 실패'}")
        else:
            self.logger.error(f'  마지막 데이터')
            
        profit = float(self.balance - self.initial_balance)
        profit_rate = float((self.balance - self.initial_balance) * 100 / self.initial_balance)
        self.logger.render(f'학습 마지막 위치: {self.current_step}')
        self.logger.render(f'Balance: {float(self.balance):.2f}')
        self.logger.render(f'Profit: {float(profit):.2f}, Profit Rate: {float(profit_rate):.2f}%')
        