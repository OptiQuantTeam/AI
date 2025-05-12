import numpy as np
import pandas as pd
import gym
from gym import spaces
from Logger import Logger
import torch

# position constant
LONG = 1
SHORT = -1
FLAT = 0

# action constant
BUY = 1
SELL = -1
HOLD = 0

class FuturesEnv3(gym.Env):
    def __init__(self, path=None, logger=None):
        '''
        self.logger : 로그 기록 객체
        self.path : 환경 데이터
        self.initial_balance : 초기 자산
        self.actions : 행동 공간
        self.leverage : 레버리지
        self.trade_fee : 수수료
        self.load_data() : 데이터 로드
        self.returns_history : 수익률 기록
        self.max_steps : 최대 스텝
        self.action_space : 행동 공간
        self.observation_space : 관찰 공간
        '''

        self.logger = logger
        self.path = path
        self.initial_balance = 100000000
        self.actions = ['LONG', 'SHORT', 'FLAT']
        self.leverage = 2
        self.total_trade = 0
        self.trade_fee = 0.0002
        self._load_data()
        self.returns_history = []
        self.max_steps = 1000
        self.action_space = spaces.Discrete(3, start=-1)
        #self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.position_history = []  # 포지션 기록 추가
        self.last_step = 0  # last_step 속성 추가


    def _load_data(self):
        data = pd.read_csv(self.path)
        self.data = self._preprocess_data(data)

    def _preprocess_data(self, df):
        """데이터 전처리: 결측치 처리 및 정규화"""
        # 필요한 컬럼만 선택
        #df = df[['Open', 'Close', 'Volume', 'CHG', 'stocRSI', 'MACD']]
        columns=['Open', 'Close', 'High', 'Low', 'Volume',
                'ha_close', 'ha_open', 'ha_high', 'ha_low',
                'ha_body', 'ha_lower_wick', 'ha_upper_wick',
                'ha_signal', 'ema_200', 'ema_200_signal',
                'stoch_rsi', 'stoch_signal',
                'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
                'bb_width', 'bb_width_change']
        
        df = df[columns]

        
        # 결측치 처리
        #df = df.fillna(method='ffill')  # 앞의 값으로 채우기
        df = df.ffill()
        df = df.bfill()
        #df = df.fillna(method='bfill')  # 뒤의 값으로 채우기
        
        # 이상치 제거 (극단값 제거)
        #for column in ['Open', 'Close', 'Volume', 'CHG']:
        for column in columns:
            
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
        '''
        self.current_step : 현재 스텝, 랜덤한 위치에서 시작
        self.balance : 현재 자산
        self.position : 현재 행동에 따른 포지션
        self.action : 현재 행동
        self.size : 현재 포지션 크기
        self.entry_price : 진입 가격
        self.position_size : 현재 포지션 크기
        self.returns_history : 수익률 기록
        self.num : 현재 스텝
        self.liquidated : 청산 여부
        self.clear : 목표 달성 여부
        '''
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
        start_idx = max(0, self.current_step-35)
        window = self.data.iloc[start_idx:self.current_step+1]
        state = np.array([
            self.data.iloc[self.current_step]['ha_open'],
            self.data.iloc[self.current_step]['ha_close'],
            self.data.iloc[self.current_step]['ha_high'],
            self.data.iloc[self.current_step]['ha_low'],
            self.data.iloc[self.current_step]['ha_body'],
            self.data.iloc[self.current_step]['ha_lower_wick'],
            self.data.iloc[self.current_step]['ha_upper_wick'],
            self.data.iloc[self.current_step]['ema_200'],
            self.data.iloc[self.current_step]['ema_200_signal'],
            self.data.iloc[self.current_step]['stoch_rsi'],
            self.data.iloc[self.current_step]['stoch_signal']
        ], dtype=np.float32)
        return state
        
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        done = self.current_step >= len(self.data) - 1
        profit = 0
        reward = 0
        
        
        position_direction = action
        #action = action[-1]
        #reward = -self.num
        trade = HOLD
        '''
        # 행동에 따른 포지션 방향 결정
        if action > 0.3:
            position_direction = LONG
        elif action < -0.3:
            position_direction = SHORT
        elif action >= 0 and action <= 0.3:
            position_direction = FLAT
            trade = BUY
        else:
            position_direction = FLAT
            trade = SELL
        '''

        reward -= 1
        if self.position != position_direction:
            # 기존 포지션 청산
            if position_direction != FLAT or (trade != HOLD and self.position != trade):
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                self.size = 0
                reward = self.position * (current_price - self.entry_price) / (self.entry_price * self.num * 10)
                self.total_trade += 1

            profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
            self.profit_rate_history.append(profit_rate * 100)   
            
            if profit / self.entry_price < -0.1:
                #reward += np.exp((profit + 0.1)) - current_price
                reward += np.exp((profit/self.entry_price + 0.1)) - profit / self.entry_price / 10
            elif profit / self.entry_price > 0.2:
                #reward += np.exp(-(profit - 0.2)) + current_price
                reward += -np.exp(-(profit/self.entry_price - 0.2)) + profit / self.entry_price / 10

            if self.balance < self.initial_balance * 0.66:
                reward += -10000000
                done = True
                self.liquidated = True
            elif self.balance > self.initial_balance * 1.66:
                reward += 10000000
                done = True
                self.clear = True
            elif self.num >= 12*24*7:
                done = True
            

            # 새로운 포지션 진입
            if position_direction != FLAT and not done:
                # 진입 비용 계산
                #trade_ratio = abs(action)
                trade_ratio = 1     # 전량
                available_balance = self.balance * (1 - self.trade_fee * self.leverage)
                position_size = available_balance * self.leverage * trade_ratio / current_price
                entry_cost = position_size * current_price * self.trade_fee * self.leverage
                self.balance -= entry_cost
                self.position = LONG if position_direction == LONG else SHORT
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
            'profit_rate': float((self.balance - self.initial_balance) * 100 / self.initial_balance),
            'position': self.position
        }
        
        self.logger.render_step_state(f'    - num: {self.num}')
        self.logger.render_step_state(f'    - current_step: {self.current_step}')
        self.logger.render_step_state(f'    - action: {action}')        
        self.logger.render_step_state(f'    - balance: {self.balance}')  
        self.logger.render_step_state(f'    - profit: {profit * self.size}')
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
        self.logger.render(f'거래 횟수: {self.total_trade}')
        