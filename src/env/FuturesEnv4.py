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

class FuturesEnv4(gym.Env):
    def __init__(self, path=None, logger=None):
        '''
        환경의 초기 설정값
        Args:
            path: 데이터 파일 경로
            logger: 로거 객체
            initial_balance: 초기 자산
            actions: 행동 종류
            leverage: 레버리지
            trade_fee: 거래 수수료
            max_steps: 마지막 데이터 위치에서 최대 스텝
            action_space: 행동 공간
            observation_space: 관찰 공간
        '''
        self.path = path
        self.logger = logger
        self.initial_balance = 100000000
        self.actions = ['LONG', 'SHORT', 'FLAT']
        self.leverage = 2
        self.trade_fee = 0.0002
        self.max_steps = 1000

        self._load_data()
        
        self.action_space = spaces.Discrete(3, start=-1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

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
        '''
        환경 초기화 함수
        Args:
            current_step: 현재 스텝
            last_step: 마지막 스텝
            balance: 현재 자산
            position: 현재 포지션
            action: 현재 행동
            size: 현재 포지션 크기
            position_size: 현재 포지션 크기
            entry_price: 진입 가격
            num: 반복한 스텝
            liquidated: 청산 여부
            clear: 목한 달성 여부
            stay_count: 유지 기간
            success: 성공 횟수
            failure: 실패 횟수
            balance_profit_rate: 수익률

            returns_history: 보상 기록
            reward_history: 보상 기록
            profit_rate_history: 수익률 기록
            position_history: 포지션 기록
            price_history: 가격 기록
            balance_history: 자산 기록
            profit_history: 이익 기록
        Returns:
            state: 다음 상태
        '''
        
        self.current_step = np.random.randint(36, len(self.data) - self.max_steps)
        self.last_step = self.current_step
        self.balance = self.initial_balance
        self.position = FLAT
        self.action = HOLD
        self.size = 0
        self.position_size = 0
        self.entry_price = 1
        self.num = 0
        self.liquidated = False
        self.clear = False
        self.stay_count = 0
        self.success = 0
        self.failure = 0
        self.balance_profit_rate = 0
        
        self.returns_history = [0]
        self.reward_history = []
        self.balance_profit_rate_history = [0]
        self.position_history = []  # 포지션 기록 초기화
        self.price_history = []
        self.balance_history = []
        self.profit_history = []
        
        self.logger.render(f"학습 시작 위치: {self.current_step} (전체 데이터 중 {self.current_step/len(self.data)*100:.2f}%)")
        
        return self._next_observation()
    
    def _next_observation(self):
        '''
        다음 상태 함수
        Args:
            None
        Returns:
            state: 다음 상태
        '''
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
        '''
        행동 함수
        Args:
            action: 행동 [-1, 0, 1]
        Returns:
            state: 다음 상태
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        '''

        current_price = self.data.iloc[self.current_step]['Close']
        self.price_history.append(current_price)
        done = self.current_step >= len(self.data) - 1
        profit = 0

        # 기본 보상 설정 - 포지션 유지 시 약간의 페널티
        if self.position != FLAT:
            reward = -0.1  # 포지션 유지 시 약간의 페널티
        else:
            reward = 0  # 중립 포지션은 보상 없음

        if self.num > 12*24*7:
            done = True
        
        # 행동에 따른 포지션 방향 결정
        if action > 0.8:
            position_direction = LONG
        elif action < -0.8:
            position_direction = SHORT
        else:
            position_direction = FLAT

        # 포지션 변경 시
        if self.position != position_direction:
            # 기존 포지션 청산
            if self.position != FLAT:
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                self.size = 0
                self.position = FLAT

                # 수익에 따른 보상 계산
                profit_rate = profit / self.entry_price
                
                # 수익이 발생한 경우 더 큰 보상
                if profit > 0.05:
                    reward += profit_rate * 10  # 수익에 비례한 큰 보상
                    self.success += 1
                else:
                    reward += profit_rate * 5  # 손실에 비례한 작은 페널티
                    self.failure += 1
                
                # 연속 손실에 대한 추가 페널티
                if profit < 0 and self.failure > 3:
                    reward -= 2  # 연속 손실에 대한 추가 페널티
            
            self.balance_profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
            
            # 청산 조건 - 큰 손실 방지
            if self.balance_profit_rate < -0.05:
                reward -= 20  # 큰 손실에 대한 큰 페널티
                #done = True
                #self.liquidated = True
            # 목표 달성 조건 - 수익 실현
            elif self.balance_profit_rate > 0.1:
                reward += 15  # 목표 달성에 대한 큰 보상
                #done = True
                #self.clear = True

            # 새로운 포지션 진입
            if position_direction != FLAT and not done and self.stay_count <= 0:
                # 진입 비용 계산
                trade_ratio = 1     # 전량
                available_balance = self.balance * (1 - self.trade_fee * self.leverage)
                position_size = available_balance * self.leverage * trade_ratio / current_price
                entry_cost = position_size * current_price * self.trade_fee
                self.balance -= entry_cost
                self.position = position_direction
                self.size = position_size
                self.entry_price = current_price
                self.stay_count = 7
                
                # 포지션 진입 시 약간의 보상 (탐색 유도)
                reward += 0.5
        
        self.stay_count -= 1

        self.position_history.append(self.position)  # 포지션 기록
        self.balance_profit_rate_history.append(self.balance_profit_rate * 100)  
        self.profit_history.append(profit)
        self.balance_history.append(self.balance)
        self.reward_history.append(reward)

        if done:
            self.returns_history.append(self.current_step)
            self.last_step = self.current_step
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
        self.logger.render_step_state(f'    - success: {self.success}')
        self.logger.render_step_state(f'    - failure: {self.failure}')
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
        
