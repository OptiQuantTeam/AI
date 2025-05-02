import numpy as np
import pandas as pd
import gym
from gym import spaces
from preprocess import preprocess_data
from Logger import Logger

# position constant
LONG = 1
SHORT = -1
FLAT = 0

# action constant
BUY = 1
SELL = -1
HOLD = 0

class FuturesEnv9(gym.Env):
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
        self.leverage = 1.2  # 레버리지 감소
        self.trade_fee = 0.0002
        self.max_steps = 2100
        self.max_position_ratio = 0.2  # 최대 포지션 크기 비율 감소
        self.stop_loss_threshold = 0.008  # 손절매 임계값 감소
        self.trailing_stop = 0.004  # 트레일링 스탑 감소
        
        # 상태 데이터 관련 설정
        self.ma_periods = [5, 20, 60]  # 이동평균선 기간
        self.bb_period = 20  # 볼린저 밴드 기간
        self.bb_std = 2  # 볼린저 밴드 표준편차
        self.atr_period = 14  # ATR 기간
        self.recent_trades = 10  # 최근 거래 기록 수
        self.vwap_period = 20  # VWAP 기간
        
        self._load_data()
        
        # 상태 공간 확장 (15개 특성)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(16,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3, start=-1)

        self.next_step = 0
        self.next = True
        self.last_step = np.random.randint(36, len(self.data)//4)
        self.save_step = self.last_step

    def _load_data(self):
        data = pd.read_csv(self.path)
        self.data = preprocess_data(data)
        
    
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
            max_price: 최고 가격 (트레일링 스탑용)
            min_price: 최저 가격 (트레일링 스탑용)

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
        self.success = 0
        self.failure = 0
        self.balance_profit_rate = 0
        self.max_price = 0  # 최고 가격 초기화
        self.min_price = float('inf')  # 최저 가격 초기화
        self.total_trade = 0
        
        self.returns_history = [0]
        self.reward_history = []
        self.balance_profit_rate_history = [0]
        self.position_history = []  # 포지션 기록 초기화
        self.price_history = []
        self.balance_history = []
        self.profit_history = []
        self.profit_rate_history = []
        
        self.logger.render(f"학습 시작 위치: {self.current_step} (전체 데이터 중 {self.current_step/len(self.data)*100:.2f}%)")
        
        return self._next_observation()
    
    def _next_observation(self):
        '''
        다음 상태 함수 - 기술적 지표 기반 상태 데이터
        Returns:
            state: 다음 상태 (15개 특성)
        '''
       
        # 기술적 지표만 사용 (총 15개)
        state = np.array([
            # 가격 관련 지표
            self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['Open'],
            self.data.iloc[self.current_step]['High'],
            self.data.iloc[self.current_step]['Low'],
            self.data.iloc[self.current_step]['Volume'],
            
            # 이동평균선
            self.data.iloc[self.current_step]['sma_5'],
            self.data.iloc[self.current_step]['sma_20'],
            self.data.iloc[self.current_step]['sma_60'],
            
            # RSI
            self.data.iloc[self.current_step]['RSI'],
            
            # MACD
            self.data.iloc[self.current_step]['MACD'],
            self.data.iloc[self.current_step]['MACD_Signal'],
            
            # 볼린저 밴드
            self.data.iloc[self.current_step]['bb_middle'],
            self.data.iloc[self.current_step]['bb_std'],
            self.data.iloc[self.current_step]['bb_upper'],
            self.data.iloc[self.current_step]['bb_lower'],
            
            # 가격 변화율 (수익률 예측을 위한 지표)
            (self.data.iloc[self.current_step]['Close'] - self.data.iloc[self.current_step-1]['Close']) / self.data.iloc[self.current_step-1]['Close'] if self.current_step > 0 else 0
        ], dtype=np.float32)
        
        return state
        
    def step(self, action):
        '''
        행동 함수
        Args:
            action: 행동 [-1, 0, 1]
            - 1 (LONG): 현재 Long이면 유지, Flat이면 Long 진입, Short이면 Flat으로 청산
            - -1 (SHORT): 현재 Long이면 Flat으로 청산, Flat이면 Short 진입, Short이면 유지
            - 0 (FLAT): 현재 포지션 유지
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
        trade_success = False
        profit_rate = 0
        target_profit_rate = 0.02
        
        # 기본 보상 초기화
        reward = 0
        action_reward = 0  # 행동 기반 보상
        
        if self.num > 12*24*7:
            done = True
        
        # 행동에 따른 포지션 방향 결정
        if action == LONG:
            if self.position == LONG:
                position_direction = LONG
            elif self.position == FLAT:
                position_direction = LONG
            else:
                position_direction = FLAT
        
        elif action == SHORT:
            if self.position == LONG:
                position_direction = FLAT
            elif self.position == FLAT:
                position_direction = SHORT
            else:
                position_direction = SHORT
        
        else:
            position_direction = self.position

        # 포지션 변경 시
        if self.position != position_direction:
            # 기존 포지션 청산
            if self.position != FLAT:
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                self.size = 0
                self.position = FLAT

                # 수익률 계산
                profit_rate = profit / self.entry_price
                
                # 수익률에 따른 보상 계산
                reward = profit_rate * 100  # 수익률을 퍼센트로 변환하여 보상으로 사용
                
                if profit_rate > 0:  # 수익인 경우
                    self.success += 1
                    trade_success = True
                    action_reward += 1.0  # 수익 실현에 대한 추가 보상
                else:  # 손실인 경우
                    self.failure += 1
                    if abs(profit_rate) > self.stop_loss_threshold:
                        action_reward -= 0.5  # 큰 손실에 대한 페널티
                
                self.total_trade += 1
                
            self.balance_profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
            
            # 새로운 포지션 진입
            if position_direction != FLAT and not done:
                trade_ratio = self.max_position_ratio
                available_balance = self.balance * (1 - self.trade_fee * self.leverage)
                position_size = available_balance * self.leverage * trade_ratio / current_price
                entry_cost = position_size * current_price * self.trade_fee
                self.balance -= entry_cost
                self.position = position_direction
                self.size = position_size
                self.entry_price = current_price
                
                if self.position == LONG:
                    self.max_price = current_price
                    # 기술적 지표 기반 진입 보상
                    if current_price > self.data.iloc[self.current_step]['sma_20']:
                        action_reward += 0.2  # 상승 추세에서 LONG 진입 보상
                else:
                    self.min_price = current_price
                    # 기술적 지표 기반 진입 보상
                    if current_price < self.data.iloc[self.current_step]['sma_20']:
                        action_reward += 0.2  # 하락 추세에서 SHORT 진입 보상
        
        # 포지션 유지 중일 때 손절매 및 트레일링 스탑 로직
        elif self.position != FLAT:
            # 손절매 로직
            if self.position == LONG and (current_price - self.entry_price) / self.entry_price < -self.stop_loss_threshold:
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                self.size = 0
                self.position = FLAT
                position_direction = FLAT
                profit_rate = profit / self.entry_price
                reward = profit_rate * 100
                action_reward += 0.5  # 손절매 실행에 대한 보상 (리스크 관리)
            elif self.position == SHORT and (self.entry_price - current_price) / self.entry_price < -self.stop_loss_threshold:
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                self.size = 0
                self.position = FLAT
                position_direction = FLAT
                profit_rate = profit / self.entry_price
                reward = profit_rate * 100
                action_reward += 0.5  # 손절매 실행에 대한 보상 (리스크 관리)

        # 최종 보상 계산 (수익률 기반 보상 + 행동 기반 보상)
        total_reward = reward + action_reward

        self.position_history.append(self.position)
        self.balance_profit_rate_history.append(self.balance_profit_rate * 100)  
        self.profit_history.append(profit)
        self.balance_history.append(self.balance)
        self.reward_history.append(total_reward)
        self.profit_rate_history.append(profit_rate*100)

        if done:
            self.returns_history.append(self.current_step)
            self.last_step = self.current_step
            
            # 최종 수익률에 따른 보상
            final_profit_rate = (self.balance - self.initial_balance) / self.initial_balance
            reward = final_profit_rate * 100
            
            if final_profit_rate > 0:
                self.next = True
                self.next_step += 1
            else:
                self.next = False
        else:
            self.current_step += 1

        info = {
            'liquidated': self.liquidated,
            'clear': self.clear,
            'balance': self.balance,
            'profit_rate': float((self.balance - self.initial_balance) * 100 / self.initial_balance),
            'trade_success': trade_success,
            'position': position_direction
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
        self.logger.render_step_state(f'    - reward: {total_reward}')
        self.logger.render_step_state(f'    - success: {self.success}')
        self.logger.render_step_state(f'    - failure: {self.failure}')
        self.logger.render_step_state(f'    - done: {done}\n')

        return self._next_observation(), float(total_reward), done, info

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
        self.logger.render(f'거래 횟수: {self.total_trade}, 학습 횟수: {self.next_step}')
