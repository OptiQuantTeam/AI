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

class FuturesEnv9(gym.Env):
    def __init__(self, path=None, logger=None):
        '''
        환경의 초기 설정값
        Variables:
            path: 데이터 파일 경로
            logger: 로거 객체
            initial_balance: 초기 자산
            actions: 행동 종류
            initial_leverage: 초기 레버리지
            leverage: 레버리지
            min_leverage: 최소 레버리지
            max_leverage: 최대 레버리지
            leverage_step: 레버리지 조정 단위
            trade_fee: 거래 수수료
            max_steps: 마지막 데이터 위치에서 최대 스텝
            min_steps: PPO 배치 사이즈를 고려한 최소 스텝 수
            profit_target: 수익 목표
            max_position_ratio: 최대 포지션 크기 비율 감소
            stop_loss_threshold: 손절매 임계값 감소
            action_space: 행동 공간
            observation_space: 관찰 공간
            data: 환경 데이터
            success_episodes: 다음 스텝 시작 위치
            last_step: 마지막 스텝(학습 시작 위치)
        '''

        self.path = path
        self.logger = logger
        self.initial_balance = 1000000
        self.actions = ['LONG', 'SHORT', 'FLAT']
        self.initial_leverage = 2  # 초기 레버리지
        self.leverage = self.initial_leverage  # 현재 레버리지
        self.min_leverage = 1.0  # 최소 레버리지
        self.max_leverage = 2.0  # 최대 레버리지
        self.leverage_step = 0.2  # 레버리지 조정 단위
        self.trade_fee = 0.0002
        self.max_steps = 2100
        self.min_steps = 2048  # PPO 배치 사이즈를 고려한 최소 스텝 수
        self.profit_target = 0.1  # 목표 수익률 (10%)
        self.max_position_ratio = 0.2  # 최대 포지션 크기 비율 감소
        self.stop_loss_threshold = 0.02  # 손절매 임계값 감소
        self.recurrence = 0
               
        # 레버리지에 따른 손실 제한 관련 파라미터
        self.leverage_loss_limits = {
            2.0: 0.05,  # 2배 레버리지일 때 5%
            1.8: 0.045, # 1.8배 레버리지일 때 4.5%
            1.6: 0.04,  # 1.6배 레버리지일 때 4%
            1.4: 0.035, # 1.4배 레버리지일 때 3.5%
            1.2: 0.03,  # 1.2배 레버리지일 때 3%
            1.0: 0.025  # 1배 레버리지일 때 2.5%
        }
        
        self._load_data()
        
        # 상태 공간 확장 (변동성과 다중 시간프레임 정보 추가)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(9,),  # 9개 feature
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3, start=-1)

        self.success_episodes = 0
        self.last_step = 100
        self.consecutive_losses = 0  # 연속 손실 횟수

    def _load_data(self):
        '''
        데이터 로드 및 전처리
        '''
        data = pd.read_csv(self.path)
        self.data = self._preprocess_data(data)

    def _preprocess_data(self, df):
        '''
        데이터 전처리: 결측치 처리 및 정규화
        '''
        columns = [
            'Close', 'Volume', 'EMA_4_slope', 'EMA_12_slope', 'EMA_24_slope',
            'stochRSI', 'MACD', 'MACD_Signal', 'bb_width'
        ]
        df = df[columns]
        df = df.ffill().bfill()
        for column in columns:
            q1 = df[column].quantile(0.01)
            q3 = df[column].quantile(0.99)
            df[column] = df[column].clip(q1, q3)
        return df
    
    def _adjust_loss_limit(self):
        '''
        레버리지에 따라 손실 한도를 조정하는 함수
        '''
        # 가장 가까운 레버리지 단계 찾기
        closest_leverage = min(self.leverage_loss_limits.keys(), 
                             key=lambda x: abs(x - self.leverage))
        
        # 해당 레버리지에 맞는 손실 한도 설정
        self.stop_loss_threshold = self.leverage_loss_limits[closest_leverage]
        #self.logger.render(f"현재 레버리지: {self.leverage:.1f}x, 손실 한도: {self.stop_loss_threshold*100:.1f}%")

    def _adjust_leverage(self, profit_rate):
        '''
        수익률에 따라 레버리지를 조정하는 함수
        Args:
            profit_rate: 현재 거래의 수익률
        '''
        if profit_rate < 0.01:  # 손실인 경우
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # 연속 손실에 따라 레버리지 감소
            if self.consecutive_losses >= 2:
                self.leverage = max(self.min_leverage, self.leverage - self.leverage_step)
                self._adjust_loss_limit()  # 레버리지 변경 시 손실 한도 재조정
                #self.logger.render(f"연속 손실로 레버리지 감소: {self.leverage:.2f}")
        else:  # 수익인 경우
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # 연속 수익에 따라 레버리지 증가
            if self.consecutive_wins >= 2:
                self.leverage = min(self.max_leverage, self.leverage + self.leverage_step)
                self._adjust_loss_limit()  # 레버리지 변경 시 손실 한도 재조정
                #self.logger.render(f"연속 수익으로 레버리지 증가: {self.leverage:.2f}")

    def reset(self):
        '''
        환경 초기화 함수
        Variables:
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
            clear: 목표 달성 여부
            success: 성공 횟수
            failure: 실패 횟수
            balance_profit_rate: 수익률
            total_trade: 총 거래 횟수
            consecutive_wins: 연속 수익 횟수
            consecutive_losses: 연속 손실 횟수
            leverage: 레버리지 초기화

            reward_history: 현재 에피소드의 step 별 보상 기록
            balance_profit_rate_history: 현재 에피소드의 step 별 자산 수익률 기록
            position_history: 현재 에피소드의 step 별 포지션 기록
            price_history: 현재 에피소드의 step 별 가격 기록
            balance_history: 현재 에피소드의 step 별 자산 기록
            profit_history: 현재 에피소드의 step 별 수익 기록
            profit_rate_history: 현재 에피소드의 step 별 수익률 기록
            returns_history: 현재 에피소드의 step 별 수익률 기록
        Returns:
            state: 다음 상태
        '''

        
        
        
        if self.recurrence < 10 and self.recurrence > 0:
            self.current_step = self.tmp_current
            self.recurrence += 1
        else:
            self.current_step = np.random.randint(36, len(self.data) - self.max_steps)
            self.recurrence = 0 if self.recurrence == 10 else self.recurrence + 1
        
        '''

        self.current_step = self.last_step + 1
        if self.current_step >= len(self.data) - self.max_steps:
            self.current_step = 100
        '''

        self.last_step = self.current_step
        self.tmp_current = self.current_step
        self.balance = self.initial_balance
        self.position = FLAT
        self.action = HOLD
        self.size = 1e-6
        self.position_size = 0
        self.entry_price = 1
        self.num = 0
        self.liquidated = False
        self.clear = False
        self.success = 0
        self.failure = 0
        self.balance_profit_rate = 0
        self.total_trade = 0
        self.consecutive_wins = 0  # 연속 수익 횟수 초기화
        self.consecutive_losses = 0  # 연속 손실 횟수 초기화
        self.leverage = self.initial_leverage
        
        self._adjust_loss_limit()  # 레버리지에 따른 손실 한도 조정
        
        self.reward_history = []
        self.balance_profit_rate_history = [0]
        self.position_history = []  # 포지션 기록 초기화
        self.price_history = []
        self.balance_history = []
        self.profit_history = []
        self.profit_rate_history = []
        self.returns_history = []  # 이 줄 추가
        
        self.logger.render(f"학습 시작 위치: {self.current_step} (전체 데이터 중 {self.current_step/len(self.data)*100:.2f}%)")
        self.logger.render(f"현재 손실 한도: {self.stop_loss_threshold*100:.1f}%")
        
        return self._next_observation()
    
    def _next_observation(self):
        '''
        다음 상태 함수 - 기술적 지표 기반 상태 데이터
        Returns:
            state: 다음 상태 (9개 특성)
        '''
       
        # 기술적 지표만 사용 (총 9개)
        state = np.array([
            self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['Volume'],
            self.data.iloc[self.current_step]['EMA_4_slope'],
            self.data.iloc[self.current_step]['EMA_12_slope'],
            self.data.iloc[self.current_step]['EMA_24_slope'],
            self.data.iloc[self.current_step]['stochRSI'],
            self.data.iloc[self.current_step]['MACD'],
            self.data.iloc[self.current_step]['MACD_Signal'],
            self.data.iloc[self.current_step]['bb_width'],
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
        done = self.current_step >= len(self.data) - 1
        profit = 0
        reward = 0
        trade_success = False  # 초기화 추가
        
        # action을 position_direction으로 변환
        position_direction = action
        
        # 1. 행동에 대한 보상 제거 (실제 수익/손실에만 집중)
        action_reward = 0
        
        # 2. 포지션 진입/청산 시 보상
        if self.position != position_direction:
            self.total_trade += 1  # 거래 횟수 증가
            
            if self.position != FLAT:
                profit = self.position * (current_price - self.entry_price)
                profit_rate = profit / self.entry_price
                
                # 수익/손실에 대한 보상 강화
                if profit_rate > 0.01:  # 1% 이상 수익
                    reward = profit_rate * 2  # 수익률의 2배
                    self.success += 1
                    trade_success = True
                elif profit_rate < -0.01:  # 1% 이상 손실
                    reward = profit_rate * 3  # 손실률의 3배 (더 큰 페널티)
                    self.failure += 1
                else:
                    reward = profit_rate  # 작은 수익/손실은 그대로 반영
                
                # 연속 손실에 대한 추가 페널티 (중복 제거)
                if profit_rate < 0:
                    self.consecutive_losses += 1
                    if self.consecutive_losses > 3:  # 3번 연속 손실
                        reward *= 1.5  # 페널티 50% 증가
                else:
                    self.consecutive_losses = 0
        
        # 3. 미실현 손익에 대한 보상 수정
        if self.position != FLAT:
            unrealized_profit = self.position * (current_price - self.entry_price) * self.size / self.entry_price
            # 미실현 손익이 클수록 보상 감소 (불필요한 포지션 유지 방지)
            position_reward = unrealized_profit * (1 - abs(unrealized_profit))
            reward += position_reward
        
        # 4. 추가 보상/페널티
        if self.balance > self.initial_balance * (1 + self.profit_target):
            reward += 1.0  # 목표 수익 달성 보상
            self.clear = True
        elif self.balance < self.initial_balance * 0.7:
            reward -= 2.0  # 청산 페널티
            self.liquidated = True
        
        # 5. 포지션 업데이트 전 손절매 체크
        if self.position != FLAT:
            current_profit_rate = self.position * (current_price - self.entry_price) / self.entry_price
            
            # 손절매 조건 체크
            if current_profit_rate < -self.stop_loss_threshold:
                # 손절매 실행
                profit = self.position * (current_price - self.entry_price) * self.size
                self.balance += profit
                self.position = FLAT
                self.size = 1e-6
                self.entry_price = current_price
                reward -= 1.0  # 손절매에 대한 페널티
                self.failure += 1
                done = True  # 에피소드 종료
                self.liquidated = True
                
                # 히스토리 업데이트 추가
                self.price_history.append(current_price)
                self.balance_history.append(self.balance)
                self.profit_history.append(profit)
                self.profit_rate_history.append(current_profit_rate)
                self.position_history.append(self.position)
                self.reward_history.append(reward)
                
                # info 딕셔너리 생성
                info = {
                    'liquidated': self.liquidated,
                    'clear': self.clear,
                    'balance': self.balance,
                    'profit_rate': float((self.balance - self.initial_balance) * 100 / self.initial_balance),
                    'trade_success': trade_success,
                    'position': position_direction
                }
                
                # 학습 상태 기록
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
                
                return self._next_observation(), float(reward), done, info
        
        # 5. 포지션 업데이트
        if self.position != position_direction:
            # 기존 포지션 청산
            if self.position != FLAT:
                profit = self.position * (current_price - self.entry_price) * self.size
                self.balance += profit
                
            # 새로운 포지션 진입
            if position_direction != FLAT and not done:
                # 레버리지 조정
                self._adjust_leverage(profit_rate if 'profit_rate' in locals() else 0)
                
                trade_ratio = abs(action)
                position_size = self.balance * self.leverage * trade_ratio / current_price
                entry_cost = position_size * current_price * self.trade_fee
                
                # 포지션 크기 제한
                max_position_size = self.balance * self.max_position_ratio
                position_size = min(position_size, max_position_size)
                
                self.balance -= entry_cost
                self.position = position_direction
                self.size = position_size
                self.entry_price = current_price
        
        if done:
            self.last_step = self.current_step
            
            # 최종 수익률에 따른 보상
            final_profit_rate = (self.balance - self.initial_balance) / self.initial_balance
            
            # 학습 종료 시 다음 학습 step 여부 결정
            if final_profit_rate > 0.01:
                self.success_episodes += 1
        else:
            self.current_step += 1

        # 학습 종료 시 추가 정보
        info = {
            'liquidated': self.liquidated,
            'clear': self.clear,
            'balance': self.balance,
            'profit_rate': float((self.balance - self.initial_balance) * 100 / self.initial_balance),
            'trade_success': trade_success,
            'position': position_direction
        }

        # 학습 종료 시 학습 상태 기록
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
        self.logger.render(f'거래 횟수: {self.total_trade}, 성공 횟수: {self.success_episodes}')