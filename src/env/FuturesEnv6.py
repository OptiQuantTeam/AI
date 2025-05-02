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

class FuturesEnv6(gym.Env):
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
            shape=(15,), 
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
        

        if self.next:
            if self.last_step < len(self.data) - 1:
                self.current_step = self.last_step
                self.save_step = self.last_step
            else:
                self.current_step = np.random.randint(36, len(self.data) // 2)
                self.save_step = self.current_step
        else:
            self.current_step = self.save_step
        #self.current_step = np.random.randint(36, len(self.data) - self.max_steps)
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
        다음 상태 함수 - 확장된 상태 데이터
        Returns:
            state: 다음 상태 (15개 특성)
        '''
        def get_composite_change_score(data):
            short_term = np.mean(np.diff(data, n=1))
            mid_term = np.mean(np.diff(data, n=5))
            long_term = np.mean(np.diff(data, n=21))
            weights = [0.5, 0.3, 0.2]
            composite_score = np.average([short_term, mid_term, long_term], weights=weights)
            return np.tanh(composite_score)

        # 기본 가격 데이터
        current_price = self.data.iloc[self.current_step]['Close']
        current_volume = self.data.iloc[self.current_step]['Volume']
        
        # 1. 가격 기반 지표
        lookback = max(60, self.bb_period)  # 가장 긴 기간에 맞춤
        price_data = self.data.iloc[self.current_step-lookback:self.current_step+1]['Close']
        volume_data = self.data.iloc[self.current_step-lookback:self.current_step+1]['Volume']
        
        # 이동평균선
        mas = [np.mean(price_data[-period:]) for period in self.ma_periods]
        ma_features = [(current_price - ma) / ma for ma in mas]  # 현재 가격과의 차이
        
        # 볼린저 밴드
        bb_ma = np.mean(price_data[-self.bb_period:])
        bb_std = np.std(price_data[-self.bb_period:])
        bb_upper = bb_ma + self.bb_std * bb_std
        bb_lower = bb_ma - self.bb_std * bb_std
        bb_features = [
            (current_price - bb_upper) / bb_upper,  # 상단 밴드와의 거리
            (current_price - bb_lower) / bb_lower   # 하단 밴드와의 거리
        ]
        
        # ATR (변동성)
        high_low = self.data.iloc[self.current_step-self.atr_period:self.current_step+1]['High'] - \
                  self.data.iloc[self.current_step-self.atr_period:self.current_step+1]['Low']
        high_close = np.abs(self.data.iloc[self.current_step-self.atr_period:self.current_step+1]['High'] - \
                          self.data.iloc[self.current_step-self.atr_period:self.current_step+1]['Close'].shift(1))
        low_close = np.abs(self.data.iloc[self.current_step-self.atr_period:self.current_step+1]['Low'] - \
                         self.data.iloc[self.current_step-self.atr_period:self.current_step+1]['Close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = np.mean(true_range)
        atr_feature = atr / current_price  # 현재 가격 대비 변동성
        
        # VWAP (거래량 가중 가격)
        vwap_data = self.data.iloc[self.current_step-self.vwap_period:self.current_step+1]
        vwap = np.sum(vwap_data['Close'] * vwap_data['Volume']) / np.sum(vwap_data['Volume'])
        vwap_feature = (current_price - vwap) / vwap  # 현재 가격과 VWAP의 차이
        
        # 2. 거래량 기반 지표
        volume_score = get_composite_change_score(volume_data[-35:])  # 최근 35개 데이터만 사용
        
        # OBV (On Balance Volume)
        price_diff = price_data.diff()
        obv = np.sum(np.where(price_diff > 0, volume_data, 
                             np.where(price_diff < 0, -volume_data, 0)))
        obv_feature = obv / np.mean(volume_data)  # 평균 거래량 대비 OBV
        
        # 3. 포지션 관련 지표
        position_profit = 0
        if self.position != FLAT:
            position_profit = (current_price - self.entry_price) / self.entry_price * self.position
        
        # 4. 리스크 관리 지표
        risk_features = [
            self.balance_profit_rate,  # 전체 수익률
            self.failure / (self.success + self.failure + 1e-6),  # 최근 거래 승률
            position_profit  # 현재 포지션 수익률
        ]
        
        # 5. 추가 시장 상태 지표
        market_state = [
            self.data.iloc[self.current_step]['stocRSI'],  # 스토캐스틱 RSI
            self.data.iloc[self.current_step]['MACD'],  # MACD
            vwap_feature  # VWAP 대비 현재 가격
        ]
        
        # 모든 특성 결합 (총 15개)
        state = np.array([
            current_price / self.entry_price - 1,  # 현재 가격 변화율
            *ma_features,  # 3개 이동평균선 특성
            *bb_features,  # 2개 볼린저 밴드 특성
            atr_feature,  # ATR 특성
            volume_score,  # 거래량 변화 점수
            obv_feature,  # OBV 특성
            *market_state,  # 3개 시장 상태 지표
            *risk_features  # 3개 리스크 관리 특성
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
        trade_success = False  # 거래 성공 여부

        # 기본 보상 설정 - 포지션 유지 시 약간의 페널티
        if self.position != FLAT:
            reward = -0.1  # 포지션 유지 시 페널티 증가
        else:
            reward = 0.2  # 중립 포지션은 더 큰 보상 (안전한 포지션 장려)

        if self.num > 12*24*7:
            done = True
        
        # 행동에 따른 포지션 방향 결정
        if action == LONG:  # action = 1 (LONG)
            if self.position == LONG:
                # Long 포지션 유지
                position_direction = LONG
            elif self.position == FLAT:
                # Flat에서 Long으로 진입
                position_direction = LONG
            else:  # self.position == SHORT
                # Short에서 Flat으로 청산
                position_direction = FLAT
        
        elif action == SHORT:  # action = -1 (SHORT)
            if self.position == LONG:
                # Long에서 Flat으로 청산
                position_direction = FLAT
            elif self.position == FLAT:
                # Flat에서 Short으로 진입
                position_direction = SHORT
            else:  # self.position == SHORT
                # Short 포지션 유지
                position_direction = SHORT
        
        else:  # action = 0 (FLAT)
            # 현재 포지션 유지
            position_direction = self.position

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
                if profit > 0:
                    reward += profit_rate * 15  # 수익에 비례한 보상 감소
                    self.success += 1
                    trade_success = True  # 거래 성공
                else:
                    reward += profit_rate * 10  # 손실에 비례한 페널티 감소
                    self.failure += 1
                
                # 연속 손실에 대한 추가 페널티
                if profit < 0 and self.failure > 1:
                    reward -= 15  # 연속 손실에 대한 페널티 증가
            
            self.balance_profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
            
            # 청산 조건 - 큰 손실 방지
            if self.balance_profit_rate < -0.01:  # 손실 허용 범위 감소
                reward -= 50  # 큰 손실에 대한 페널티 증가
                #done = True
                #self.liquidated = True
            # 목표 달성 조건 - 수익 실현
            elif self.balance_profit_rate > 0.02:  # 목표 수익률 감소
                reward += 20  # 목표 달성에 대한 보상 감소
                #done = True
                #self.clear = True

            # 새로운 포지션 진입
            if position_direction != FLAT and not done:  # stay_count 체크 제거
                # 진입 비용 계산
                trade_ratio = self.max_position_ratio  # 자본의 일부만 사용
                available_balance = self.balance * (1 - self.trade_fee * self.leverage)
                position_size = available_balance * self.leverage * trade_ratio / current_price
                entry_cost = position_size * current_price * self.trade_fee
                self.balance -= entry_cost
                self.position = position_direction
                self.size = position_size
                self.entry_price = current_price
                
                # 최고/최저 가격 초기화
                if self.position == LONG:
                    self.max_price = current_price
                else:
                    self.min_price = current_price
                
                # 포지션 진입 시 약간의 보상 (탐색 유도)
                reward += 0.2  # 진입 보상 감소
        
        # 포지션 유지 중일 때 손절매 및 트레일링 스탑 로직
        elif self.position != FLAT:
            # 손절매 로직
            if self.position == LONG and (current_price - self.entry_price) / self.entry_price < -self.stop_loss_threshold:
                # 손절매 실행
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                self.size = 0
                self.position = FLAT
                position_direction = FLAT
                reward += 8  # 손절매 실행에 대한 보상 증가
                self.logger.render_step_state(f'    - 손절매 실행: {(current_price - self.entry_price) / self.entry_price:.4f}')
            elif self.position == SHORT and (self.entry_price - current_price) / self.entry_price < -self.stop_loss_threshold:
                # 손절매 실행
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                self.size = 0
                self.position = FLAT
                position_direction = FLAT
                reward += 8  # 손절매 실행에 대한 보상 증가
                self.logger.render_step_state(f'    - 손절매 실행: {(self.entry_price - current_price) / self.entry_price:.4f}')
            
            # 트레일링 스탑 로직
            elif self.position == LONG:
                # 최고 가격 업데이트
                if current_price > self.max_price:
                    self.max_price = current_price
                # 트레일링 스탑 실행
                elif (self.max_price - current_price) / self.max_price > self.trailing_stop:
                    profit = self.position * (current_price - self.entry_price)
                    self.balance += profit * self.size
                    self.size = 0
                    self.position = FLAT
                    position_direction = FLAT
                    reward += 15  # 트레일링 스탑 실행에 대한 보상 증가
                    trade_success = profit > 0  # 수익이 발생했으면 거래 성공
                    self.logger.render_step_state(f'    - 트레일링 스탑 실행: {(self.max_price - current_price) / self.max_price:.4f}')
            elif self.position == SHORT:
                # 최저 가격 업데이트
                if current_price < self.min_price:
                    self.min_price = current_price
                # 트레일링 스탑 실행
                elif (current_price - self.min_price) / self.min_price > self.trailing_stop:
                    profit = self.position * (current_price - self.entry_price)
                    self.balance += profit * self.size
                    self.size = 0
                    self.position = FLAT
                    position_direction = FLAT
                    reward += 15  # 트레일링 스탑 실행에 대한 보상 증가
                    trade_success = profit > 0  # 수익이 발생했으면 거래 성공
                    self.logger.render_step_state(f'    - 트레일링 스탑 실행: {(current_price - self.min_price) / self.min_price:.4f}')
        
        self.position_history.append(self.position)  # 포지션 기록
        self.balance_profit_rate_history.append(self.balance_profit_rate * 100)  
        self.profit_history.append(profit)
        self.balance_history.append(self.balance)
        self.reward_history.append(reward)

        if done:
            self.returns_history.append(self.current_step)
            self.last_step = self.current_step
            
            # 종료 시 최종 수익률에 따른 추가 보상/페널티
            final_profit_rate = (self.balance - self.initial_balance) / self.initial_balance
            if final_profit_rate > 0:
                reward += 20  # 수익에 대한 보상 감소
                self.next = True
                self.next_step += 1
            elif final_profit_rate < -0.01:  # 손실 허용 범위 감소
                reward -= 40  # 손실에 대한 페널티 증가
                self.next = False
        else:
            self.current_step += 1

        info = {
            'liquidated': self.liquidated,
            'clear': self.clear,
            'balance': self.balance,
            'profit_rate': float((self.balance - self.initial_balance) * 100 / self.initial_balance),
            'trade_success': trade_success  # 거래 성공 여부 추가
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
        self.logger.render(f'학습 횟수: {self.next_step}')
