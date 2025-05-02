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

class FuturesEnv7(gym.Env):
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
        self.leverage = 1.2
        self.trade_fee = 0.0002
        self.max_steps = 2100
        self.max_position_ratio = 0.3  # 포지션 크기 증가
        self.stop_loss_threshold = 0.015  # 손절매 임계값 완화
        self.trailing_stop = 0.008  # 트레일링 스탑 완화
        
        # 캔들스틱 패턴 관련 설정
        self.lookback = 10
        self.pattern_types = 5
        self.features_per_candle = 6
        
        self._load_data()
        
        # 상태 공간 정의 (캔들스틱 패턴 기반)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.lookback * (self.features_per_candle + self.pattern_types),), 
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
        
    def _calculate_body(self, row: pd.Series) -> float:
        """캔들 몸통 크기 계산"""
        return abs(row['Close'] - row['Open'])
    
    def _calculate_upper_shadow(self, row: pd.Series) -> float:
        """윗꼬리 길이 계산"""
        return row['High'] - max(row['Open'], row['Close'])
    
    def _calculate_lower_shadow(self, row: pd.Series) -> float:
        """아랫꼬리 길이 계산"""
        return min(row['Open'], row['Close']) - row['Low']
    
    def _is_bullish(self, row: pd.Series) -> bool:
        """상승 캔들 여부 확인"""
        return row['Close'] > row['Open']
    
    def _identify_pattern(self, current: pd.Series, previous: pd.Series = None) -> str:
        """캔들스틱 패턴 식별 - 더 유연한 조건 적용"""
        body = self._calculate_body(current)
        upper = self._calculate_upper_shadow(current)
        lower = self._calculate_lower_shadow(current)
        total_length = body + upper + lower
        
        # 도지 패턴 - 더 유연한 조건
        if body <= total_length * 0.2:  # 기존 0.1에서 0.2로 완화
            return 'doji'
            
        # 해머 패턴 - 더 유연한 조건
        if (lower >= 1.5 * body) and (upper <= body * 0.3):  # 기존 2.0에서 1.5로 완화
            return 'hammer'
            
        # 샛별 패턴 - 더 유연한 조건
        if (upper >= 1.5 * body) and (lower <= body * 0.3):  # 기존 2.0에서 1.5로 완화
            return 'shooting_star'
            
        # 잉여 패턴 - 더 유연한 조건
        if previous is not None:
            if self._is_bullish(current) != self._is_bullish(previous):
                current_body = self._calculate_body(current)
                previous_body = self._calculate_body(previous)
                if current_body > previous_body * 0.8:  # 기존 1.0에서 0.8로 완화
                    return 'bullish_engulfing' if self._is_bullish(current) else 'bearish_engulfing'
        
        # 추가 패턴: 마루바닥/천장
        if body > total_length * 0.7:  # 큰 몸통
            if self._is_bullish(current) and upper <= body * 0.1:  # 상승 마루바닥
                return 'bullish_marubozu'
            elif not self._is_bullish(current) and lower <= body * 0.1:  # 하락 천장
                return 'bearish_marubozu'
        
        # 추가 패턴: 스피닝 탑
        if body <= total_length * 0.3 and upper >= total_length * 0.3 and lower >= total_length * 0.3:
            return 'spinning_top'
        
        return 'normal'

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
        다음 상태 함수 - 캔들스틱 패턴 기반 상태 데이터
        Returns:
            state: 다음 상태 (캔들스틱 패턴 특성)
        '''
        # 최근 lookback 기간의 데이터
        recent_data = self.data.iloc[self.current_step-self.lookback+1:self.current_step+1]
        
        state_vector = []
        
        for i in range(len(recent_data)):
            current = recent_data.iloc[i]
            previous = recent_data.iloc[i-1] if i > 0 else None
            
            # 기본 특성
            features = [
                self._calculate_body(current),
                self._calculate_upper_shadow(current),
                self._calculate_lower_shadow(current),
                float(self._is_bullish(current)),
                current['Volume'],
                (current['Close'] - previous['Close']) / previous['Close'] if previous is not None else 0.0
            ]
            
            # 패턴 원-핫 인코딩
            pattern = self._identify_pattern(current, previous)
            pattern_features = np.zeros(self.pattern_types)
            if pattern == 'doji':
                pattern_features[1] = 1
            elif pattern == 'hammer':
                pattern_features[2] = 1
            elif pattern == 'shooting_star':
                pattern_features[3] = 1
            elif 'engulfing' in pattern:
                pattern_features[4] = 1
            else:
                pattern_features[0] = 1
            
            # 특성 결합
            state_vector.extend(features)
            state_vector.extend(pattern_features)
        
        return np.array(state_vector, dtype=np.float32)
        
    def _should_close_position(self, current_candle, previous_candle):
        """
        캔들패턴에 따른 포지션 정리 여부 결정 - 더 유연한 조건 적용
        """
        if self.position == FLAT:
            return False, None
            
        # 손절매 조건
        if self.position == LONG:
            if (current_candle['Close'] - self.entry_price) / self.entry_price < -self.stop_loss_threshold:
                return True, '손절매'
        else:  # SHORT
            if (self.entry_price - current_candle['Close']) / self.entry_price < -self.stop_loss_threshold:
                return True, '손절매'
        
        # 트레일링 스탑
        if self.position == LONG:
            if current_candle['Close'] > self.max_price:
                self.max_price = current_candle['Close']
            elif (self.max_price - current_candle['Close']) / self.max_price > self.trailing_stop:
                return True, '트레일링 스탑'
        else:  # SHORT
            if current_candle['Close'] < self.min_price:
                self.min_price = current_candle['Close']
            elif (current_candle['Close'] - self.min_price) / self.min_price > self.trailing_stop:
                return True, '트레일링 스탑'
        
        # 캔들패턴 기반 정리 조건 - 더 유연한 조건 적용
        current_pattern = self._identify_pattern(current_candle, previous_candle)
        
        # 롱 포지션 정리 조건
        if self.position == LONG:
            # 상승 추세 반전 패턴 + 거래량 증가
            if current_pattern in ['shooting_star', 'bearish_engulfing', 'bearish_marubozu'] and current_candle['Volume'] > previous_candle['Volume'] * 0.8:  # 거래량 조건 완화
                return True, f'반전 패턴: {current_pattern} + 거래량 증가'
            # 연속 도지 또는 스피닝 탑
            elif current_pattern in ['doji', 'spinning_top'] and previous_candle is not None and self._identify_pattern(previous_candle) in ['doji', 'spinning_top']:
                return True, '연속 불확실성 패턴'
        
        # 숏 포지션 정리 조건
        elif self.position == SHORT:
            # 하락 추세 반전 패턴 + 거래량 증가
            if current_pattern in ['hammer', 'bullish_engulfing', 'bullish_marubozu'] and current_candle['Volume'] > previous_candle['Volume'] * 0.8:  # 거래량 조건 완화
                return True, f'반전 패턴: {current_pattern} + 거래량 증가'
            # 연속 도지 또는 스피닝 탑
            elif current_pattern in ['doji', 'spinning_top'] and previous_candle is not None and self._identify_pattern(previous_candle) in ['doji', 'spinning_top']:
                return True, '연속 불확실성 패턴'
        
        return False, None

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
        position = FLAT

        # 보상 구조 개선
        if self.position != FLAT:
            reward = 0.1  # 포지션 유지 시 약간의 보상
        else:
            reward = -0.1  # 중립 포지션에 대한 약간의 페널티

        if self.num > 24*7*12:
            done = True
        
        # 현재 및 이전 캔들 데이터
        current_candle = self.data.iloc[self.current_step]
        previous_candle = self.data.iloc[self.current_step-1] if self.current_step > 0 else None
        
        # 포지션 정리 여부 확인
        should_close, close_reason = self._should_close_position(current_candle, previous_candle)
        
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
        '''
        self.logger.render(f'    - action: {action}')
        self.logger.render(f'    - current_position: {self.position}')
        self.logger.render(f'    - position_direction: {position_direction}')
        self.logger.render(f' ')
        '''

        # 포지션 변경 또는 정리
        if self.position != position_direction or should_close:
            if self.position != FLAT:
                profit = self.position * (current_price - self.entry_price)
                '''
                self.logger.render(f'    - profit: {profit}, size: {self.size}')
                self.logger.render(f'    - self.position: {self.position}, position_direction: {position_direction}')
                self.logger.render(f'    - current_price: {current_price}, entry_price: {self.entry_price}')
                self.logger.render(f'    - balance: {self.balance}')
                self.logger.render(f' ')
                '''
                self.balance += profit * self.size
                self.size = 0
                self.position = FLAT
                position = position_direction

                profit_rate = profit / self.entry_price
                
                if profit > 0:
                    reward += profit_rate * 20  # 수익에 대한 보상 증가
                    self.success += 1
                    trade_success = True
                else:
                    reward += profit_rate * 15  # 손실에 대한 페널티 완화
                    self.failure += 1
                
                if profit < 0 and self.failure > 1:
                    reward -= 10  # 연속 손실에 대한 페널티 완화
            
            self.balance_profit_rate = (self.balance - self.initial_balance) / self.initial_balance 
            
            if self.balance_profit_rate < -0.02:  # 손실 허용 범위 증가
                reward -= 30  # 큰 손실에 대한 페널티 완화
            elif self.balance_profit_rate > 0.03:  # 목표 수익률 증가
                reward += 30  # 목표 달성에 대한 보상 증가

        if position_direction != FLAT and not done and not should_close:
            trade_ratio = self.max_position_ratio
            available_balance = self.balance * (1 - self.trade_fee * self.leverage)
            position_size = available_balance * self.leverage * trade_ratio / current_price
            entry_cost = position_size * current_price * self.trade_fee
            self.balance -= entry_cost
            self.position = position_direction
            self.size = position_size
            self.entry_price = current_price
            
            position = position_direction

            if self.position == LONG:
                self.max_price = current_price
            else:
                self.min_price = current_price
            
            reward += 0.5  # 포지션 진입에 대한 보상 증가
        
        self.position_history.append(self.position)
        self.balance_profit_rate_history.append(self.balance_profit_rate * 100)  
        self.profit_history.append(profit)
        self.balance_history.append(self.balance)
        self.reward_history.append(reward)

        if done:
            self.returns_history.append(self.current_step)
            self.last_step = self.current_step
            
            final_profit_rate = (self.balance - self.initial_balance) / self.initial_balance
            if final_profit_rate > 0:
                reward += 30  # 수익에 대한 보상 증가
                self.next = True
                self.next_step += 1
            elif final_profit_rate < -0.02:  # 손실 허용 범위 증가
                reward -= 30  # 손실에 대한 페널티 완화
                self.next = False
        else:
            self.current_step += 1

        info = {
            'liquidated': self.liquidated,
            'clear': self.clear,
            'balance': self.balance,
            'profit_rate': float((self.balance - self.initial_balance) * 100 / self.initial_balance),
            'trade_success': trade_success,
            'close_reason': close_reason if should_close else None,
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
        self.logger.render(f'학습 횟수: {self.next_step}')
