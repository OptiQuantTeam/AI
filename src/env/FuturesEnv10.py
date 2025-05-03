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

class FuturesEnv10(gym.Env):
    def __init__(self, path=None, logger=None):
        '''
        환경의 초기 설정값
        Variables:
            path: 데이터 파일 경로
            logger: 로거 객체
            initial_balance: 초기 자산
            actions: 행동 종류
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
        self.leverage = 12  # 고정 레버리지 12배
        self.trade_fee = 0.0002
        self.max_steps = 2100
        self.min_steps = 2048
        self.profit_target = 0.10  # 10% 수익 목표
        self.max_position_ratio = 0.10  # 최대 포지션 크기 비율 10%
        self.stop_loss_threshold = 0.10  # 손절매 임계값 10%
        self.trailing_stop = 0.05  # 트레일링 스탑 5%
        
        # 볼린저 밴드 관련 파라미터
        self.bb_period = 20
        self.bb_std = 2
        
        # 스토캐스틱 RSI 관련 파라미터
        self.stoch_rsi_period = 14
        self.stoch_rsi_k = 3
        self.stoch_rsi_d = 3
        
        # EMA 관련 파라미터
        self.ema_period = 200
        
        # 매매 규칙 관련 파라미터
        self.initial_stop_loss = 0.10  # 초기 손절 10%
        self.initial_take_profit = 0.14  # 초기 익절 14%
        self.breakeven_threshold = 0.10  # 10% 수익 시 손절라인 본전으로 상향
        self.breakeven_stop_loss = 0.00  # 본전 손절라인
        
        # 횡보장 관련 파라미터
        self.sideways_profit_target = 0.05  # 횡보장 수익 목표 5%
        self.sideways_stop_loss = 0.07  # 횡보장 손절 7%
        
        # 수면 매매 관련 파라미터
        self.sleep_trade_ratio = 0.05  # 수면 매매 포지션 비율 5%
        self.sleep_stop_loss = 0.05  # 수면 매매 손절 5%
        
        # 연속 손실/수익 관련 파라미터
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_consecutive_losses = 3  # 최대 연속 손실 횟수
        
        # HEIKIN ASHI 캔들 관련 파라미터
        self.prev_ha_open = None
        self.prev_ha_close = None
        self.prev_ha_high = None
        self.prev_ha_low = None
        
        # 3일 연속 상승/하락 관련 파라미터
        self.price_history_3d = []
        
        self._load_data()
        
        # 상태 공간 확장 (변동성과 다중 시간프레임 정보 추가)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,),  # 상태 공간 확장
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3, start=-1)

        self.success_episodes = 0
        self.last_step = 100

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
        # 필요한 컬럼만 선택
        columns = ['Open', 'Close', 'High', 'Low', 'Volume', 'EMA_4_slope', \
                 'EMA_12_slope', 'EMA_24_slope', 'stochRSI', 'MACD', 'MACD_Signal', \
                    'Divergence Signal', 'Trade Signal', 'Cross Signal', 'bb_width', \
                        'bb_width_change', 'price_change']
        df = df[columns]

        # 결측치 처리
        df = df.ffill()
        df = df.bfill()
                
        # 이상치 제거 (극단값 제거)
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

        return df
    
    def _calculate_heikin_ashi(self, current_price):
        '''
        HEIKIN ASHI 캔들 계산
        '''
        if self.prev_ha_close is None:
            self.prev_ha_open = current_price
            self.prev_ha_close = current_price
            self.prev_ha_high = current_price
            self.prev_ha_low = current_price
            return
        
        ha_close = (current_price + self.prev_ha_open + self.prev_ha_high + self.prev_ha_low) / 4
        ha_open = (self.prev_ha_open + self.prev_ha_close) / 2
        ha_high = max(current_price, ha_open, ha_close)
        ha_low = min(current_price, ha_open, ha_close)
        
        self.prev_ha_open = ha_open
        self.prev_ha_close = ha_close
        self.prev_ha_high = ha_high
        self.prev_ha_low = ha_low
        
        return ha_open, ha_high, ha_low, ha_close

    def _check_trading_signals(self):
        '''
        매매 신호 확인
        '''
        current_price = self.data.iloc[self.current_step]['Close']
        bb_width = self.data.iloc[self.current_step]['bb_width']
        stoch_rsi = self.data.iloc[self.current_step]['stochRSI']
        ema_200 = self.data.iloc[self.current_step]['EMA_200']
        
        # HEIKIN ASHI 캔들 계산
        ha_open, ha_high, ha_low, ha_close = self._calculate_heikin_ashi(current_price)
        
        # 3일 연속 상승/하락 확인
        self.price_history_3d.append(current_price)
        if len(self.price_history_3d) > 3:
            self.price_history_3d.pop(0)
        
        signals = {
            'long': False,
            'short': False,
            'exit': False
        }
        
        # 볼린저 밴드 + 스토캐스틱 RSI 기반 신호
        if bb_width < 0.1:  # 볼린저 밴드 수축 (횡보장)
            if stoch_rsi < 20:  # 과매도
                signals['long'] = True
            elif stoch_rsi > 80:  # 과매수
                signals['short'] = True
                
        # HEIKIN ASHI + 200EMA 기반 신호
        if current_price > ema_200:  # 200EMA 위
            if ha_close > ha_open and ha_close - ha_open > self.prev_ha_close - self.prev_ha_open:  # 양봉 크기 증가
                if stoch_rsi < 20:  # 과매도
                    signals['long'] = True
        else:  # 200EMA 아래
            if ha_close < ha_open and ha_open - ha_close > self.prev_ha_open - self.prev_ha_close:  # 음봉 크기 증가
                if stoch_rsi > 80:  # 과매수
                    signals['short'] = True
        
        # 3일 연속 상승/하락 기반 신호
        if len(self.price_history_3d) == 3:
            if all(self.price_history_3d[i] < self.price_history_3d[i+1] for i in range(2)):  # 3일 연속 상승
                signals['short'] = True
            elif all(self.price_history_3d[i] > self.price_history_3d[i+1] for i in range(2)):  # 3일 연속 하락
                signals['long'] = True
        
        return signals

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

            reward_history: 현재 에피소드의 step 별 보상 기록
            balance_profit_rate_history: 현재 에피소드의 step 별 자산 수익률 기록
            position_history: 현재 에피소드의 step 별 포지션 기록
            price_history: 현재 에피소드의 step 별 가격 기록
            balance_history: 현재 에피소드의 step 별 자산 기록
            profit_history: 현재 에피소드의 step 별 수익 기록
            profit_rate_history: 현재 에피소드의 step 별 수익률 기록
        Returns:
            state: 다음 상태
        '''

        '''
        # 일정 확률로 랜덤 시작, 그 외에는 연속적인 시작
        while True:
            if np.random.random() < 0.3:
                self.current_step = np.random.randint(36, len(self.data) - self.max_steps)
            else:
                self.current_step = self.last_step + 1
            if self.current_step < len(self.data) - self.max_steps:
                break
        '''

        self.current_step = self.last_step + 1
        if self.current_step >= len(self.data) - self.max_steps:
            self.current_step = 100
        

        self.last_step = self.current_step
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
        
        self.reward_history = []
        self.balance_profit_rate_history = [0]
        self.position_history = []  # 포지션 기록 초기화
        self.price_history = []
        self.balance_history = []
        self.profit_history = []
        self.profit_rate_history = []
        
        self.logger.render(f"학습 시작 위치: {self.current_step} (전체 데이터 중 {self.current_step/len(self.data)*100:.2f}%)")
        self.logger.render(f"현재 손실 한도: {self.stop_loss_threshold*100:.1f}%")
        
        return self._next_observation()
    
    def _next_observation(self):
        '''
        다음 상태 함수 - 기술적 지표 기반 상태 데이터
        Returns:
            state: 다음 상태 (17개 특성)
        '''
       
        # 기술적 지표만 사용 (총 17개)
        state = np.array([
            # 가격 관련 지표
            self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['Open'],
            self.data.iloc[self.current_step]['High'],
            self.data.iloc[self.current_step]['Low'],
            self.data.iloc[self.current_step]['Volume'],
            
            # 이동평균선
            self.data.iloc[self.current_step]['EMA_4_slope'],
            self.data.iloc[self.current_step]['EMA_12_slope'],
            self.data.iloc[self.current_step]['EMA_24_slope'],
            
            # RSI
            self.data.iloc[self.current_step]['stochRSI'],
            
            # MACD
            self.data.iloc[self.current_step]['MACD'],
            self.data.iloc[self.current_step]['MACD_Signal'],
            self.data.iloc[self.current_step]['Divergence Signal'],
            self.data.iloc[self.current_step]['Trade Signal'],
            self.data.iloc[self.current_step]['Cross Signal'],
            
            # 볼린저 밴드
            self.data.iloc[self.current_step]['bb_width'],
            self.data.iloc[self.current_step]['bb_width_change'],
            
            # 가격 변화율
            self.data.iloc[self.current_step]['price_change'],

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
        signals = self._check_trading_signals()
        
        # 매매 신호에 따른 포지션 조정
        if signals['long'] and self.position != LONG:
            if self.position == SHORT:
                # 숏 포지션 청산
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                exit_cost = current_price * self.trade_fee
                self.balance -= exit_cost * self.size
                self.size = 1e-6
                self.position = FLAT
            
            # 롱 포지션 진입 (10% 자본 사용)
            trade_ratio = self.max_position_ratio
            available_balance = self.balance * (1 - self.trade_fee * self.leverage)
            position_size = available_balance * self.leverage * trade_ratio / current_price
            entry_cost = position_size * current_price * self.trade_fee
            self.balance -= entry_cost
            self.position = LONG
            self.size = position_size
            self.entry_price = current_price
            self.initial_stop_loss = -0.10  # 초기 손절 -10%
            self.initial_take_profit = 0.14  # 초기 익절 +14%
            
        elif signals['short'] and self.position != SHORT:
            if self.position == LONG:
                # 롱 포지션 청산
                profit = self.position * (current_price - self.entry_price)
                self.balance += profit * self.size
                exit_cost = current_price * self.trade_fee
                self.balance -= exit_cost * self.size
                self.size = 1e-6
                self.position = FLAT
            
            # 숏 포지션 진입 (10% 자본 사용)
            trade_ratio = self.max_position_ratio
            available_balance = self.balance * (1 - self.trade_fee * self.leverage)
            position_size = available_balance * self.leverage * trade_ratio / current_price
            entry_cost = position_size * current_price * self.trade_fee
            self.balance -= entry_cost
            self.position = SHORT
            self.size = position_size
            self.entry_price = current_price
            self.initial_stop_loss = -0.10  # 초기 손절 -10%
            self.initial_take_profit = 0.14  # 초기 익절 +14%
        
        # 손절/익절 로직
        if self.position != FLAT:
            profit_rate = (current_price - self.entry_price) / self.entry_price * self.position
            
            # 횡보장 확인 (볼린저 밴드 수축)
            bb_width = self.data.iloc[self.current_step]['bb_width']
            is_sideways = bb_width < 0.1
            
            # 횡보장에서의 손절/익절
            if is_sideways:
                if profit_rate > self.sideways_profit_target:  # 5% 수익 시 익절
                    # 포지션 청산
                    profit = self.position * (current_price - self.entry_price)
                    self.balance += profit * self.size
                    exit_cost = current_price * self.trade_fee
                    self.balance -= exit_cost * self.size
                    self.size = 1e-6
                    self.position = FLAT
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                elif profit_rate < -self.sideways_stop_loss:  # -7% 손실 시 손절
                    # 포지션 청산
                    profit = self.position * (current_price - self.entry_price)
                    self.balance += profit * self.size
                    exit_cost = current_price * self.trade_fee
                    self.balance -= exit_cost * self.size
                    self.size = 1e-6
                    self.position = FLAT
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
            # 일반 시장에서의 손절/익절
            else:
                # 손절 조건
                if profit_rate < self.initial_stop_loss:
                    # 포지션 청산
                    profit = self.position * (current_price - self.entry_price)
                    self.balance += profit * self.size
                    exit_cost = current_price * self.trade_fee
                    self.balance -= exit_cost * self.size
                    self.size = 1e-6
                    self.position = FLAT
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                
                # 익절 조건
                elif profit_rate > self.initial_take_profit:
                    # 포지션 청산
                    profit = self.position * (current_price - self.entry_price)
                    self.balance += profit * self.size
                    exit_cost = current_price * self.trade_fee
                    self.balance -= exit_cost * self.size
                    self.size = 1e-6
                    self.position = FLAT
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                
                # 본전 손절라인 상향 (10% 수익 시)
                elif profit_rate > self.breakeven_threshold:
                    self.initial_stop_loss = self.breakeven_stop_loss
        
        # 보상 계산
        reward = 0
        if self.position != FLAT:
            unrealized_profit = self.position * (current_price - self.entry_price) * self.size / self.entry_price
            reward = unrealized_profit
        
        # 에피소드 종료 조건
        done = self.current_step >= len(self.data) - 1 or self.consecutive_losses >= self.max_consecutive_losses
        
        info = {
            'position': self.position,
            'profit_rate': (self.balance - self.initial_balance) / self.initial_balance,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'is_sideways': is_sideways if 'is_sideways' in locals() else False
        }
        
        if not done:
            self.current_step += 1
            
        return self._next_observation(), reward, done, info

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
