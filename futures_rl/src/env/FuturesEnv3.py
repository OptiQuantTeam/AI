import numpy as np
import pandas as pd
import gym
from gym import spaces
from preprocess import preprocess_data

# position constant
LONG = 1
SHORT = -1
FLAT = 0

# action constant
BUY = 1
SELL = -1
HOLD = 0

class FuturesEnv3(gym.Env):
    def __init__(self, path=None):
        self.path = path
        self.initial_balance = 10000  # 한 번 거래당 사용 금액: 1만원
        self.wallet_balance = 1000000  # 지갑 초기 자금: 100만원
        self.initial_wallet = 1000000  # 초기 지갑 자금 저장
        self.actions = ['LONG', 'SHORT', 'FLAT']
        self.leverage = 2
        self.trade_fee = 0.0002
        self.load_data()
        self.returns_history = []
        self.max_steps = 1000
        self.action_space = spaces.Discrete(3)  # 0: FLAT, 1: LONG, 2: SHORT
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # 승패 기록
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.last_trade_result = None  # 'win', 'loss', or None

    def load_data(self):
        data = pd.read_csv(self.path)
        self.data = preprocess_data(data)
    
    def reset(self):
        self.current_step = np.random.randint(0, len(self.data) - self.max_steps)
        self.balance = self.initial_balance  # 거래 금액 초기화
        self.wallet_balance = self.initial_wallet  # 지갑 잔액 초기화
        self.initial_wallet = self.wallet_balance  # 초기 지갑 자금 저장
        self.position = FLAT
        self.action = HOLD
        self.size = 0
        self.entry_price = 1
        self.position_size = 0
        self.returns_history = [0]
        self.num = 0
        self.liquidated = False
        self.clear = False
        
        # 승패 기록 초기화
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.last_trade_result = None
        
        print(f"학습 시작 위치: {self.current_step} (전체 데이터 중 {self.current_step/len(self.data)*100:.2f}%)\n")
        return self._next_observation()
    
    def _next_observation(self):
        return np.array([self.data.iloc[self.current_step]['Open'], 
                        self.data.iloc[self.current_step]['Close'], 
                        self.data.iloc[self.current_step]['Volume'], 
                        self.data.iloc[self.current_step]['CHG'], 
                        self.data.iloc[self.current_step]['stocRSI'], 
                        self.data.iloc[self.current_step]['MACD']], dtype=np.float32)
    
    def _handle_trade_result(self, profit):
        profit_rate = profit / self.entry_price
        
        # 수익률 50% 도달 시 이익 실현
        if profit_rate >= 0.50:
            self.balance += profit
            self.last_trade_result = 'win'
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # 2연승 시 이익을 지갑으로 이체
            if self.consecutive_wins >= 2:
                transfer_amount = self.balance - self.initial_balance
                self.wallet_balance += transfer_amount
                self.balance = self.initial_balance
                print(f"2연승 달성! {transfer_amount:.2f}를 지갑으로 이체했습니다.")
                self.consecutive_wins = 0
            return True
            
        # 손실률 50% 도달 시 손절
        elif profit_rate <= -0.50:
            self.balance += profit
            self.last_trade_result = 'loss'
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # 2연패 시 지갑에서 자금 추가
            if self.consecutive_losses >= 2:
                if self.wallet_balance > 0:
                    transfer_amount = min(self.wallet_balance, self.initial_balance)
                    self.wallet_balance -= transfer_amount
                    self.balance = self.initial_balance
                    print(f"2연패 발생! 지갑에서 {transfer_amount:.2f}를 추가했습니다.")
                else:
                    print("지갑에 자금이 없습니다.")
                self.consecutive_losses = 0
            return True
            
        return False
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        done = False
        profit = 0
        reward = 0

        # 게임 종료 조건 체크
        if self.wallet_balance <= 0:
            done = True
            self.liquidated = True
        elif self.wallet_balance >= self.initial_wallet * 2:  # 초기 자본의 두배 도달
            done = True
            self.clear = True

        # 액션을 포지션 방향으로 변환 (0: FLAT, 1: LONG, 2: SHORT)
        if action == 1:  # LONG
            position_direction = 1
        elif action == 2:  # SHORT
            position_direction = -1
        else:  # FLAT
            position_direction = 0

        if self.position != position_direction and not done:
            # 기존 포지션 청산
            if self.position != 0:
                profit = self.position * (current_price - self.entry_price) * self.size
                self.balance += profit
                reward = profit
                
                # 수익/손실 처리
                if profit > 0:
                    self.last_trade_result = 'win'
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                    
                    # 2연승 시 이익을 지갑으로 이체
                    if self.consecutive_wins >= 2:
                        transfer_amount = self.balance - self.initial_balance
                        self.wallet_balance += transfer_amount
                        self.balance = self.initial_balance
                        self.consecutive_wins = 0
                else:
                    self.last_trade_result = 'loss'
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                    
                    # 2연패 시 지갑에서 자금 추가
                    if self.consecutive_losses >= 2:
                        if self.wallet_balance > 0:
                            transfer_amount = min(self.wallet_balance, self.initial_balance)
                            self.wallet_balance -= transfer_amount
                            self.balance = self.initial_balance
                        self.consecutive_losses = 0
            
            # 새로운 포지션 진입
            if position_direction != 0 and not done:
                available_balance = self.balance *(1- self.trade_fee * self.leverage)
                position_size = available_balance * self.leverage / current_price #개래 수수료를 거래마다 추가해야함
                entry_cost = position_size * current_price * self.trade_fee #거래 수수료에 래버리지 포함되야함
                self.balance -= entry_cost
                self.position = position_direction
                self.size = position_size
                self.entry_price = current_price
        
        if not done:
            if self.current_step >= len(self.data) - 1:
                done = True
            else:
                self.current_step += 1

        info = {
            'liquidated': self.liquidated,
            'clear': self.clear,
            'balance': self.balance,
            'wallet_balance': self.wallet_balance,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses
        }
        
        return self._next_observation(), float(reward), done, info

    def render(self):
        if self.liquidated:
            print(f"에피소드 종료 - 최종 지갑 잔액: {float(self.wallet_balance):.2f}원 (자본 소진)")
        elif self.clear:
            print(f"에피소드 종료 - 최종 지갑 잔액: {float(self.wallet_balance):.2f}원 (목표 달성)")
        else:
            print(f"에피소드 종료 - 최종 지갑 잔액: {float(self.wallet_balance):.2f}원") 