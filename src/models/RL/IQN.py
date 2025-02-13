import torch
import pandas as pd
from .IQNcomponent.IQNAgent import IQNAgent
from env import SpotTradingEnv

class IQN:
    def __init__(self):
        # Training Loop
        data = pd.read_csv(f'/workspace/data/raw/BTCUSDT/BTCUSDT-1h-2021.csv', index_col=0)
        data = data[['Open','High','Low','Close']]
        self.env = SpotTradingEnv(data)
        
        self.agent = IQNAgent(self.env.observation_space.shape[0], self.env.action_space.n)

        
    def train(self):
        EPISODES = 50
        EPSILON_START = 1.0

        epsilon = EPSILON_START
        for episode in range(EPISODES):
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.memory.push(state, action, reward, next_state, done)
                self.agent.update()
                state = next_state
            self.agent.update_target_network()
            print(f"Episode {episode}, Balance: {self.env.balance}, holdfings: {self.env.holdings}")


    def get_state(self):
        '''
        # IQN 모델 저장 (신경망 가중치만 저장)
        torch.save(iqn_model.state_dict(), "iqn_model.pth")

        # IQNAgent 저장 (모델 가중치 포함하여 전체 상태 저장)
        agent_state = {
            "model_state": iqn_model.state_dict(),  # IQN 모델 가중치
            "optimizer_state": optimizer.state_dict(),  # 옵티마이저 상태 (Adam 등)
            "epsilon": agent.epsilon,  # 탐색률 (필요하다면)
            "hyperparameters": agent.hyperparameters,  # 하이퍼파라미터 (있다면)
            "replay_buffer": agent.replay_buffer,  # 경험 리플레이 버퍼 (선택적)
        }
        torch.save(agent_state, "iqn_agent.pth")
        '''
        pass