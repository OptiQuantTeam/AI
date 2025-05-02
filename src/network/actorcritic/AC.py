import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()

        self.state_dim = state_dim
        self.action_size = action_dim
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)  # 학습률 설정
        self.entropy_coef = 0.01

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.forward(state)
        action = np.random.normal(mu, std)
        action = np.clip(action, -1, 1)
        log_policy = self.log_pdf(mu, std, action)
        return log_policy, action
    
    def log_pdf(self, mu, std, action):
        std = torch.clamp(std, min=1e-4)
        var = std ** 2
        log_policy = -torch.log(std) - 0.5 * torch.log(2 * np.pi) - 0.5 * (action - mu) ** 2 / var
        return log_policy
    
    def train(self, old_policy, states, actions, gaes):
        # 액션을 원-핫 인코딩으로 변환
        actions = torch.nn.functional.one_hot(actions, self.action_size)
        actions = actions.reshape(-1, self.action_size)
        actions = actions.float()  # float 타입으로 변환

        # 그래디언트 계산
        self.optimizer.zero_grad()  # 그래디언트 초기화
        
        # 현재 정책으로 예측
        curr_P = self.forward(states)
        
        # 손실 계산
        loss = self.compute_loss(old_policy, curr_P, actions, gaes)
        
        # 역전파 및 옵티마이저 스텝
        loss.backward()
        self.optimizer.step()
        
        return loss.item()  # 손실값 반환

    def compute_loss(self, old_policy, curr_P, actions, gaes):
        gaes = gaes.detach()

        old_log_p = torch.log(torch.sum(old_policy * actions, dim=1)).detach()
        log_p = torch.log(torch.sum(curr_P * actions, dim=1))

        ratio = torch.exp(log_p - old_log_p)

        clip_ratio = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        surrogate = -torch.min(ratio * gaes, clipped_ratio * gaes)
        actor_loss = torch.mean(surrogate)

        return actor_loss
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)  # 학습률 설정

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, states, td_targets):
        # 그래디언트 초기화
        self.optimizer.zero_grad()
        
        # 현재 상태의 가치 예측
        v_pred = self.forward(states)
        
        # 손실 계산
        loss = self.compute_loss(v_pred, td_targets)
        
        # 역전파
        loss.backward()
        
        # 옵티마이저 스텝
        self.optimizer.step()
        
        return loss.item()  # 손실값 반환

    def compute_loss(self, v_pred, td_targets):
        mse = nn.MSELoss()
        return mse(v_pred, td_targets)