import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .Network import Network
from .ReplayBuffer import ReplayBuffer
import random

# Hyperparameters
GAMMA = 0.99
LR = 5e-4
BUFFER_SIZE = 100000
BATCH_SIZE = 32
N_QUANTILES = 32
TAU = 0.05  # Soft update factor


class IQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = Network(state_dim, action_dim).to(self.device)
        self.q_target = Network(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.action_dim = action_dim

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim-1)
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        taus = torch.rand(1, N_QUANTILES, device=self.device)
        
        with torch.no_grad():
            q_values = self.q_net(state, taus).mean(dim=1)

        return q_values.argmax().item()
    
    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        state, action, reward, next_state, done = (state.to(self.device), action.to(self.device), 
                                                   reward.to(self.device), next_state.to(self.device), 
                                                   done.to(self.device))
        
        taus = torch.rand(BATCH_SIZE, N_QUANTILES, device=self.device)
        next_taus = torch.rand(BATCH_SIZE, N_QUANTILES, device=self.device)

        q_values = self.q_net(state, taus).gather(2, action.view(BATCH_SIZE, 1, 1).expand(-1, N_QUANTILES, -1))

        with torch.no_grad():
            next_q_values = self.q_target(next_state, next_taus)
            best_actions = next_q_values.mean(dim=1).argmax(dim=1, keepdim=True)
            target_q_values = next_q_values.gather(2, best_actions.unsqueeze(1).expand(-1, N_QUANTILES, -1))
            targets = reward.unsqueeze(1) + GAMMA * (1-done.unsqueeze(1)) * target_q_values
        
        loss = (targets - q_values).abs().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        