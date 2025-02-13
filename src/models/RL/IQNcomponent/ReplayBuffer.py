import torch
import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.tensor(np.array(state), dtype=torch.float32),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(np.array(next_state), dtype=torch.float32),
                torch.tensor(done,dtype=torch.float32))
    
    def __len__(self):
        return len(self.buffer)