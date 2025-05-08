import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from network.actorcritic import RNNPolicy, SequenceExperience

class SequencePPO:
    def __init__(self, env, sequence_length=32, batch_size=8):
        self.env = env
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        
        # 정책 네트워크 초기화
        self.policy = RNNPolicy(
            input_dim=env.observation_space.shape[0],
            hidden_dim=128,
            action_dim=env.action_space.n
        )
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
    def collect_sequences(self, num_sequences):
        sequences = []
        for _ in range(num_sequences):
            sequence = SequenceExperience(self.sequence_length)
            state = self.env.reset()
            done = False
            
            while not done and len(sequence.states) < self.sequence_length:
                # 현재 정책으로 행동 선택
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    policy, value, _ = self.policy(state_tensor)
                    action_probs = F.softmax(policy, dim=-1)
                    
                    action = torch.multinomial(action_probs, 1).item()
                    log_prob = torch.log(action_probs[0, action])
                
                # 환경과 상호작용
                next_state, reward, done, _ = self.env.step(action)
                
                # 시퀀스에 추가
                sequence.add(
                    state, action, reward, next_state, done,
                    value.item(), log_prob.item()
                )
                
                state = next_state
            
            sequences.append(sequence.get_sequence())
        
        return sequences

    def train(self, sequences):
        for sequence in sequences:
            # 시퀀스 데이터 준비
            states = torch.FloatTensor(sequence['states'])
            actions = torch.LongTensor(sequence['actions'])
            rewards = torch.FloatTensor(sequence['rewards'])
            old_values = torch.FloatTensor(sequence['values'])
            old_log_probs = torch.FloatTensor(sequence['log_probs'])
            
            # GAE 계산
            advantages = self.compute_gae(rewards, old_values)
            
            # 여러 에포크 동안 학습
            for _ in range(10):  # PPO 업데이트 에포크
                # 미니배치로 나누기
                indices = np.random.permutation(len(states))
                for start_idx in range(0, len(states), self.batch_size):
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    
                    # 현재 정책으로 행동 확률 계산
                    policy, value, _ = self.policy(states[batch_indices])
                    new_log_probs = F.log_softmax(policy, dim=-1)
                    
                    print(new_log_probs)
                    print(batch_indices)
                    # PPO 목적 함수 계산
                    ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                    surr1 = ratio * advantages[batch_indices]
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[batch_indices]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 가치 함수 손실
                    value_loss = F.mse_loss(value.squeeze(), old_values[batch_indices])
                    
                    # 전체 손실
                    loss = policy_loss + 0.5 * value_loss
                    
                    # 최적화
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def compute_gae(self, rewards, values, gamma=0.99, lambda_=0.95):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)