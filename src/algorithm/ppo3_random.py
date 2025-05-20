import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import datetime
import network.actorcritic as AC
import network.indicator as ID
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class PPO3_RANDOM:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        model_name=None,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon=0.2,
        epochs=10,
        batch_size=32,
        kl_target=0.01,  # KL 발산 목표값
        kl_coef=0.5,     # KL 발산 계수
        alpha=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.actor_critic = AC.ActorCritic2(state_dim, action_dim).to(device)
        self.indicator_distribution = ID.IndicatorDistribution3(state_dim, action_dim).to(device)
        
        # 액터 옵티마이저
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.feature_extraction.parameters()},
            {'params': self.actor_critic.actor_direction.parameters()},
            {'params': self.actor_critic.actor_direction_std},
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic}
        ], lr=lr_actor)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.memory = deque()
        self.performances = deque()
        self.alpha = 0.9
        # KL 발산 관련 파라미터
        self.kl_target = kl_target
        self.kl_coef = kl_coef
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state = state[:, 5:]  # 5번 인덱스부터 마지막까지의 데이터만 사용
        state2 = state[:, [7,9,10,11,16,17]]
        self.actor_critic.eval()
        with torch.no_grad():
            value = torch.tensor([0.0])

            action_dist = torch.distributions.Categorical(torch.tensor([0.25,0.5,0.25]))
            action_idx = action_dist.sample()
            action = action_idx.float() - 1.0
            log_prob = action_dist.log_prob(action_idx)

            
        return (
            action.cpu().numpy(),
            value.cpu().numpy(),
            log_prob.cpu().numpy()
        )
        
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self, success_rate=None):
        if len(self.memory) < self.batch_size:
            return 0

        # 메모리 비우기
        self.memory.clear()
        return 1
    
    def save_model(self, path):
        model_state = {
            # 모델 기본 정보
            'model_name': self.model_name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            
            # 모델 가중치 및 옵티마이저 상태
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'indicator_distribution_state_dict': self.indicator_distribution.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # 학습 파라미터
            'learning_params': {
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epochs': self.epochs,
                'lr_actor': self.optimizer.param_groups[0]['lr'],
                'lr_critic': self.optimizer.param_groups[-1]['lr'],
                'batch_size': self.batch_size,
                'device': str(self.device)
            }
        }
        
        # 변환된 데이터 저장
        torch.save(model_state, path)
    
    def save_learning_state(self, info, path):
        learning_state = {
            # 학습 진행 상태
            'training_state': {
                'current_episode': info['training_state']['current_episode'],
                'total_episodes': info['training_state']['total_episodes'],
                'last_step': info['training_state']['last_step'],
                'checkpoint_term': info['training_state']['checkpoint_term']
            },
            
            # 학습 결과
            'training_results': {
                'rewards_history': info['training_results']['rewards_history'],
                'episode_results': info['training_results']['episode_results'],
                'completed_episodes': sum(info['training_results']['episode_results']),
                'win_rate': info['training_results']['win_rate'],
                'profit_rate_history': info['training_results']['profit_rate_history'],
                'all_balance_history': info['training_results']['all_balance_history'],
                'step_num_history': info['training_results']['step_num_history']
            },

            # 환경 정보
            'environment_info': {
                'data_path': info['environment_info']['data_path'],
                'total_data_length': info['environment_info']['total_data_length'],
                'training_period': info['environment_info']['training_period']
            },

            # 세션 정보
            'session_info': {
                'session_type': info['session_info']['session_type'],
                'session_time': info['session_info']['session_time'],
                'log_file': info['session_info']['log_file'],
                'previous_checkpoints': info['session_info']['previous_checkpoints'],
                'previous_episodes': info['session_info']['previous_episodes'],
                'current_session_episodes': info['session_info']['current_session_episodes'],
                'training_sessions': info['session_info']['training_sessions']
            }
        }

        # NumPy 배열과 숫자를 Python 기본 타입으로 변환
        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, np.number)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        learning_state = convert_to_serializable(learning_state)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(learning_state, f, indent=4, ensure_ascii=False)
    
    def store_performance(self, performance):
        self.performances.append((1,1,1,1,1,1))
    
    def plot_performance(self, path):
        self.performances.clear()