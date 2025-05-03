import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import datetime
import network.actorcritic as AC
import network.indicator as ID
import json

class PPO3:
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
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.actor_critic = AC.ActorCritic2(state_dim, action_dim).to(device)
        self.indicator_distribution = ID.IndicatorDistribution(state_dim, action_dim).to(device)
        
        self.optimizer = optim.SGD([
            {'params': self.actor_critic.feature_extraction.parameters()},
            {'params': self.actor_critic.actor_direction.parameters()},
            {'params': self.actor_critic.actor_direction_std},            
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic},
            {'params': self.indicator_distribution.final_network.parameters(), 'lr': lr_critic}
        ], lr=lr_actor, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.memory = deque()
        
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state = state[:, 5:]  # 5번 인덱스부터 마지막까지의 데이터만 사용
        with torch.no_grad():
            value, action_probs, action_logits = self.actor_critic(state)
            pi_I = self.indicator_distribution(state)
            alpha = 0.3
            pi = alpha * action_probs + (1 - alpha) * pi_I
            
            # Categorical 분포에서 액션 샘플링
            action_dist = torch.distributions.Categorical(pi)
            action_idx = action_dist.sample()
            action = action_idx.float() - 1.0
            log_prob = action_dist.log_prob(action_idx)
        
        return (
            action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0]
        )
        
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self, success_rate=None):
        if len(self.memory) < self.batch_size:
            return 0
            
        # 메모리에서 데이터 추출
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        log_prob_batch = []
        value_batch = []
        done_batch = []
        
        for transition in self.memory:
            state, action, reward, next_state, log_prob, value, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            log_prob_batch.append(log_prob)
            value_batch.append(value)
            done_batch.append(done)
        
        # 텐서로 변환
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        old_log_prob_batch = torch.FloatTensor(np.array(log_prob_batch)).to(self.device)
        old_value_batch = torch.FloatTensor(np.array(value_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # GAE 계산
        advantages = []
        returns = []
        gae = 0
        
        with torch.no_grad():
            next_state_batch = next_state_batch[:, 5:]
            next_value = self.actor_critic(next_state_batch)[0]  # value는 첫 번째 반환값
            next_value = next_value.squeeze()
            
            for r, v, done, next_v in zip(
                reversed(reward_batch),
                reversed(old_value_batch),
                reversed(done_batch),
                reversed(next_value)
            ):
                if done:
                    delta = r - v
                    gae = delta
                else:
                    delta = r + self.gamma * next_v - v
                    gae = delta + self.gamma * 0.95 * gae
                
                returns.insert(0, gae + v)
                advantages.insert(0, gae)
                
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트
        for _ in range(self.epochs):
            # 미니배치 생성 (섞지 않고 순차적으로)
            for start_idx in range(0, len(state_batch), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(state_batch))
                idx = range(start_idx, end_idx)
                
                if len(idx) < self.batch_size:
                    break
                
                # 현재 미니배치
                state = state_batch[idx]
                state = state[:, 5:]
                action = action_batch[idx]
                advantage = advantages[idx]
                return_ = returns[idx]
                old_log_prob = old_log_prob_batch[idx]
                
                # 현재 정책의 행동 분포
                value, action_probs, action_logits = self.actor_critic(state)
                pi_I = self.indicator_distribution(state)
                alpha = 0.3
                pi = alpha * action_probs + (1 - alpha) * pi_I
                
                # Categorical 분포에서 액션 샘플링
                action_dist = torch.distributions.Categorical(pi)
                action_idx = action_dist.sample()
                new_action = action_idx.float() - 1.0
                new_log_prob = action_dist.log_prob(action_idx)
                
                # PPO 비율 계산
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # 핵심 손실 함수들
                
                # 1. Actor Loss - PPO 클리핑 손실
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 2. Critic Loss - Huber Loss
                value = value.squeeze(-1)
                critic_loss = nn.SmoothL1Loss()(value, return_)
                
                # 3. 엔트로피 손실 (탐색을 위한)
                entropy_loss = -0.01 * action_dist.entropy().mean()

                
                # 전체 손실 함수
                loss = actor_loss + 0.5 * critic_loss + entropy_loss
                
                # 역전파 및 최적화
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
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
        # NumPy 스칼라를 Python 기본 타입으로 변환하는 함수
        def convert_numpy_scalars(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_scalars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_scalars(v) for v in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 데이터 변환
        #converted_data = convert_numpy_scalars(data)
        
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
    
    def load_model(self):
        self.actor_critic.load_state_dict(torch.load(f'models/{self.model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))
    
    def load_learning_state(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                learning_state = json.load(f)
        except UnicodeDecodeError:
            # UTF-8 디코딩 실패 시 다른 인코딩 시도
            with open(path, 'r', encoding='cp949') as f:
                learning_state = json.load(f)
        
        training_state = learning_state['training_state']
        training_results = learning_state['training_results']
        environment_info = learning_state['environment_info']
        session_info = learning_state['session_info']

        # 리스트를 numpy 배열로 변환
        def convert_to_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return np.array(obj)
            return obj

        training_results = convert_to_numpy(training_results)

        info = {
            # 학습 진행 상태
            'training_state':{
                'current_episode': training_state.get('current_episode', 0),
                'total_episodes': training_state.get('total_episodes', 0),
                'last_step': training_state.get('last_step', 0),
                'checkpoint_term': training_state.get('checkpoint_term', 0)
            },

            # 학습 결과
            'training_results':{
                'rewards_history': training_results.get('rewards_history', []),
                'episode_results': training_results.get('episode_results', []),
                'completed_episodes': training_results.get('completed_episodes', 0),
                'win_rate': training_results.get('win_rate', 0),
                'profit_rate_history': training_results.get('profit_rate_history', []),
                'all_balance_history': training_results.get('all_balance_history', []),
                'step_num_history': training_results.get('step_num_history', [])
            },

            # 환경 정보
            'environment_info':{
                'data_path': environment_info.get('data_path', ''),
                'total_data_length': environment_info.get('total_data_length', 0),
                'training_period': environment_info.get('training_period', {})
            },

            # 세션 정보
            'session_info':{
                'session_type': session_info.get('session_type', 'new'),
                'session_time': session_info.get('session_time', datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
                'log_file': session_info.get('log_file', ''),
                'previous_checkpoints': session_info.get('previous_checkpoints', []),
                'previous_episodes': session_info.get('previous_episodes', 0),
                'current_session_episodes': session_info.get('current_session_episodes', 0),
                'training_sessions': session_info.get('training_sessions', 0)
            }
        }
        return info
