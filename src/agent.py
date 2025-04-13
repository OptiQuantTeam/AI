import env
import ac
import numpy as np
import torch

class Agent():
    def __init__(self, env_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env.FuturesEnv3(path=env_path)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # 모델을 device로 이동
        self.actor = ac.Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = ac.Critic(self.state_dim).to(self.device)

        self.max_episode_num = 500
        self.actor_lr = 0.0005
        self.critic_lr = 0.001
        self.gamma = 0.99
        self.update_interval = 50
        self.epochs = 10
        self.gae_lambda = 0.95

    def train(self):
        episode = 0
        while self.max_episode_num > episode:
            episode_reward = 0
            done = False
            state = self.env.reset()

            states = []
            actions = []
            rewards = []
            old_policys = []

            while not done:
                # 상태를 텐서로 변환하고 device로 이동
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                probs = self.actor(state_tensor)
                
                # CPU로 이동 후 numpy 변환
                probs_np = probs.detach().cpu().numpy()
                print(probs_np)
                action = np.random.choice(range(self.action_dim), p=probs_np[0])

                next_state, reward, done, info = self.env.step(action)

                # numpy 배열로 저장
                states.append(state)
                actions.append(action)
                rewards.append(reward * 0.01)
                old_policys.append(probs_np[0])

                state = next_state
                episode_reward += reward

            if len(states) >= self.update_interval or done:
                # 모든 배치 데이터를 동일한 device로 이동
                states_batch = torch.FloatTensor(self.list_to_batch(states)).to(self.device)
                actions_batch = torch.LongTensor(self.list_to_batch(actions)).to(self.device)
                rewards_batch = torch.FloatTensor(self.list_to_batch(rewards)).to(self.device)
                old_policys_batch = torch.FloatTensor(self.list_to_batch(old_policys)).to(self.device)
                next_state_batch = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

                # Critic 예측 (이미 device에 있음)
                with torch.no_grad():
                    curr_Qs = self.critic(states_batch)
                    next_Qs = self.critic(next_state_batch)

                # CPU로 이동 후 GAE 계산
                gaes, td_targets = self.gae_target(
                    rewards_batch.cpu().numpy(),
                    curr_Qs.cpu().numpy(),
                    next_Qs.cpu().numpy(),
                    done
                )
                
                # 다시 device로 이동
                gaes = torch.FloatTensor(gaes).to(self.device)
                td_targets = torch.FloatTensor(td_targets).to(self.device)

                # 학습 (모든 텐서가 동일한 device에 있음)
                for epoch in range(self.epochs):
                    actor_loss = self.actor.train(old_policys_batch, states_batch, actions_batch, gaes)
                    critic_loss = self.critic.train(states_batch, td_targets)

                states = []
                actions = []
                rewards = []
                old_policys = []

            print('EP{} EpisodeReward={}'.format(episode+1, episode_reward))
            episode += 1

    def gae_target(self, rewards, curr_Qs, next_Qs, done):
        td_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        R_to_go = 0

        if not done:
            R_to_go = next_Qs[0]

        for k in reversed(range(len(rewards))):
            delta = rewards[k] + self.gamma * R_to_go - curr_Qs[k]
            gae_cumulative = self.gamma * self.gae_lambda * gae_cumulative + delta
            gae[k] = gae_cumulative
            R_to_go = curr_Qs[k]
            td_targets[k] = gae_cumulative + curr_Qs[k]

        return gae, td_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
if __name__ == "__main__":
    agent = Agent('data/preprocess/BTCUSDT/BTCUSDT-5m.csv')
    agent.train()