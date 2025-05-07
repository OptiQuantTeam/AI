import torch
import torch.nn as nn
import torch.nn.functional as f

class Network(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x
    

class ActorCriticBase(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()

        self.discount_factor = 0.99

        self.actor = Network(in_features, hidden_dimensions, out_features, dropout)
        self.critic = Network(in_features, hidden_dimensions, 1, dropout)

    def forward(self, x):
        action_pred = self.actor(x)
        value_pred = self.critic(x)
        return action_pred, value_pred
    
    def calculate_returns(self, rewards):
        returns = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + cumulative_reward * self.discount_factor
            returns.insert(0, cumulative_reward)
        returns = torch.tensor(returns)
        # normalize the return
        returns = (returns - returns.mean()) / returns.std()
        return returns
    
    def calculate_advantages(self, returns, values):
        advantages = returns - values
        # Normalize the advantage
        advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages
    
    def calculate_surrogate_loss(
            self,
            actions_log_probability_old,
            actions_log_probability_new,
            epsilon,
            advantages):
        advantages = advantages.detach()
        policy_ratio = (
                actions_log_probability_new - actions_log_probability_old
                ).exp()
        surrogate_loss_1 = policy_ratio * advantages
        surrogate_loss_2 = torch.clamp(
                policy_ratio, min=1.0-epsilon, max=1.0+epsilon
                ) * advantages
        surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        return surrogate_loss
    
    def calculate_losses(surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
        entropy_bonus = entropy_coefficient * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).sum()
        value_loss = f.smooth_l1_loss(returns, value_pred).sum()
        return policy_loss, value_loss
    
    def init_training():
        states = []
        actions = []
        actions_log_probability = []
        values = []
        rewards = []
        done = False
        episode_reward = 0
        return states, actions, actions_log_probability, values, rewards, done, episode_reward