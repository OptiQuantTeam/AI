import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    def __init__(self, state_dim, action_dim, quantiles=32):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.quantiles = quantiles

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.quantile_layer = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, action_dim)

    def forward(self, state, tau):
        batch_size = state.shape[0]
        state_feature = self.feature_layer(state)

        tau = tau.view(batch_size, self.quantiles, 1)
        quantile_embedding = torch.cos(np.pi * tau * torch.arange(1, 129).float().to(state.device))
        quantile_embedding = self.quantile_layer(quantile_embedding).relu()

        x = state_feature.unsqueeze(1) + quantile_embedding
        x = self.value_layer(x)

        return x
    