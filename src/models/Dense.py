import torch
import torch.nn as nn

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

        # 활성화 함수 설정
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif self.activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif self.activation == 'tanh':
            self.activation_fn = nn.Tanh()
        else:
            self.activation_fn = None

    def forward(self, x):
        x = self.linear(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x