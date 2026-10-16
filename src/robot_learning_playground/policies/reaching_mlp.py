import torch
from torch import nn


class ReachingMLP(nn.Module):
    def __init__(self, obs_dim=20, action_dim=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)