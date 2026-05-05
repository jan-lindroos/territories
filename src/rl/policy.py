import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)

    return layer


class Policy(nn.Module):
    def __init__(self, window_radius):
        super().__init__()

        side = 2 * window_radius + 1
        self.backbone = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            flat_size = self.backbone(torch.zeros(1, 1, side, side)).shape[1]

        self.actor = nn.Sequential(
            layer_init(nn.Linear(flat_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 4), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(flat_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(self.backbone(x))

    def get_action_and_value(self, x, action=None):
        features = self.backbone(x)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)
