import torch.nn as nn
from copy import deepcopy
import numpy as np
from numpy import ndarray


class LinearModel(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        n_observations = np.prod(state_shape)
        self.online = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.target = deepcopy(self.online)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x, model='online'):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        if model == "online":
            return self.online(x)
        elif model == 'target':
            return self.target(x)
        else:
            raise Exception(f'model must be in [target, online], got {model}')

    def name(self):
        return 'LinearModel'
