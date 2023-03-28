import torch.nn as nn
from copy import deepcopy


class LinearModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
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

    def forward(self, x, model):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        if model == "online":
            return self.online(x)
        elif model == 'target':
            return self.target(x)
        else:
            raise Exception(f'model must be in [target, online], got {model}')
