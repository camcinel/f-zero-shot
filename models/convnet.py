import torch.nn as nn
from copy import deepcopy


class ConvNet(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        c, h, w = state_shape
        in_dim = (h // 4) * (w // 4) * 64

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.lin1 = nn.Linear(in_features=in_dim, out_features=512)
        self.lin2 = nn.Linear(in_features=512, out_features=n_actions)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)

        x = x.view(batch_size, -1)
        x = self.relu(self.lin1(x))

        return self.lin2(x)


class ConvModel(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.online = ConvNet(state_shape, n_actions)
        self.target = deepcopy(self.online)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == 'target':
            return self.target(x)
        else:
            raise Exception(f'model must be in [target, online], got {model}')

    def __str__(self):
        return 'ConvModel'
