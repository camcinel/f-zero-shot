import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import random


class Racer:
    def __init__(self, state_dim, action_dim, save_dir, net):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.actions_done = np.zeros(action_dim, dtype=int)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = net(self.state_dim, self.action_dim).float()
        self.net = self.net.to(self.device)

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.99
        self.curr_step = 0

        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4
        self.save_every = 5e4

        self.memory = deque(maxlen=25000)
        self.batch_size = 128

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, dim=1).item()
            self.actions_done[action_idx] += 1

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1- done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f'racer_net_{int(self.curr_step // self.save_every)}.chkpt'
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f'RacerNet saved to {save_path} at step {self.curr_step}')

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    def reset_actions(self):
        self.actions_done = np.zeros(self.action_dim, dtype=int)

    def print_actions(self):
        print(f'Non-random actions done: {self.actions_done}')
