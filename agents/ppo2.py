import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from utils.logger import MetricLoggerPPO
from utils.wrappers import wrap_environment
import numpy as np
from itertools import cycle
import retro
import os
import random


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, 'custom_integrations'))


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, net):
        super().__init__()
        self.actor = net(state_dim, action_dim)
        self.critic = net(state_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.softmax(self.actor(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def act_best(self, state):
        action = torch.argmax(self.actor(state))
        return action.detach()

    def evaluate(self, state, action):
        action_probs = self.softmax(self.actor(state))
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO2:
    def __init__(self, env, state_dim, action_dim, net, save_dir, K_epochs=40):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.env = env
        self.save_dir = save_dir
        self.save_every = 1e5
        self.logger = MetricLoggerPPO(save_dir)

        self.clip = 0.2
        self.gamma = 0.99
        self.eps = 1e-10
        self.K_epochs = K_epochs
        self.max_timesteps_per_episode = 10000
        self.update_timestep = self.max_timesteps_per_episode

        lr_actor = 0.00005
        lr_critic = 0.0005

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, net).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, net).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse = nn.MSELoss()

    def train(self, max_timesteps):
        time_step = 0
        i_episode = 0
        n_saves = 0
        mean_loss = 0
        while time_step <= max_timesteps:
            state = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.max_timesteps_per_episode + 1):
                action = self.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                self.logger.log_step(reward)

                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                if time_step % self.update_timestep == 0:
                    mean_loss = self.update()

                if time_step % self.save_every == 0:
                    n_saves += 1
                    self.save(n_saves)

                if done:
                    break
            i_episode += 1
            self.logger.log_episode()
            self.logger.record(i_episode, time_step, mean_loss)

    def select_action_best(self, state):
        with torch.no_grad():
            state = state.__array__()
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            action = self.policy_old.act_best(state)

        return action.item()

    def select_action(self, state):
        with torch.no_grad():
            state = state.__array__()
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        print('Updating Policy')
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        losses = []
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.mse(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            losses.append(loss.mean().item())

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()
        return np.mean(losses)

    def save(self, number):
        save_path = (
                self.save_dir / f'racerPPO2_net_{number}.chkpt'
        )
        torch.save(self.policy_old.state_dict(), save_path)
        print(f'RacerNet saved to {save_path} at iteration {number}')

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class PPO2MultipleStates(PPO2):
    def __init__(self, init_env, state_list, state_dim, action_dim, net, save_dir, actions_key, K_epochs=40):
        super().__init__(init_env, state_dim, action_dim, net, save_dir, K_epochs=K_epochs)
        self.switch_every = 1
        self.state_list = state_list
        self.state_cycle = cycle(state_list)
        self.n_episodes_for_switch = 0

        self.shape = state_dim[1:]
        self.n_frames = state_dim[0]
        self.actions_key = actions_key

    def train(self, max_timesteps):
        time_step = 0
        i_episode = 0
        n_saves = 0
        mean_loss = 0
        while time_step <= max_timesteps:
            state = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.max_timesteps_per_episode + 1):
                action = self.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                self.logger.log_step(reward)

                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                if time_step % self.update_timestep == 0:
                    mean_loss = self.update()

                if time_step % self.save_every == 0:
                    n_saves += 1
                    self.save(n_saves)

                if done:
                    break
            i_episode += 1
            self.logger.log_episode()
            self.logger.record(i_episode, time_step, mean_loss)
            self.swap_environments()

    def swap_environments(self, random_select=True, verbose=False):
        self.env.close()
        if random_select:
            next_state = random.choice(self.state_list)
        else:
            next_state = next(self.state_cycle)
        self.env = retro.make('FZero-Snes', state=next_state, inttype=retro.data.Integrations.CUSTOM)
        self.env = wrap_environment(self.env, shape=self.shape, n_frames=self.n_frames, actions_key=self.actions_key)
        if verbose:
            print(f'Swapped States To {next_state}')
