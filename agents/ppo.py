import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch.nn import MSELoss, Softmax
import numpy as np
from utils.logger import MetricLoggerPPO


class RacerPPO:
    def __init__(self, env, state_dim, action_dim, save_dir, net):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.timesteps_per_batch = 5_000
        self.max_timesteps_per_episodes = 2_500
        self.n_updates_per_iteration = 5
        self.save_every = 10
        self.logger = MetricLoggerPPO(save_dir)

        self.eps = 1e-10
        self.clip = 0.2
        self.gamma = 0.99

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.actor = net(state_dim, action_dim)
        self.actor = self.actor.to(self.device)
        self.critic = net(state_dim, 1)
        self.critic = self.critic.to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.0025)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.0025)
        self.mse = MSELoss()
        self.softmax = Softmax(dim=1)

    def train(self, n_iterations):
        steps = 0
        for iteration in range(n_iterations):
            actor_losses = []
            batch_observations, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lengths = self.rollout()
            steps += np.sum(batch_lengths)

            V, _ = self.evaluate(batch_observations, batch_actions)
            A_k = batch_rewards_to_go - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + self.eps)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_observations, batch_actions)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = self.mse(V, batch_rewards_to_go)
                print(critic_loss)
                actor_losses.append(actor_loss.detach().item())

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            if iteration > 0 and iteration % self.save_every == 0:
                self.save(iteration)
            self.logger.record(iteration, steps, np.mean(actor_losses))

    def rollout(self):
        batch_observations = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_lengths = []

        t = 0
        while t < self.timesteps_per_batch:
            episode_rewards = []
            observation = self.env.reset()

            for episode_t in range(self.max_timesteps_per_episodes):
                t += 1
                observation = observation[0].__array__() if isinstance(observation, tuple) else observation.__array__()
                observation = torch.tensor(observation, device=self.device)
                batch_observations.append(observation)
                action, log_prob = self.get_action(observation.unsqueeze(0))
                observation, reward, done, _, _ = self.env.step(action)

                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                self.logger.log_step(reward)

                if done:
                    self.logger.log_episode()
                    break

            batch_lengths.append(episode_t+1)
            batch_rewards.append(episode_rewards)

        batch_observations = torch.stack(batch_observations)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float, device=self.device)
        batch_log_probs = torch.stack(batch_log_probs)
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        return batch_observations, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lengths

    def get_action(self, observation):
        output = self.actor(observation)
        probs = self.softmax(output).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().item(), log_prob.detach()

    def act(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        output = self.actor(state)
        probs = self.softmax(output).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().item()

    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []
        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rewards_to_go.insert(0, discounted_reward)
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float, device=self.device)
        return batch_rewards_to_go

    def evaluate(self, batch_observations, batch_actions):
        V = self.critic(batch_observations).squeeze()

        probs = self.softmax(self.actor(batch_observations))
        dist = Categorical(probs)
        log_probs = dist.log_prob(batch_actions)
        return V, log_probs

    def save(self, iteration):
        save_path = (
                self.save_dir / f'racerDQN_net_{int(iteration // self.save_every)}.chkpt'
        )
        torch.save(
            dict(actor=self.actor.state_dict(), critic=self.critic.state_dict()),
            save_path,
        )
        print(f'RacerNet saved to {save_path} at iteration {iteration}')
