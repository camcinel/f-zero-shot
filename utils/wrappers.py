import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import numpy as np
from torchvision import transforms as T
import torch

ACTION_DICT = {
    'STANDARD_ACTIONS': [
        ["B"],  # accelerate
        ["LEFT"], ["RIGHT"],  # turn
        ["B", "LEFT"], ["B", "RIGHT"],  # accelerate through turn
        ["L", "LEFT"], ["R", "RIGHT"],  # drifting
        ["B", "L", "LEFT"], ["B", "R", "RIGHT"],  # accelerate through drift
        ["A"],  # boost
        ["Y"]  # brake
    ],
    'ONLY_DRIVE': [
        ["B"],  # accelerate
        ["LEFT"], ["RIGHT"],  # turn
        ["B", "LEFT"], ["B", "RIGHT"],  # accelerate through turn
        ["L", "LEFT"], ["R", "RIGHT"],  # drifting
        ["B", "L", "LEFT"], ["B", "R", "RIGHT"]  # accelerate through drift
    ]
}


class Discretizer(gym.ActionWrapper):
    def __init__(self, env, actions_key='STANDARD_ACTIONS'):
        super().__init__(env)
        actions = ACTION_DICT[actions_key]

        buttons = ['B',
                   'Y',
                   'SELECT',
                   'START',
                   'UP',
                   'DOWN',
                   'LEFT',
                   'RIGHT',
                   'A', 'X', 'L', 'R']

        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class PermuteToTensor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class Resizer(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True)]
        )
        observation = transforms(observation)
        return observation.squeeze(0)


class ColabHelperObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        if isinstance(observation, tuple):
            state, info = observation
            return state
        else:
            return observation


class ColabHelperStep(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        return obs, reward, done, info


def wrap_environment(env, shape, n_frames, actions_key='STANDARD_ACTIONS', colab=False):
    if colab:
        env = ColabHelperStep(env)
        env = ColabHelperObservation(env)
    env = Discretizer(env, actions_key=actions_key)
    env = GrayScaleObservation(env)
    env = Resizer(env, shape=shape)
    env = FrameStack(env, num_stack=n_frames)
    return env
