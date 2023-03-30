import retro
import os
from utils.wrappers import Discretizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, 'custom_integrations'))
    print(retro.data.Integrations.CUSTOM.paths)
    print(retro.data.EMU_EXTENSIONS)
    env = Discretizer(retro.make('FZero-Snes', state='FZero.KnightCup.Easy.state', inttype=retro.data.Integrations.CUSTOM))
    observation = env.reset()
    print(env.action_space.n)
    print(env.action_space)
    current_reward = 0
    for i in range(10000):
        sample = env.action_space.sample()
        observation, reward, done, info = env.step(10)
        current_reward += reward
        if done:
            print(f'Total reward earned:\t{current_reward}')
            current_reward = 0
            observation = env.reset()
    env.close()


if __name__ == '__main__':
    main()
