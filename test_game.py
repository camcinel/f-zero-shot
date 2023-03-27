import retro
import os
import matplotlib.pyplot as plt
from PIL import Image as im


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, 'custom_integrations'))
    print(retro.data.get_file_path('FZero-Snes', 'rom.sha', inttype=retro.data.Integrations.CUSTOM))
    env = retro.make('FZero-Snes', state='FZero.KnightCup.Easy.state', inttype=retro.data.Integrations.CUSTOM)
    observation = env.reset()
    current_reward = 0
    for i in range(10000):
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()
        current_reward += reward
        if done:
            print(f'Total reward earned:\t{current_reward}')
            current_reward = 0
            observation = env.reset()
    env.close()


if __name__ == '__main__':
    main()
