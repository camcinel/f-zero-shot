import retro
import os
from utils.wrappers import Discretizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, 'custom_integrations'))
    print(retro.data.Integrations.CUSTOM.paths)
    print(retro.data.EMU_EXTENSIONS)
    env = Discretizer(retro.make('FZero-Snes', state='FZero.MuteCity1.Beginner.RaceStart.state', inttype=retro.data.Integrations.CUSTOM), actions_key='ONLY_DRIVE')
    print(env.action_space.n)
    print(env.action_space)
    for j in range(9):
        observation = env.reset()
        i = 0
        current_reward = 0
        while True:
            sample = j
            observation, reward, done, info = env.step(sample)
            env.render()
            current_reward += reward
            if info['health'] == 176:
                i += 1
            if done:
                print(info['lives'])
                print(f'number of zero health checks {i}')
                print(f'Total reward earned:\t{current_reward}')
                break
    env.close()


if __name__ == '__main__':
    main()
