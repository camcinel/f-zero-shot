import retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, 'custom_integrations'))
    env = retro.make('FZero-Snes', state='FZero.KnightCup.Easy.state', inttype=retro.data.Integrations.ALL)
    observation = env.reset()
    for _ in range(10000):
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            observation = env.reset()
    env.close()


if __name__ == '__main__':
    main()
