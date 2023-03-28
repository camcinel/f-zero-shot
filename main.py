import retro
import os
from wrappers import Discretizer, Resizer, GrayScaleObservation
from gym.wrappers import FrameStack
from pathlib import Path
from agent import Racer
from logger import MetricLogger
import datetime
from linear_model import LinearModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPISODES = 1000

retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, 'custom_integrations'))
env = Discretizer(retro.make('FZero-Snes', state='FZero.KnightCup.Easy.state', inttype=retro.data.Integrations.CUSTOM))
env = GrayScaleObservation(env)
env = Resizer(env, shape=64)
env = FrameStack(env, num_stack=4)

racer = Racer(state_dim = 4 * 64 * 64, action_dim=env.action_space.n, save_dir=SAVE_DIR, net=LinearModel)

logger = MetricLogger(SAVE_DIR)


def main():
    for episode in range(NUM_EPISODES):
        state = env.reset()

        while True:
            action = racer.act(state)

            next_state, reward, done, info = env.step(action)
            env.render()

            racer.cache(state, next_state, action, reward, done)

            q, loss = racer.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if done:
                break

        logger.log_episode()

        if episode % 20 == 0:
            logger.record(episode=episode, epsilon=racer.exploration_rate, step=racer.curr_step)


if __name__ == '__main__':
    main()