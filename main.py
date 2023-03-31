import retro
import os
from utils.wrappers import wrap_environment
from pathlib import Path
from agent import Racer
from utils.logger import MetricLogger
import datetime
from models.linear_model import LinearModel
from models.convnet import ConvModel
from models.convnetnew import ConvModelNew
from pyvirtualdisplay import Display
import argparse

IMPLEMENTED_MODELS = {
    'LinearModel': LinearModel,
    'ConvModel': ConvModel,
    'ConvModelNew': ConvModelNew
}


def main(n_episodes=20, model_name='LinearModel', allowed_actions='STANDARD_ACTIONS', render=False, colab=False):
    if colab:
        display = Display(visible=0, size=(1400, 900))
        display.start()

    try:
        model = IMPLEMENTED_MODELS[model_name]
    except KeyError:
        raise NotImplementedError(f'model_name == {model_name} is not implemented\n'
                                  f'Implemented Models:\t{list(IMPLEMENTED_MODELS.keys())}')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = f'{model_name}-{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}'
    save_dir = Path("checkpoints") / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))
    env = retro.make('FZero-Snes', state='FZero.MuteCity1.Beginner.RaceStart.state', inttype=retro.data.Integrations.CUSTOM)
    env = wrap_environment(env, shape=128, n_frames=4, actions_key=allowed_actions.upper(), colab=colab)
    state = env.reset()

    racer = Racer(state_dim=state.shape, action_dim=env.action_space.n, save_dir=save_dir, net=model)

    logger = MetricLogger(save_dir)
    for episode in range(n_episodes):
        state = env.reset()
        step = 0
        while True:
            step += 1
            action = racer.act(state)

            next_state, reward, done, info = env.step(action)

            if render:
                env.render()

            racer.cache(state, next_state, action, reward, done)

            q, loss = racer.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if done:
                break

        logger.log_episode()

        if episode % 1 == 0:
            logger.record(episode=episode, epsilon=racer.exploration_rate, step=racer.curr_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F-Zero-Shot')
    parser.add_argument('--n-episodes', type=int, default=20,
                        help='number of episodes to train for (default 20)')
    parser.add_argument('--render', action='store_true', help='render the output while training')
    parser.add_argument('--model', type=str, default='LinearModel',
                        help='model type to train')
    parser.add_argument('--colab', action='store_true', help='for training on Google Colab')
    parser.add_argument('--allowed-actions', type=str, default='standard_actions',
                        help='decide which actions are allowed (default standard_actions')

    parameters = parser.parse_args()

    main(n_episodes=parameters.n_episodes, render=parameters.render, model_name=parameters.model,
         allowed_actions=parameters.allowed_actions, colab=parameters.colab)
