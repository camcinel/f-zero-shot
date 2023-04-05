import retro
import os
from utils.wrappers import wrap_environment
from pathlib import Path
from agents.dqn import RacerDQN
from agents.ppo import RacerPPO
from agents.ppo2 import PPO2, PPO2MultipleStates
from utils.logger import MetricLoggerDQN
import datetime
from models.linear_model import LinearModel
from models.convnet import ConvModel
from models.convnetnew import ConvModelNew, ConvNetNew
from pyvirtualdisplay import Display
import argparse

IMPLEMENTED_MODELS = {
    'LinearModel': LinearModel,
    'ConvModel': ConvModel,
    'ConvModelNew': ConvModelNew
}

IMPLEMENTED_ALGOS = {
    'DQN': RacerDQN,
    'PPO': RacerPPO,
    'PPO2': PPO2,
    'PPO2MultipleStates': PPO2MultipleStates
}


def main(n_episodes=20, model_name='LinearModel', algo_name='PPO', allowed_actions='STANDARD_ACTIONS', colab=False):
    if colab:
        display = Display(visible=0, size=(1400, 900))
        display.start()

    try:
        model = IMPLEMENTED_MODELS[model_name]
    except KeyError:
        raise NotImplementedError(f'model_name == {model_name} is not implemented\n'
                                  f'Implemented Models:\t{list(IMPLEMENTED_MODELS.keys())}')

    try:
        algo = IMPLEMENTED_ALGOS[algo_name]
    except KeyError:
        raise NotImplementedError(f'algorithm {algo_name} is not implemented\n'
                                  f'Implemented Algorithms:\t{list(IMPLEMENTED_ALGOS.keys())}')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = f'{model_name}-{algo_name}-{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}'
    save_dir = Path("checkpoints") / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))
    env = retro.make('FZero-Snes', state='FZero.MuteCity1.Beginner.RaceStart.state', inttype=retro.data.Integrations.CUSTOM)
    env = wrap_environment(env, shape=84, n_frames=4, actions_key=allowed_actions.upper())
    state = env.reset()

    if algo_name != 'PPO2MultipleStates':
        racer = algo(env=env, state_dim=state.shape, action_dim=env.action_space.n, save_dir=save_dir, net=model)
    else:
        available_states = [
            'FZero.MuteCity1.Beginner.RaceStart.state',
            'FZero.BigBlue1.Beginner.RaceStart.state',
            'FZero.SandOcean.Beginner.RaceStart.state',
            'FZero.DeathWind.Beginner.RaceStart.state',
            'FZero.Silence.Beginner.RaceStart.state'
        ]
        racer = PPO2MultipleStates(init_env=env, state_list=available_states, state_dim=state.shape, action_dim=env.action_space.n,
                                   save_dir=save_dir, net=model, actions_key=allowed_actions.upper())
    racer.train(n_episodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F-Zero-Shot')
    parser.add_argument('--n-episodes', type=int, default=200000,
                        help='number of episodes to train for (default 20)')
    parser.add_argument('--model', type=str, default='LinearModel',
                        help='model type to train')
    parser.add_argument('--algo', type=str, default='PPO2MultipleStates',
                        help='reinforcement learning algorithm to use')
    parser.add_argument('--colab', action='store_true', help='for training on Google Colab')
    parser.add_argument('--allowed-actions', type=str, default='standard_actions',
                        help='decide which actions are allowed (default standard_actions')

    parameters = parser.parse_args()

    main(n_episodes=parameters.n_episodes, model_name=parameters.model, algo_name=parameters.algo,
         allowed_actions=parameters.allowed_actions, colab=parameters.colab)
