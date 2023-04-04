from models.linear_model import LinearModel
from models.convnet import ConvModel
from models.convnetnew import ConvModelNew
import retro
from utils.wrappers import wrap_environment
import os
import datetime
from pathlib import Path
from agents.dqn import RacerDQN
from agents.ppo import RacerPPO
from agents.ppo2 import PPO2
import torch

IMPLEMENTED_MODELS = {
    'LinearModel': LinearModel,
    'ConvModel': ConvModel,
    'ConvModelNew': ConvModelNew
}


def test_model(model_name, saved_model_dict, n_episodes):
    # TODO make this function work for any model
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
    env = wrap_environment(env, shape=84, n_frames=4, actions_key='ONLY_DRIVE')
    state = env.reset()

    racer = PPO2(env=env, state_dim=state.shape, action_dim=env.action_space.n, save_dir=save_dir, net=model)

    racer.load(saved_model_dict)
    best_reward = 0
    for i in range(n_episodes):
        with torch.no_grad():
            env.record_movie(f'output_{i}.bk2')
            total_reward = 0
            state = env.reset()
            step = 0
            while True:
                step += 1
                action = racer.select_action(state)

                next_state, reward, done, trunc, info = env.step(action)
                total_reward += reward

                state = next_state

                if done:
                    if total_reward > best_reward:
                        best_run = i
                        best_reward = total_reward
                    print(f'Total length is {step}')
                    print(f'Total reward is {total_reward}')
                    env.stop_record()
                    break
    print(f'Best run is output_{best_run}.bk2')


if __name__ == '__main__':
    test_model('ConvModelNew', 'trained_models/slower_ppo2/racerPPO2_net_1.chkpt', 10)
