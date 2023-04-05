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
from agents.ppo2 import PPO2, PPO2MultipleStates
import torch
import time
import numpy as np

IMPLEMENTED_MODELS = {
    'LinearModel': LinearModel,
    'ConvModel': ConvModel,
    'ConvModelNew': ConvModelNew
}


def test_model(model_name, saved_model_dict, n_episodes):
    # TODO make this function work for any model
    print(f'Testing model at {saved_model_dict} for {n_episodes} episodes')
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
    env = retro.make('FZero-Snes', state='FZero.Silence.Beginner.RaceStart.state', inttype=retro.data.Integrations.CUSTOM)
    env = wrap_environment(env, shape=84, n_frames=4, actions_key='ONLY_DRIVE')
    state = env.reset()

    available_states = [
        'FZero.MuteCity1.Beginner.RaceStart.state',
        'FZero.BigBlue1.Beginner.RaceStart.state',
        'FZero.SandOcean.Beginner.RaceStart.state',
        'FZero.DeathWind.Beginner.RaceStart.state',
        'FZero.Silence.Beginner.RaceStart.state',
        'FZero.MuteCity2.Beginner.RaceStart.state'
    ]
    racer = PPO2MultipleStates(init_env=env, state_list=available_states, state_dim=state.shape, action_dim=env.action_space.n, actions_key='ONLY_DRIVE', save_dir=save_dir, net=model)
    env.close()
    racer.load(saved_model_dict)
    best_reward = 0
    for state in available_states:
        state_name = state.split('.')[1]
        print(f'Running state {state_name}')
        with torch.no_grad():
            env = retro.make('FZero-Snes', state=state, inttype=retro.data.Integrations.CUSTOM)
            env = wrap_environment(env, shape=84, n_frames=4, actions_key='ONLY_DRIVE')
            env.record_movie(f'output_{state_name}.bk2')
            total_reward = 0
            state = env.reset()
            step = 0
            start_time = time.time()
            while True:
                step += 1
                action = racer.select_action_best(state)

                next_state, reward, done, trunc, info = env.step(action)
                total_reward += reward

                state = next_state

                if done:
                    if total_reward > best_reward:
                        best_run = state_name
                        best_reward = total_reward
                    end_time = time.time()
                    total_time = np.round(end_time - start_time, 3)
                    print(f'Took a total of {total_time} seconds')
                    print(f'Total length is {step}')
                    print(f'Total reward is {total_reward}')
                    env.stop_record()
                    env.close()
                    break

                if step > 25000:
                    if total_reward > best_reward:
                        best_run = state_name
                        best_reward = total_reward
                    end_time = time.time()
                    total_time = np.round(end_time - start_time, 3)
                    print(f'Took a total of {total_time} seconds')
                    print('Hard stop')
                    print(f'Total length is {step}')
                    print(f'Total reward is {total_reward}')
                    env.stop_record()
                    env.close()
                    break
    print(f'Best run is output_{best_run}.bk2')


if __name__ == '__main__':
    test_model('ConvModelNew', 'trained_models/multistate_ppo2_onlyspeed/racerPPO2_net_2.chkpt', 6)
