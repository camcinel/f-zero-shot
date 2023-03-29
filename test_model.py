from models.linear_model import LinearModel
from models.convnet import ConvModel
import retro
from utils.wrappers import wrap_environment
import os
import datetime
from pathlib import Path
from agent import Racer
import torch

IMPLEMENTED_MODELS = {
    'LinearModel': LinearModel,
    'ConvModel': ConvModel
}


def test_model(model_name, saved_model_dict, n_episodes):
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
    env = retro.make('FZero-Snes', state='FZero.KnightCup.Easy.state', inttype=retro.data.Integrations.CUSTOM)
    env = wrap_environment(env, shape=64, n_frames=4)
    state = env.reset()

    racer = Racer(state_dim=state.shape, action_dim=env.action_space.n, save_dir=save_dir, net=model)

    loaded_dict = torch.load(saved_model_dict, map_location=torch.device('cpu'))
    racer.net.load_state_dict(loaded_dict['model'])
    racer.exploration_rate = 0
    racer.exploration_rate_min = 0

    with torch.no_grad():
        for episode in range(n_episodes):
            state = env.reset()
            step = 0
            while True:
                step += 1
                action = racer.act(state)

                next_state, reward, done, info = env.step(action)

                env.render()

                state = next_state

                if done:
                    break


if __name__ == '__main__':
    test_model('ConvModel', 'trained_models/ConvModel-13hours-3-28/racer_net_2.chkpt', 100)
