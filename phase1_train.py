import numpy as np
from airctrl import context
from env import Fak
from model import EncoderPhase1
from airctrl.algorithm.dac.learner import Learner
from airctrl.algorithm.dac.memory import Memory
from airctrl.learning_engine import NaiveEngine
from airctrl.data import DataManager
from airctrl.utils.scheduler import LinerScheduler
from airctrl.utils.normalize import normalize
import model as fak_model
from torch.optim import Adam
from utils.tool import set_seed
import torch

# context.set_context(mode='dev')
# from memory_profiler import profile


def obs_callback(obs_raw):
    gun_angle = obs_raw["current_gun_angle"]
    ammo_left = obs_raw["remain_ammo"]
    gun_angle_norm = normalize(gun_angle, [0, 90], [-1, 1])
    ammo_left_norm = normalize(ammo_left, [0, 100], [-1, 1])
    state_list = [gun_angle_norm, ammo_left_norm]
    positions = obs_raw["past_pos"]

    for pos in positions:
        pos = normalize(pos, [0, 1000], [-1, 1])
        state_list.append(pos)
    return state_list


def action_callback(action_raw):
    if action_raw[1] > 1:
        action_raw[1] = 1.0
    action = {"radar_angle": 45,  # normalize(action_raw[0], [0, 1], [-45, 45]),
              "gun_angle": normalize(action_raw[0], [-1, 1], [0, 90]),
              "is_fire": True if 0 < action_raw[1] else False
              }
    return action


def should_stop_callback(episode_log):
    # step_count = episode_log["step_count"]
    # avg_reward = episode_log['average_episode_reward']
    # avg_epilen = episode_log['average_episode_length']
    # if step_count > 1e10 or avg_epilen < 50:
    #     return True
    # else:
    #     return False
    return False


if __name__ == "__main__":

    # context.set_context(mode='dev')
    server = Memory.init_server(5000, samples_per_insert=3)
    server.run()
    ENV_NUM = 32
    for i in range(30):
        set_seed(25 * i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        agent = fak_model.Agent(device)

        critic_optim = Adam(filter(lambda x: x.requires_grad, list(agent.critic.parameters())), lr=1e-4)
        actor_optim = Adam(filter(lambda x: x.requires_grad, list(agent.actor.parameters())), lr=1e-4)

        learner = Learner(agent, 512, actor_optim=actor_optim, critic_optim=critic_optim, priority_ratio=0.6)
        learner.add_memory_server(f'localhost:{server.port}')

        data_manager = DataManager([learner], Fak, ENV_NUM, max_episode_length=1000, obs_callback=obs_callback,
                                   action_callback=action_callback)
        engine = NaiveEngine([learner], data_manager, env_class=Fak, env_num=10, saving_freq=5000,
                             saving_dir=f'./models/phase1_{i}/',
                             max_episode_length=1000, should_stop_callback=should_stop_callback,
                             obs_callback=obs_callback,
                             rew_callback=None, action_callback=action_callback)

        engine.start()

    # # validation
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # agent = fak_model.Agent(device)
    # env = Fak(obs_callback=obs_callback, action_callback=action_callback, rew_callback=None)
    # env.render()
    # for x in range(25300, 25400, 100):
    #     agent.load(f'./models/best-phase1-3/')
    #     agent.save_encoder('./models/forphase2/')
    #     agent.exploit()
    #     for _ in range(100):
    #         env.run_episode(agent, 1000)
    #     print(f"best-Avg: {env.step_count / 100}")
    #     env.step_count = 0
    # env.close()

 

