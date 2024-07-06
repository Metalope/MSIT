import numpy as np
import pandas as pd
from airctrl import context
from env import Fak
from airctrl.algorithm.dac.learner import Learner
from airctrl.algorithm.dac.memory import Memory
from airctrl.learning_engine import NaiveEngine
from airctrl.data import DataManager
from airctrl.utils.normalize import normalize
from baseline.DDPG.model import Agent as DDPG
from torch.distributions import Categorical
from torch.optim import Adam
from utils.tool import set_seed
import torch
from multiprocessing import Process, Manager, Lock

# context.set_context(mode='dev')
# from memory_profiler import profile


def obs_callback(obs_raw):
    gun_angle = obs_raw["current_gun_angle"]
    ammo_left = obs_raw["remain_ammo"]
    gun_angle_norm = normalize(gun_angle, [0, 90], [-1, 1])
    ammo_left_norm = normalize(ammo_left, [0, 400], [-1, 1])
    state_list = [gun_angle_norm, ammo_left_norm]
    past_pos = obs_raw["past_pos_detect"]

    for pos in past_pos:
        pos = normalize(pos, [0, 1000], [-1, 1])
        state_list.append(pos)

    radar_angle = obs_raw["current_radar_angle"]
    radar_angle_norm = normalize(radar_angle, [-115, 15], [-1.0, 1.0])
    target_pos = obs_raw["target_pos"]

    target_pos_norm_x = normalize(target_pos[0], [0, 1000], [-1.0, 1.0])
    target_pos_norm_y = normalize(target_pos[1], [0, 1000], [-1.0, 1.0])
    state_list += [radar_angle_norm] + [target_pos_norm_x, target_pos_norm_y] + [obs_raw["detect"]]

    return state_list


def action_callback(action_raw):
    if action_raw[1] > 1:
        action_raw[1] = 1.0
    leagal_action = [-1, 0, 1]
    radar_index = Categorical(torch.softmax(torch.FloatTensor(action_raw[-3:]), dim=-1)).sample()
    action = {"radar_angle": leagal_action[int(radar_index)],  # normalize(action_raw[0], [0, 1], [-45, 45]),
              "gun_angle": normalize(action_raw[0], [-1, 1], [0, 90]),
              "is_fire": True if 0 < action_raw[1] else False}
    # print(action['gun_angle'], action['is_fire'])
    return action


def should_stop_callback(episode_log):
    return False
    # step_count = episode_log["step_count"]
    # avg_epilen = episode_log['average_episode_length']
    # if step_count > 1e10 or avg_epilen < 300:
    #     return True
    # else:
    #     return False


def train():
    # context.set_context(mode='dev')
    set_seed(20)
    server = Memory.init_server(5000, samples_per_insert=3)
    server.run()

    ENV_NUM = 24

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = DDPG(device)

    critic_optim = Adam(filter(lambda x: x.requires_grad, list(agent.critic.parameters())), lr=1e-4)
    actor_optim = Adam(filter(lambda x: x.requires_grad, list(agent.actor.parameters())), lr=1e-4)
    learner = Learner(agent, 512, actor_optim=actor_optim, critic_optim=critic_optim, priority_ratio=0.6)
    learner.add_memory_server(f'localhost:{server.port}')
    data_manager = DataManager([learner], Fak, ENV_NUM, max_episode_length=1000, obs_callback=obs_callback,
                               action_callback=action_callback)
    engine = NaiveEngine([learner], data_manager, env_class=Fak, env_num=20, saving_freq=5000,
                         max_episode_length=1000, should_stop_callback=should_stop_callback, obs_callback=obs_callback,
                         rew_callback=None, action_callback=action_callback)
    with torch.autograd.set_detect_anomaly(True):
        engine.start()


def render_show(_freq_f, _freq_d, model_path="./models/best/"):
    device = torch.device('cpu')
    env = Fak(obs_callback=obs_callback, action_callback=action_callback, rew_callback=None,
              fire_interval=_freq_f, detect_delay=_freq_d)
    agent = DDPG(device)
    agent.load(model_path)
    env.render()
    env.run_episode(agent, 1000)
    env.close()


def worker(_freq_f, _freq_d, episode_num, agent, _analysis_dict, lock):
    env = Fak(obs_callback=obs_callback, action_callback=action_callback, rew_callback=None,
              fire_interval=_freq_f, detect_delay=_freq_d)
    for _ in range(episode_num):
        env.run_episode(agent, 1000)
        with lock:
            _analysis_dict["epi_length"] += [env.get_episode_length()]
            _analysis_dict["hit_rate"] += [env.hit_rate()]
            _analysis_dict["detect_rate"] += [env.detect_rate()]


def validation(freq_f, freq_d, model_path="./models/best/"):
    # validation
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    agent = DDPG(device)

    agent.load(model_path)
    agent.exploit()
    episode_total_num = 5000
    worker_num = 50
    episode_worker_num = int(episode_total_num / worker_num)

    with Manager() as manager:
        analysis_dict = manager.dict({"epi_length": [], "hit_rate": [], "detect_rate": []})
        processes = []
        lock = Lock()
        for i in range(worker_num):
            p = Process(target=worker, args=(freq_f, freq_d, episode_worker_num, agent, analysis_dict, lock))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        df = pd.DataFrame(dict(analysis_dict))
        df.to_csv(f'../../analysis/DDPG_f{freq_f}_d{freq_d}.csv')


if __name__ == '__main__':

    freq_fire_list = [19, 9, 4, 2, 1, 0, 0, 0, 0, 0]
    freq_detect_list = [0, 0, 0, 0, 0, 1, 2, 4, 9, 19]
    # train()
    # render_show(freq_fire, freq_detect)
    for freq_fire, freq_detect in zip(freq_fire_list, freq_detect_list):
        print(freq_fire, freq_detect)
        validation(freq_fire, freq_detect)


