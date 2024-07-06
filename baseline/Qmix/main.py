from airctrl import context
from env import Fak
from airctrl.algorithm.qmix import Learner, Memory
from airctrl.learning_engine import NaiveEngine
from airctrl.data import DataManager
import torch
import pandas as pd
from torch.optim import RMSprop
from model import MultiAgent
from airctrl.utils.scheduler import LinerScheduler
from airctrl.utils.normalize import normalize
from utils.tool import set_seed
from multiprocessing import Process, Manager, Lock


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

    obs = [state_list[:-4], state_list[-4:]]

    return state_list, obs


def action_callback(action_raw):
    angle_action = [-1, 0, 1]
    fire_action = [[1, True], [0, True], [-1, True], [1, False], [0, False], [-1, False]]
    action = {"radar_angle": angle_action[int(action_raw[1])],  # normalize(action_raw[0], [0, 1], [-45, 45]),
              "gun_angle": angle_action[int(fire_action[action_raw[0]][0])],
              "is_fire": bool(fire_action[action_raw[0]][1]),
              "is_discrete": True}
    return action


def should_stop_callback(episode_log):
    return False
    # print(episode_log['average_episode_reward'])
    # if episode_log['average_episode_reward'] >= 30:
    #     return True
    # else:
    #     return False


def train():
    # context.set_context(mode='dev')

    server = Memory.init_server(2, samples_per_insert=32, memory_size=int(5e3))
    server.run()

    ENV_NUM = 30

    for i in range(5):
        set_seed(20*i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        epsilon = LinerScheduler(1, 0.05, max_step=40000)
        multi_agent = MultiAgent(device, epsilon)

        optim = RMSprop(filter(lambda x: x.requires_grad, multi_agent.parameters()), lr=5e-4)

        learner = Learner(multi_agent, 128, optim, discount_factor=0.99, tao=0.008, worker_num=2, max_grad_norm=10, double_q=True)
        learner.add_memory_server(f'localhost:{server.port}')

        # data_manager = DataManager('tcp://localhost:1229', None, [learner], StarCraft2, ENV_NUM, max_episode_length=120)
        data_manager = DataManager([learner], Fak, ENV_NUM, max_episode_length=1000,
                                   obs_callback=obs_callback,
                                   action_callback=action_callback)

        engine = NaiveEngine([learner], data_manager, env_class=Fak, env_num=6, saving_freq=5000, update_freq=5,
                             saving_dir=f'./qmix_models_seed_{i}',
                             max_episode_length=1000,
                             obs_callback=obs_callback,
                             action_callback=action_callback,
                             should_stop_callback=should_stop_callback)
        engine.start()


def render_show(_freq_f, _freq_d, model_path="./models/best/"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = LinerScheduler(1, 0.05, max_step=40000)
    env = Fak(obs_callback=obs_callback, action_callback=action_callback, rew_callback=None,
              fire_interval=_freq_f, detect_delay=_freq_d)
    multi_agent = MultiAgent(device, epsilon)
    multi_agent.load(model_path)
    env.render()
    env.run_episode(multi_agent, 1000)
    env.close()


def worker(_freq_f, _freq_d, episode_num, multi_agent, _analysis_dict, lock):
    env = Fak(obs_callback=obs_callback, action_callback=action_callback, rew_callback=None,
              fire_interval=_freq_f, detect_delay=_freq_d)
    for _ in range(episode_num):
        env.run_episode(multi_agent, 1000)
        with lock:
            _analysis_dict["epi_length"] += [env.get_episode_length()]
            _analysis_dict["hit_rate"] += [env.hit_rate()]
            _analysis_dict["detect_rate"] += [env.detect_rate()]


def validation(freq_f, freq_d, model_path="./models/best/"):
    # validation
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    epsilon = LinerScheduler(1, 0.05, max_step=40000)
    multi_agent = MultiAgent(device, epsilon)
    multi_agent.load(model_path)
    for agent in multi_agent.agent_list:
        agent.init_hidden_states(batch_size=1)
    multi_agent.exploit()

    episode_total_num = 5000
    worker_num = 50
    episode_worker_num = int(episode_total_num / worker_num)

    with Manager() as manager:
        analysis_dict = manager.dict({"epi_length": [], "hit_rate": [], "detect_rate": []})
        processes = []
        lock = Lock()
        for i in range(worker_num):
            p = Process(target=worker, args=(freq_f, freq_d, episode_worker_num, multi_agent, analysis_dict, lock))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        df = pd.DataFrame(dict(analysis_dict))
        df.to_csv(f'../../analysis/QMIX_f{freq_f}_d{freq_d}.csv')


if __name__ == '__main__':
    freq_fire_list = [19, 9, 4, 2, 1, 0, 0, 0, 0, 0]
    freq_detect_list = [0, 0, 0, 0, 0, 1, 2, 4, 9, 19]
    # train()
    # render_show()
    for freq_fire, freq_detect in zip(freq_fire_list, freq_detect_list):
        print(freq_fire, freq_detect)
        validation(freq_fire, freq_detect)
