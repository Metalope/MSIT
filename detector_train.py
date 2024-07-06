from env import Fak
from airctrl import context
from airctrl.algorithm.a2c.learner import Learner
from airctrl.algorithm.a2c.memory import Memory
from airctrl.utils.scheduler import LinerScheduler
from airctrl.utils.normalize import normalize
import model as fak_model
from airctrl.data import DataManager
from torch.optim import Adam
from airctrl.learning_engine import NaiveEngine
import torch


# context.set_context(mode='dev')
# from memory_profiler import profile


def obs_callback(obs_raw):
    radar_angle = obs_raw["current_radar_angle"]
    radar_angle_norm = normalize(radar_angle, [-115, 15], [-1.0, 1.0])
    target_pos = obs_raw["target_pos_detect"]

    if target_pos[0] == target_pos[1] == -1:
        state_list = [radar_angle_norm] + [-1000, -1000]
    else:
        target_pos_norm_x = normalize(target_pos[0], [0, 1000], [-1.0, 1.0])
        target_pos_norm_y = normalize(target_pos[1], [0, 1000], [-1.0, 1.0])
        state_list = [radar_angle_norm] + [target_pos_norm_x, target_pos_norm_y] + [obs_raw["detect"]]

    return state_list


def action_callback(action_raw):
    leagal_action = [-1, 0, 1]
    action = {"radar_angle": leagal_action[action_raw],
              "gun_angle": 45,
              "is_fire": False}

    return action


def should_stop_callback(episode_log):
    if episode_log['minimum_episode_reward'] >= 950:
        return True
    else:
        return False


if __name__ == "__main__":

    # server = Memory.init_server(5000, samples_per_insert=3)
    # server.run()
    #
    # ENV_NUM = 24
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # agent = fak_model.RadarAgent(device)
    #
    # optim = Adam(filter(lambda x: x.requires_grad, list(agent.actor.parameters()) + list(agent.critic.parameters())),
    #              lr=1e-4)
    # learner = Learner(agent, 512, optim, entropy_coeff=LinerScheduler(0.05, 0.01, max_step=1e3),
    #                   tao=LinerScheduler(0.5, 0.05, max_step=10), priority_ratio=0.6)
    # learner.add_memory_server(f'localhost:{server.port}')
    #
    # data_manager = DataManager([learner], Fak, ENV_NUM, max_episode_length=1000, obs_callback=obs_callback,
    #                            action_callback=action_callback)
    # engine = NaiveEngine([learner], data_manager, env_class=Fak, env_num=3, saving_freq=5000,
    #                      max_episode_length=1000, should_stop_callback=should_stop_callback, obs_callback=obs_callback,
    #                      rew_callback=None, action_callback=action_callback)
    #
    # engine.start()

    # validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = fak_model.RadarAgent(device)
    env = Fak(obs_callback=obs_callback, action_callback=action_callback, rew_callback=None)
    env.render()
    for x in range(3300, 3500, 100):
        agent.load(f'./models/best')
        # agent.save_encoder('./models/')
        agent.exploit()
        for _ in range(20):
            env.run_episode(agent, 1000)
        print(f"best-Avg: {env.step_count/20}")
        env.step_count = 0
    env.close()
