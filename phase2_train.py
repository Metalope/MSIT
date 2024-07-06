# -*- coding:utf-8 -*-
# @time: 2022/11/09 21:05
# @author: Metalope
import torch.nn as nn
import torch
from model import EncoderPhase1 as Encoder
from model import RadarActor
from env import Fak
from env_new.main_game import FindKill
import torch.utils.data as Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from airctrl.utils.normalize import normalize


def load(network, file_path):
    assert isinstance(network, nn.Module)
    network.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))


def obs_callback(obs_raw):
    radar_angle = obs_raw["current_radar_angle"]
    radar_angle_norm = normalize(radar_angle, [-115, 15], [-1.0, 1.0])
    target_pos = obs_raw["target_pos"]

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


def generate_data(encoder, data_num, detector_model):
    # initial the env and forward for 9 steps.
    env = Fak()
    detector = RadarActor()
    detector.load_state_dict(torch.load(detector_model, map_location=torch.device('cpu')))
    pos = []
    obscure_pos = []

    for episode in range(data_num//1000):
        obs_raw, _ = env.reset()
        for _ in range(1000):
            pos.append([normalize(i, [0, 1000], [-1, 1]) for i in obs_raw["past_pos"]])
            obscure_pos.append([normalize(i, [0, 1000], [-1, 1]) for i in obs_raw["past_pos_detect"]])
            obs = obs_callback(obs_raw)
            action_raw = detector(torch.FloatTensor(obs))
            action = action_callback(action_raw.sample())
            obs_raw, _, _ = env.step(action)

    # Transfer to dataset
    if torch.cuda.is_available():
        samples = torch.FloatTensor(obscure_pos).cuda()
        labels = encoder(torch.FloatTensor(pos).cuda()).detach()
    else:
        samples = torch.FloatTensor(obscure_pos)
        labels = encoder(torch.FloatTensor(pos)).detach()

    np.save("./src/samples.npy", np.array(samples.cpu()))
    np.save("./src/labels.npy", np.array(labels.cpu()))

    if torch.cuda.is_available():
        samples = torch.FloatTensor(np.load("./src/samples.npy")).cuda()
        labels = torch.FloatTensor(np.load("./src/labels.npy")).cuda()
    else:
        samples = torch.FloatTensor(np.load("./src/samples.npy"))
        labels = torch.FloatTensor(np.load("./src/labels.npy"))

    return Data.TensorDataset(samples, labels)


if __name__ == "__main__":
    encoder_filepath = "./models/forphase2/actor_encoder.pkl"
    detector_filepath = "./models/detector-best/actor.pkl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ph1_encoder = Encoder()
    ph2_encoder = Encoder()
    load(ph1_encoder, encoder_filepath)

    ph1_encoder.to(device)
    ph2_encoder.to(device)

    dataset = generate_data(ph1_encoder, 3000000, detector_filepath)

    batch_size = 16384
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=False,
                                sampler=torch.utils.data.sampler.RandomSampler(dataset))

    optim = torch.optim.Adam(ph2_encoder.parameters(), lr=0.01)
    # optim = torch.optim.SGD(ph2_encoder.parameters(), lr=0.001)
    loss_f = nn.MSELoss()
    # loss_f = nn.L1Loss()
    episode_num = 3000
    pbar = tqdm(range(episode_num))

    loss_draw = []

    early_stop = 0
    for idx in pbar:
        episode_loss = 0
        for x, y in data_iter:
            pred = ph2_encoder(x)
            loss = loss_f(pred, y)
            optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(ph2_encoder.parameters(), max_norm=1)
            optim.step()
            episode_loss += float(loss.detach())

        loss_avg = episode_loss/len(data_iter)
        loss_draw.append(loss_avg)
        pbar.set_description(f"Episode {idx}")
        pbar.set_postfix(average_loss=loss_avg)
        if idx % 100 == 0:
            torch.save(ph2_encoder.state_dict(), f"./models/forphase2/EPI{idx}_trained_actor_encoder.pkl")
        if loss_avg < 0.0001:
            early_stop += 1
            if early_stop >= 5:
                torch.save(ph2_encoder.state_dict(), "./models/forphase2/best_trained_actor_encoder.pkl")
                break
        else:
            early_stop = 0

    plt.plot(np.array(loss_draw))

    plt.show()
