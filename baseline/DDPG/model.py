import torch
import torch.nn as nn
import numpy as np

from airctrl.algorithm.dac.agent import DACAgentBase


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 2)
        self.encoder = EncoderPhase1()

    def forward(self, obs):
        obs_ori = obs[:, :2]
        obs_encode = obs[:, 2:]

        obs_encoded = self.encoder(obs_encode)
        if len(obs_encoded.shape) == 1:
            obs_encoded = obs_encoded.unsqueeze(0)

        obs_cat = torch.cat((obs_ori, obs_encoded), dim=1)
        x = torch.relu(self.fc1(obs_cat))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # x = torch.tanh(self.fc3(x))

        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(17, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.encoder = EncoderPhase1()

    def forward(self, obs, act):
        obs_ori = obs[:, :2]
        obs_encode = obs[:, 2:]
        obs_encoded = self.encoder(obs_encode)
        if len(obs_encoded.shape) == 1:
            obs_encoded = obs_encoded.unsqueeze(0)

        obs_cat = torch.cat((obs_ori, obs_encoded), dim=1)
        x = torch.cat((obs_cat, act), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        return x


class RadarActor(nn.Module):
    def __init__(self):
        super(RadarActor, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=-1)
        # x = Categorical(torch.softmax(x, dim=-1))  # for a2c
        # x = torch.tanh(x)
        return x


class EncoderPhase1(nn.Module):
    def __init__(self):
        super(EncoderPhase1, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        x = torch.nn.functional.leaky_relu(x)
        return x


class Agent(DACAgentBase):
    def __init__(self, device):
        super().__init__(device)
        self.actor = Actor()
        self.critic = Critic()
        self.radar = RadarActor()
        self.to(device)
        self.count = 0

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.radar.to(device)

    def get_action(self, obs):
        self.count += 1
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        obs_fire = obs[:, :-4]
        obs_radar = obs[:, -4:]
        action_fire = self.actor(obs_fire)
        action_radar = self.radar(obs_radar)

        if self.is_explore():
            noise = torch.rand_like(action_fire) / 2 + 0.75
            return torch.cat((noise * action_fire, action_radar), dim=1)
        else:
            for i in range(len(obs_radar[:, -1])):
                if not bool(obs_radar[i, -1]):
                    action_fire[i, 1] = -1
            return torch.cat((action_fire, action_radar), dim=1)

    def get_value(self, obs, act):
        obs_fire = obs[:, :-4]
        act_fire = act
        return self.critic(obs_fire, act_fire)
