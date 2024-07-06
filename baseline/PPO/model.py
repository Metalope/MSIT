import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from airctrl.algorithm.ppo.agent import PPOAgentBase


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
        x = torch.tanh(x)
        # x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.encoder = EncoderPhase1()

    def forward(self, obs):
        obs_ori = obs[:, :2]
        obs_encode = obs[:, 2: -4]
        obs_radar = obs[:, -4:]
        obs_encoded = self.encoder(obs_encode)
        if len(obs_encoded.shape) == 1:
            obs_encoded = obs_encoded.unsqueeze(0)
        obs_cat = torch.cat((obs_ori, obs_encoded, obs_radar), dim=1)
        x = torch.relu(self.fc1(obs_cat))
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
        # x = torch.tanh(self.fc3(x))
        # x = Categorical(torch.softmax(x, dim=-1))  # for a2c
        x = torch.tanh(x)
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


class Agent(PPOAgentBase):
    def __init__(self, device, action_var=0.1):
        super().__init__(device)
        self.actor = Actor()
        self.critic = Critic()
        self.radar = RadarActor()
        self.count = 0
        self.action_var = torch.diag(torch.full((5,), action_var**2))
        self.to(device)
    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.radar.to(device)
        self.action_var = self.action_var.to(device)

    def get_action_dist(self, obs):
        self.count += 1
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        obs_fire = obs[:, :-4]
        obs_radar = obs[:, -4:]
        action_fire = self.actor(obs_fire)
        action_radar = self.radar(obs_radar)
        action_mean = torch.cat((action_fire, action_radar), dim=1)
        action_dist = MultivariateNormal(action_mean, self.action_var)
        return action_dist

    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if isinstance(obs, list):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        return self.critic(obs)

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.radar.parameters())
