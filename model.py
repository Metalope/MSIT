from abc import ABC

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from airctrl.algorithm.dac.agent import DACAgentBase
from airctrl.algorithm.a2c.agent import A2CAgentBase
from airctrl.utils.normalize import normalize


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
        # x = torch.tanh(x.clone())
        x = torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.encoder = EncoderPhase1()

    def forward(self, obs, act):
        obs_ori = obs[:, :2]
        obs_encode = obs[:, 2:]
        obs_encoded = self.encoder(obs_encode)
        if len(obs_encoded.shape) == 1:
            obs_encoded = obs_encoded.unsqueeze(0)

        # # Remove gradient
        # obs_encoded = obs_encoded.detach()

        obs_cat = torch.cat((obs_ori, obs_encoded), dim=1)
        x = torch.cat((obs_cat, act), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        return x


class Agent(DACAgentBase):
    def __init__(self, device):
        super().__init__(device)
        self.critic = Critic()
        self.actor = Actor()
        self.to(device)
        self.count = 0

    def get_action(self, obs):
        self.count += 1
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = self.actor(obs)
        if self.is_explore():
            noise = torch.rand_like(action) / 2 + 0.75
            return noise * action
        else:
            return action

    def get_value(self, obs, act):
        return self.critic(obs, act)

    def save_encoder(self, file_path):
        # assert isinstance(self.encoder, nn.Module)
        torch.save(self.actor.encoder.state_dict(), file_path + "actor_encoder.pkl")
        torch.save(self.critic.encoder.state_dict(), file_path + "critic_encoder.pkl")


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
        x = Categorical(torch.softmax(x, dim=-1))  # for a2c
        # x = torch.tanh(x)
        return x


class RadarCritic(nn.Module):
    def __init__(self):
        super(RadarCritic, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs):
        # x = torch.cat((obs, act), dim=1)
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        return x


class RadarAgent(A2CAgentBase):
    def __init__(self, device):
        super().__init__(device)
        self.critic = RadarCritic()
        self.actor = RadarActor()
        self.to(device)

    def get_action_dist(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = self.actor(obs)
        return action

    def get_value(self, obs):
        return self.critic(obs)

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())


class AgentPhase3(DACAgentBase):
    def __init__(self, device, actor_path, critic_path, actor_encoder_path, radar_path):
        super().__init__(device)
        self.actor = Actor()
        self.critic = Critic()
        self.radar = RadarActor()

        self.actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load(critic_path, map_location=torch.device('cpu')))
        self.actor.encoder.load_state_dict(torch.load(actor_encoder_path, map_location=torch.device('cpu')))
        self.critic.encoder.load_state_dict(torch.load(actor_encoder_path, map_location=torch.device('cpu')))

        self.radar.load_state_dict(torch.load(radar_path, map_location=torch.device('cpu')))

        for p in self.radar.parameters():
            p.requires_grad = False

        # # Parameters in Encoder should be trained
        # for p in self.actor.parameters():
        #     p.requires_grad = False
        # for p in self.critic.parameters():
        #     p.requires_grad = False
        # for p in self.actor.encoder.parameters():
        #     p.requires_grad = True
        # for p in self.critic.encoder.parameters():
        #     p.requires_grad = True

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
        action_radar = self.radar(obs_radar).sample()

        if self.is_explore():
            noise = torch.rand_like(action_fire) / 2 + 0.75

            return torch.cat((noise * action_fire, action_radar.unsqueeze(1)), dim=1)
        else:
            # # Fire when detected
            # for i in range(len(obs_radar[:, -1])):
            #     if not bool(obs_radar[i, -1]):
            #         action_fire[i, 1] = -1
            return torch.cat((action_fire, action_radar.unsqueeze(1)), dim=1)

    def get_value(self, obs, act):
        obs_fire = obs[:, :-4]
        act_fire = act[:, :-1]
        return self.critic(obs_fire, act_fire)

    def save_encoder(self, file_path):
        # assert isinstance(self.encoder, nn.Module)
        torch.save(self.actor.encoder.state_dict(), file_path + "actor_encoder.pkl")
        torch.save(self.critic.encoder.state_dict(), file_path + "critic_encoder.pkl")


