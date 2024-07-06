import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from airctrl.algorithm.qmix import SubAgentBase, QMIXAgentBase
from torch.distributions import Categorical
from airctrl.utils.nn_utils import update_network


class RNNQ_net(nn.Module):

    def __init__(self, input_shape, n_actions, rnn_hidden_dim=64):
        super(RNNQ_net, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(
            input_size=rnn_hidden_dim, hidden_size=rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        hidden_state = torch.zeros([1, self.rnn_hidden_dim], dtype=torch.float32)
        return hidden_state

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        h = h.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h)
        x = self.fc2(h)
        return x, h


class QMixNet(nn.Module):
    """
    input: n agents' agent_qs (a scalar for each agent), state
    output: a scalar (Q)
    """

    def __init__(self,
                 n_agents,
                 state_shape,
                 mixing_embed_dim=32,
                 hypernet_layers=2,
                 hypernet_embed_dim=64):
        super(QMixNet, self).__init__()

        self.n_agents = n_agents
        self.state_shape = state_shape
        self.embed_dim = mixing_embed_dim
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_shape,
                                       self.embed_dim * self.n_agents)
            self.hyper_w_2 = nn.Linear(self.state_shape, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_shape, hypernet_embed_dim), nn.ReLU(),
                nn.Linear(hypernet_embed_dim, self.embed_dim * self.n_agents))
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_shape, hypernet_embed_dim), nn.ReLU(),
                nn.Linear(hypernet_embed_dim, self.embed_dim))
        else:
            raise ValueError('hypernet_layers should be "1" or "2"!')

        self.hyper_b_1 = nn.Linear(self.state_shape, self.embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.state_shape, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """
        Args:
            agent_qs: (batch_size, T, n_agents)
            states:   (batch_size, T, state_shape)
        Returns:
            q_total:  (batch_size, T, 1)
        """
        batch_size = agent_qs.size(0)
        states = states.reshape(-1, self.state_shape)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        h = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w_2(states)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        q_total = (torch.bmm(h, w2) + b2).view(batch_size, -1, 1)

        return q_total


class SubAgent(SubAgentBase):
    def __init__(self, device, agent_name, q_net, epsilon):
        super().__init__(q_net, device, agent_name)
        self._epsilon = epsilon

    def get_action(self, obs):
        available_action = obs['available_action']
        obs = obs['obs']
        with torch.no_grad():
            agents_q, self.hidden_states = self.q_net(obs, self.hidden_states)
        agents_q[available_action == 0] = -1e8

        if self.is_explore():
            random_numbers = torch.rand_like(agents_q[:, 0])
            pick_random = (random_numbers < self._epsilon).long()

            random_actions = Categorical(available_action.float()).sample().long()

            actions = pick_random * random_actions + (1 - pick_random) * agents_q.max(dim=-1)[1]

            return actions.detach().cpu().numpy()
        else:
            return agents_q.max(dim=-1)[1].detach().cpu().numpy()

    def get_value(self, obs_batch, batch_size, episode_len):
        local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :]
            obs = obs.reshape(shape=(-1, obs_batch.size(-1)))
            local_q, self.hidden_states = self.q_net(obs, self.hidden_states)
            local_q = local_q.reshape(shape=(batch_size, -1))
            local_qs.append(local_q)
        local_qs = torch.stack(local_qs)
        return local_qs


class MultiAgent(QMIXAgentBase):
    def __init__(self, device, epsilon):
        self.device = device
        self.state_shape = 26
        self.n_agents = 2
        self.radar_q_net = RNNQ_net(12, 6)
        self.fire_q_net = RNNQ_net(30, 6)
        self.qmix_net = QMixNet(2, 26)
        self.to(device)
        agent_list = [SubAgent(device, 'agent_fire', self.fire_q_net, epsilon),
                      SubAgent(device, 'agent_radar', self.radar_q_net, epsilon)]
        super().__init__(agent_list)

    def parameters(self):
        return list(self.radar_q_net.parameters()) + list(self.fire_q_net.parameters()) + list(self.qmix_net.parameters())

    def get_value(self, obs_batch, batch_size, episode_len):
        local_qs = []
        for agent in self.agent_list:
            local_qs.append(agent.get_value(obs_batch[agent.name], batch_size, episode_len))
        local_qs = torch.stack(local_qs).transpose(0, 2)

        return local_qs

    def get_global_value(self, agent_qs, states):
        return self.qmix_net(agent_qs, states)

    def to(self, device):
        self.qmix_net.to(device)
        self.radar_q_net.to(device)
        self.fire_q_net.to(device)

    def update_by(self, agent, tao=0.1):
        """
        Update the agent by the input source agent
        By default, all sub agent use the same q_net

        Args:
            agent: source agent
            tao: how depth we exchange the part of the nn
        """
        update_network(self.qmix_net, agent.qmix_net, tao)
        update_network(self.radar_q_net, agent.radar_q_net, tao)
        update_network(self.fire_q_net, agent.fire_q_net, tao)

    def load(self, base_dir):
        for agent in self.agent_list:
            agent.load(os.path.join(base_dir, agent.name))
