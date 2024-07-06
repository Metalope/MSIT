from airctrl.env.multi_agent_env_base import MultiAgentEnvBase
import torch
import numpy as np
from env_new.api import FindKillApi
from airctrl.utils.one_hot import OneHotTransform

total_step_count = 0


class Fak(MultiAgentEnvBase):
    """Package the StarCraft2 (8 agents) in Multi-Agent Environment.
    """

    agent_name_list = ['agent_fire', 'agent_radar']

    def __init__(self, obs_callback=None, rew_callback=None, action_callback=None, done_callback=None, fire_interval=0,
                 detect_delay=0):
        """
        Args:
            obs_callback: observation callback,\
                it allow developers to customize the formation of observation
            rew_callback: reward callback,\
                it allow developers to customize the formation of reward
            action_callback: action callback,\
                it allow developers to integrate the rules with the neural network output
            done_callback: done callback,\
                it allow developers to determine whether the episode is done
        """

        super().__init__(obs_callback, rew_callback, action_callback, done_callback)

        self.env = FindKillApi(fire_inerval=fire_interval, detect_delay=detect_delay, is_discrete=True)
        self.n_agents = 2
        self.n_actions = 6
        self._agent_id_one_hot_transform = OneHotTransform(self.n_agents)  # 2 agents
        self._actions_one_hot_transform = OneHotTransform(self.n_actions)  # 6 actions

        self._init_agents_id_one_hot(self.n_agents)  # 2 agents

        self._episode_length = 0
        self._episode_rew = 0
        self._is_win = False

    def _init_agents_id_one_hot(self, n_agents):
        agents_id_one_hot = []
        for agent_id in range(n_agents):
            one_hot = self._agent_id_one_hot_transform(agent_id)
            agents_id_one_hot.append(one_hot)
        self.agents_id_one_hot = np.array(agents_id_one_hot)

    def _get_agents_id_one_hot(self):
        return self.agents_id_one_hot

    def _get_actions_one_hot(self, actions):
        actions_one_hot = []
        for action in actions:
            one_hot = self._actions_one_hot_transform(action)
            actions_one_hot.append(one_hot)
        return np.array(actions_one_hot)

    def get_available_actions(self):
        available_actions = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]  # 5 actions
        return np.array(available_actions)

    def reset(self):
        """
        reset the environment

        Returns:
            The dict of obs(include state, observation and available_action) and log
        """
        state_raw = self.env.reset()

        # obs_callback should be defined
        state, obs = self._obs_callback(state_raw)

        available_actions = self.get_available_actions()
        last_actions_one_hot = np.zeros((self.n_agents, self.n_actions), dtype='float32')
        agents_id_one_hot = self._get_agents_id_one_hot()
        for i in range(self.n_agents):
            obs[i] = np.concatenate([obs[i], last_actions_one_hot[i], agents_id_one_hot[i]], axis=-1)
        obs_dict = dict()
        for agent_name, o, available_action in zip(self.agent_name_list, obs, available_actions):
            obs_dict[agent_name] = {'state': state, 'obs': o, 'available_action': available_action}

        log = {'episode reward': self._episode_rew, 'episode length': self._episode_length, 'is win': self._is_win}

        self._episode_length = 0.0
        self._episode_rew = 0.0

        return obs_dict, log

    def step(self, action_dict):
        """
        Run a step

        Args:
            action_dict: array of all the agent actions.
        Returns:
            The dict of obs(include state, observation and available_action), reward and done information after applying the input action

        Raises:
            KeyError: The key in the input `action_dict` does not match the `agent_name_list`
        """
        action_list = []
        for agent_name in self.agent_name_list:
            try:
                action = action_dict[agent_name]
            except KeyError:
                print("There is no key {} in the input action_dict".format(agent_name))
                raise KeyError
            if isinstance(action, torch.Tensor):
                action = action.tolist()
            if isinstance(action, np.ndarray):
                action = action.tolist()

            action_list.append(action)
        if self._action_callback is not None:
            _action_list = self._action_callback(action_list)
            next_state_raw, reward, done, info = self.env.step(_action_list)
        else:
            next_state_raw, reward, done, info = self.env.step(action_list)

        # obs_callback should be defined
        next_state, next_obs = self._obs_callback(next_state_raw)

        self._is_win = done
        self._episode_length += 1
        self._episode_rew += reward

        available_actions = self.get_available_actions()
        last_actions_one_hot = self._get_actions_one_hot(action_list)
        for i in range(self.n_agents):
            next_obs[i] = np.concatenate([next_obs[i], last_actions_one_hot[i], self.agents_id_one_hot[i]], axis=-1)

        next_obs_dict = dict()
        for agent_name, o, available_action in zip(self.agent_name_list, next_obs, available_actions):
            if type(o) == np.ndarray:
                o = o.tolist()
            next_obs_dict[agent_name] = {'state': next_state, 'obs': o, 'available_action': available_action}

        if self._rew_callback is not None:
            reward = self._rew_callback(reward)

        return next_obs_dict, reward, done

    def hit_rate(self):
        return self.env.hit_rate()

    def detect_rate(self):
        return self.env.detect_rate()

    def get_episode_length(self):
        return self._episode_length

    def close(self):
        """ close the environment """
        self.env.close()

    @classmethod
    def process_episode_log(cls, log_list):
        """
        process the episode logs

        Args:
            log_list: a DictList object of log.

        Returns:
            episode logs
        """
        global total_step_count
        episode_rew_list = log_list['episode reward']
        episode_length = log_list['episode length']
        max_episode_rew = max(episode_rew_list)
        min_episode_rew = min(episode_rew_list)
        ave_episode_rew = sum(episode_rew_list) / len(episode_rew_list)

        max_episode_length = max(episode_length)
        min_episode_length = min(episode_length)
        ave_episode_length = sum(episode_length) / len(episode_length)

        return {
            'maximum_episode_reward': max_episode_rew,
            'minimum_episode_reward': min_episode_rew,
            'average_episode_reward': ave_episode_rew,
            'maximum_episode_length': max_episode_length,
            'minimum_episode_length': min_episode_length,
            'average_episode_length': ave_episode_length
        }
