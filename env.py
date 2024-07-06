from airctrl.env.single_agent_env_base import SingleAgentEnvBase
from env_new.api import FindKillApi

PHASE = 1
total_step_count = 0


class Fak(SingleAgentEnvBase):
    def __init__(self, obs_callback=None, rew_callback=None, action_callback=None, done_callback=None,
                 fire_interval=0, detect_delay=0, is_discrete=False):
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
        # gym.spec('MountainCar-v0').max_episode_steps = None
        self.env = FindKillApi(is_discrete, fire_inerval=fire_interval, detect_delay=detect_delay)
        self.step_count = 0
        self._episode_rew = 0
        self._episode_length = 0

        self._obs = None

    def reset(self):
        obs = self.env.reset()
        if self._obs_callback is not None:
            obs = self._obs_callback(obs)
        self._obs = obs

        log = {'episode reward': self._episode_rew,
               'episode length': self._episode_length,
               'step count': self.step_count}
        self._episode_rew = 0
        self._episode_length = 0

        return obs, log

    def step(self, action):
        if self._render:
            self.env.render()

        if self._action_callback is not None:
            action = self._action_callback(action)

        obs, reward, done, info = self.env.step(action)

        if self._obs_callback is not None:
            obs = self._obs_callback(obs)

        if self._rew_callback is not None:
            reward += self._rew_callback(obs, self._obs)

        self._obs = obs

        self._episode_length += 1
        self._episode_rew += reward

        self.step_count += 1

        return obs, reward, done

    def hit_rate(self):
        return self.env.hit_rate()

    def detect_rate(self):
        return self.env.detect_rate()

    def get_episode_length(self):
        return self._episode_length

    def close(self):
        self.env.close()

    @classmethod
    def process_episode_log(cls, log_list):
        """
        process the episode logs

        Args:
            log_list: a DictList object of log.

        Returns:
            episode logs, \
            contains `maximum(minimum, average) episode reward` and `maximum(minimum, average) episode length`
        """

        global total_step_count
        episode_rew_list = log_list['episode reward']
        episode_length = log_list['episode length']
        step_count = log_list['step count']
        max_episode_rew = max(episode_rew_list)
        min_episode_rew = min(episode_rew_list)
        ave_episode_rew = sum(episode_rew_list) / len(episode_rew_list)

        max_episode_length = max(episode_length)
        min_episode_length = min(episode_length)
        ave_episode_length = sum(episode_length) / len(episode_length)

        total_step_count += sum(step_count)

        return {
            'maximum_episode_reward': max_episode_rew,
            'minimum_episode_reward': min_episode_rew,
            'average_episode_reward': ave_episode_rew,
            'maximum_episode_length': max_episode_length,
            'minimum_episode_length': min_episode_length,
            'average_episode_length': ave_episode_length,
            'step_count': total_step_count
        }
