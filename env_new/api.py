# -*- coding:utf-8 -*-
# @time: 2022/9/11 11:22
# @author: Metalope
from env_new.main_game import FindKill
import random


"""
State:
{
    "current_radar_angle": float,
    "current_gun_angle": float,
    "remain_ammo": int,
    * "target_center_pos": (x, y), if radar_detected is None return (-1, -1)
    * "target_HP": int, if radar_detected is None return -1
}
"""


class FindKillApi(object):
    def __init__(self, is_discrete, fire_inerval=0, detect_delay=0):
        self.game = FindKill(fire_inerval, detect_delay, is_discrete)
        self.step_count = 0

    def reset(self):
        state = self.game.reset_game()
        state_2 = self.game.get_future_pos()
        state["future_pos"] = state_2
        state_3 = self.game.get_past_pos()
        state["past_pos"] = state_3
        state_4 = self.game.get_past_pos_detect()
        state["past_pos_detect"] = state_4

        return state

    def render(self):
        self.game.render_object()

    def step(self, action: dict):
        radar_angle = action["radar_angle"]  # The direction of radar
        gun_angle = action["gun_angle"]  # The direction of radar
        is_fire = action["is_fire"]  # Fire or not
        state, reward, done, info = self.game.main_game_step(radar_angle, gun_angle, is_fire)

        state_2 = self.game.get_future_pos()
        state["future_pos"] = state_2
        state_3 = self.game.get_past_pos()
        state["past_pos"] = state_3
        state_4 = self.game.get_past_pos_detect()
        state["past_pos_detect"] = state_4
        return state, reward, done, info

    def close(self):
        self.game.game_over()

    def seed(self):
        pass

    def hit_rate(self):
        return self.game.hit_rate()

    def detect_rate(self):
        return self.game.detect_rate()


if __name__ == "__main__":
    # for single test

    env = FindKillApi(fire_inerval=0, detect_delay=9)
    env.reset()

    num = 0
    total_reward = 0
    while True:
        actions = {"radar_angle": -1, "gun_angle": random.randint(0, 90),
                   "is_fire": bool(random.randint(0, 1)), "force": 0}
        state, reward, done, info = env.step(actions)
        # print(f"state: {state}\t reward: {reward}\t done: {done}\t")
        total_reward += reward
        env.render()
        if num % 1000 == 0:
            print(total_reward)
            total_reward = 0
            env.reset()
        num += 1
