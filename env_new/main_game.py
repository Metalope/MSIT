# -*- coding:utf-8 -*-
# @time: 2022/10/29 20:05
# @author: Metalope
from env_new.objects import *
from utils.tool import *
from env_new.render import PltRender
import numpy as np
import time
import random

from utils.buffer_normalize import BufferPool


class FindKill(object):
    def __init__(self, fire_interval, detect_delay, is_discrete):
        # Game Settings
        self.render_obj = None
        self.has_called = False
        self.world_size = [1000.0, 1000.0]
        self.target_size = [50.0, 50.0]
        self.station_size = [50.0, 50.0]
        self.radar_range = [1400, 10]  # distance & angle
        self.ammo_equipment = 100
        self.fire_interval = fire_interval
        self.detect_delay = detect_delay

        # Initial
        self.step_detector_reward = 0
        self.step_fire_reward = 0

        # Normalize
        self.mean_signal = 0
        self.std_signal = 0
        self.std_pos = 0
        self.mean_pos = 0

        # Game Initiation
        self.bullets_group = []
        self.target_pos = None
        self.target_pos_detect = None
        self.target_last_info = {}
        self.past_pos = [-1000 for _ in range(20)]
        self.past_pos_detect = [-1000 for _ in range(20)]

        self.is_detected = True
        self.is_random_init = True


        self.__create_handler()
        self.hp = self.target.hp
        self.start_time = time.time()

        # Performance
        self.fire_action_step = 0
        self.detect_action_step = 0
        self.current_step = 0

        self.fire_count = 0
        self.hit_count = 0
        self.detected_count = 0
        self.total_count = 0

        self.is_discrete = is_discrete

    def main_game(self):

        while True:
            # 事件监听
            self.__event_handler()
            # 碰撞检测
            self.__check_collide()
            # 信息展示
            self.__info_handler()
            # 更新/绘制精灵组
            self.__update_objects()
            # 更新显示

    def __create_handler(self):
        if not self.is_random_init:
            movefunc = MoveFunc()
            self.target = Target(self.target_size, [0, movefunc.target_move_func(0)],
                                 movefunc.target_move_func, speed=1, hp=20)
        else:
            movefunc = MoveFunc(
                                amplitude=random.randint(50, 150),
                                phase=random.randint(50, 150),
                                offset=random.randint(400, 600)
                                )
            pos_x = random.randint(0, 1000)
            self.target = Target(self.target_size, [pos_x, movefunc.target_move_func(0)],
                                 movefunc.target_move_func, speed=1, hp=20)
        self.station = Station(self.station_size, [975, 975], station_move_func, speed=0, ammo=self.ammo_equipment)
        self.detector = Detector(initial_pos=self.station.current_pos, initial_angle=random.randint(-90, 0),
                                 detect_distance=self.radar_range[0], detect_angle_range=self.radar_range[1])

    def __event_handler(self):
        pass

    def __info_handler(self):
        pass

    def __check_collide(self):
        # bullets and target
        rectangle = [self.target.top_left, self.target.bottom_right]
        for bullet in self.bullets_group:
            segment = [bullet.past_pos, bullet.current_pos]
            if seg_rect_intersect(rectangle, segment):
                self.target.hp -= 1
                self.step_fire_reward += 1
                self.hit_count += 1
                self.bullets_group.remove(bullet)
                del bullet

    def __check_detect(self):
        # Radar detection
        point = self.target.current_pos
        if self.detector.current_pos[0] - point[0] == 0 and self.detector.current_pos[1] >= point[1]:
            target_angle = -90
        elif self.detector.current_pos[0] - point[0] == 0 and self.detector.current_pos[1] < point[1]:
            target_angle = 90
        else:
            # Different Position has different angle calculation
            if self.detector.current_pos[0] > point[0]:
                target_angle = math.atan((point[1] - self.detector.current_pos[1]) /
                                         (self.detector.current_pos[0] - point[0])) / math.pi * 180
            elif point[1] >= self.detector.current_pos[1]:
                target_angle = math.atan((point[1] - self.detector.current_pos[1]) /
                                         (self.detector.current_pos[0] - point[0])) / math.pi * 180 + 180
            else:
                target_angle = math.atan((point[1] - self.detector.current_pos[1]) /
                                         (self.detector.current_pos[0] - point[0])) / math.pi * 180 - 180

        target_dist = math.sqrt(((point[1] - self.detector.current_pos[1]) ** 2 +
                                 (self.detector.current_pos[0] - point[0]) ** 2))

        if self.detector.current_angle - 0.5 * self.detector.radar_angle_range <= target_angle \
                <= self.detector.current_angle + 0.5 * self.detector.radar_angle_range \
                and target_dist <= self.detector.radar_dist:
            self.step_detector_reward += 1
            self.is_detected = True
            self.detected_count += 1
            return True
        else:
            self.is_detected = False
            return False

    def __update_objects(self):
        self.target.update()
        self.station.update()
        self.detector.update(self.station.current_pos)
        for bullet in self.bullets_group:
            live_info = bullet.update()
            if not live_info:
                self.bullets_group.remove(bullet)
                del bullet

    """
    For API
    """

    def render_object(self):
        if not self.has_called:
            self.render_obj = PltRender(self.target_size, self.station_size)
            self.has_called = True
            
        if self.target.hp < self.hp:
            self.hp = self.target.hp
            hit_info = f"Hit Hit Hit! {self.target.hp}"
        else:
            hit_info = f"Do not Hit! {self.target.hp}"

        if self.is_detected:
            hit_info += "### Detected!"
        else:
            hit_info += "*** Undetected!"

        _, min_point, max_point = self.detector.radar_range()

        self.render_obj.update(self.bullets_group, self.station.bottom_left, self.target.bottom_left,
                               [self.detector.current_pos, min_point, max_point],
                               [self.target_pos_detect[0]-self.target.size[0]/2, self.target_pos_detect[1]-self.target.size[1]/2],
                               info=[hit_info])

    def __action_handler(self, radar_angle, gun_angle, is_fire):
        self.current_step += 1
        self.total_count += 1
        # if is_fire and self.is_detected:
        if is_fire:
            if self.current_step - self.fire_action_step > self.fire_interval:
                if not self.is_discrete:
                    bullet = self.station.fire(gun_angle)
                else:
                    bullet = self.station.fire_discrete(gun_angle)
                if bullet:
                    # FIRE penetration
                    self.step_fire_reward -= 0.1
                    self.fire_action_step = self.current_step
                    self.bullets_group.append(bullet)
                    self.fire_count += 1
        self.detector.radar_angle_edit(radar_angle)

    def main_game_step(self, radar_angle, gun_angle, is_fire):

        self.step_detector_reward = 0
        self.step_fire_reward = 0
        kill_reward = 0

        self.__event_handler()
        self.__action_handler(radar_angle, gun_angle, is_fire)
        self.__check_detect()
        self.__check_collide()
        self.__info_handler()
        self.__update_objects()

        state = self.get_state()
        info = None

        # Done
        if self.target.hp <= 0:
            done = True
            kill_reward = 40
            self.step_fire_reward += kill_reward
        else:
            done = False

        # Reward
        # step_reward = kill_reward  # Sparse Reward
        step_reward = self.step_fire_reward + self.step_detector_reward  # Base reshaped reward
        # step_reward = 0.02 * self.step_fire_reward + 0.02 * self.step_detector_reward  # Reshaped reward

        return state, step_reward, done, info

    def reset_game(self):
        # phase 1
        self.is_detected = True

        # clear memory
        del self.detector
        del self.target
        del self.station
        del self.bullets_group

        # create new
        self.__create_handler()
        self.start_time = time.time()
        self.bullets_group = []
        self.fire_action_step = 0
        self.current_step = 0

        # Performance
        self.fire_count = 0
        self.hit_count = 0
        self.detected_count = 0
        self.total_count = 0

        self.main_game_step(random.randint(-1, 1), random.randint(0, 90), False)

        return self.get_state()

    def get_state(self):
        state_dict = {}

        if self.current_step - self.detect_action_step > self.detect_delay or self.current_step <= 1:
            self.detect_action_step = self.current_step
            # Ground Truth position
            state_dict["target_pos"] = self.target.center
            self.target_pos = self.target.center

            # Position which is decided by detection results.
            if self.is_detected:
                state_dict["detect"] = 1
                state_dict["target_pos_detect"] = self.target_pos
                self.target_pos_detect = self.target_pos
                state_dict["target_HP"] = self.target.hp
            else:
                state_dict["detect"] = 0
                # self.target_pos_detect = (self.target.center[0] + random.randint(-300, 300),
                #                           self.target.center[1] + random.randint(-300, 300))
                # Test different noise
                self.target_pos_detect = (self.target.center[0] + np.random.normal(50, 400),
                                          self.target.center[1] + np.random.normal(50, 400))
                state_dict["target_pos_detect"] = self.target_pos_detect
                state_dict["target_HP"] = -1

            self.target_last_info["pos"] = state_dict["target_pos"]
            self.target_last_info["detect"] = state_dict["detect"]
            self.target_last_info["hp"] = state_dict["target_HP"]
            self.target_last_info["pos_detect"] = state_dict["target_pos_detect"]
        else:
            state_dict["target_pos"] = self.target_last_info["pos"]
            state_dict["detect"] = self.target_last_info["detect"]
            state_dict["target_HP"] = self.target_last_info["hp"]
            state_dict["target_pos_detect"] = self.target_last_info["pos_detect"]

        state_dict["current_radar_angle"] = self.detector.current_angle
        state_dict["current_gun_angle"] = self.station.fire_angle
        state_dict["remain_ammo"] = self.station.ammo

        return state_dict

    def get_past_pos(self):
        self.past_pos.pop(0)
        self.past_pos.pop(0)
        self.past_pos += self.target_pos
        return self.past_pos

    def get_past_pos_detect(self):
        self.past_pos_detect.pop(0)
        self.past_pos_detect.pop(0)
        self.past_pos_detect += self.target_pos_detect
        return self.past_pos_detect

    def get_future_pos(self):
        pos_list = self.target.get_future_pos(9)
        return pos_list

    def update_target(self):
        self.target.update()

    def hit_rate(self):
        if self.fire_count > 0:
            return self.hit_count / self.fire_count
        else:
            return -0.001

    def detect_rate(self):
        return self.detected_count / self.total_count

    def game_over(self):
        # clear memory
        del self.detector
        del self.target
        del self.station
        del self.bullets_group


class MoveFunc:
    def __init__(self, amplitude=200, phase=100, offset=500):
        self.amplitude = amplitude
        self.phase = phase
        self.offset = offset

    def target_move_func(self, x):
        y = self.amplitude * math.cos(x / self.phase) + self.amplitude * math.sin(x / self.phase) + self.offset
        return y


def station_move_func(x):
    return x


