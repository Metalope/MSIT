# -*- coding:utf-8 -*-
# @time: 2022/10/31 22:05
# @author: Metalope
import math
import random
import numpy as np
# legal pos range: [minx, miny, maxx, maxy]


class RectObjects(object):
    def __init__(self, size, initial_pos, move_function, speed, legal_pos_range=None):
        self.size = size
        self.ini_pos = initial_pos
        self.current_pos = initial_pos
        self.move_func = move_function
        self.speed = speed

        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None
        self.center = None
        self.update_rect()
        if legal_pos_range is None:
            self.legal_range = [0, 0, 1000, 1000]
        else:
            self.legal_range = legal_pos_range

    def update(self):
        next_x = self.current_pos[0] + self.speed
        next_y = self.move_func(next_x)
        self.current_pos = [next_x, next_y]
        self.update_rect()

        if self.bottom < self.legal_range[1]:
            next_y = self.legal_range[3] - self.size[1]
        if self.top > self.legal_range[3]:
            next_y = self.legal_range[1] + self.size[1]
        if self.right < self.legal_range[0]:
            next_x = self.legal_range[2] - self.size[0]
        if self.left > self.legal_range[2]:
            next_x = self.legal_range[0] + self.size[0]

        self.current_pos = [next_x, next_y]

    def reset(self, reset_pos=None):
        if reset_pos:
            self.current_pos = reset_pos
        else:
            self.current_pos = self.ini_pos
        self.update_rect()
        print("Object Reset")

    def update_rect(self):

        self.top = self.current_pos[1] + self.size[1]/2
        self.bottom = self.current_pos[1] - self.size[1]/2
        self.left = self.current_pos[0] - self.size[0]/2
        self.right = self.current_pos[0] + self.size[0]/2

        self.top_left = [self.left, self.top]
        self.top_right = [self.right, self.top]
        self.bottom_left = [self.left, self.bottom]
        self.bottom_right = [self.right, self.bottom]

        self.center = self.current_pos


class Station(RectObjects):
    def __init__(self, size, initial_pos, move_function, speed, ammo):
        super().__init__(size, initial_pos, move_function, speed)
        self.ammo = ammo
        self.fire_angle = 0
        self.current_fire_angle = random.randint(0, 90)

    def fire(self, b_angle, b_speed=100):
        if self.ammo > 0:
            self.fire_angle = b_angle
            bullet = Bullet(initial_pos=self.current_pos, angle=b_angle, speed=b_speed)
            self.ammo -= 1
            return bullet
        else:
            return None

    def fire_discrete(self, b_angle_discrete, b_speed=100):
        if self.ammo > 0:
            if b_angle_discrete > 0 and self.current_fire_angle + 1 <= 90:
                self.current_fire_angle += 1
            elif b_angle_discrete < 0 and self.current_fire_angle - 1 >= 0:
                self.current_fire_angle -= 1
            bullet = Bullet(initial_pos=self.current_pos, angle=self.current_fire_angle, speed=b_speed)
            self.ammo -= 1
            return bullet


class Target(RectObjects):
    def __init__(self, size, initial_pos, move_function, speed, hp):
        super().__init__(size, initial_pos, move_function, speed)
        self.hp = hp

    def get_future_pos(self, step_nums):
        current_pos = self.current_pos
        future_postion = [self.current_pos[0], self.current_pos[1]]

        for _ in range(step_nums):
            next_x = current_pos[0] + self.speed
            next_y = self.move_func(next_x)
            current_pos = [next_x, next_y]
            top, bottom, left, right = self.calculate_rect(current_pos)

            if bottom < self.legal_range[1]:
                next_y = self.legal_range[3] - self.size[1]
            if top > self.legal_range[3]:
                next_y = self.legal_range[1] + self.size[1]
            if right < self.legal_range[0]:
                next_x = self.legal_range[2] - self.size[0]
            if left > self.legal_range[2]:
                next_x = self.legal_range[0] + self.size[0]

            future_postion.append(next_x)
            future_postion.append(next_y)

            current_pos = [next_x, next_y]

        return future_postion

    def get_noise_pos_normal_distribution(self):
        """
        The probability of [0,0] is 0.1
        noise position(with a normal distribution) is 0.4
        position is 0.5
        Returns:

        """
        noise_pos = []
        random_index = random.random()
        range_index = 1 + np.random.normal(loc=0.0, scale=0.1, size=None)
        if random_index < 0.1:
            noise_pos = [0, 0]
        elif random_index > 0.5:
            noise_pos = self.current_pos
        else:
            for pos in self.current_pos:
                _pos = pos * range_index
                if _pos > 1000:
                    noise_pos.append(1000)
                elif _pos < 0:
                    noise_pos.append(0)
                else:
                    noise_pos.append(_pos)

        return noise_pos

    def get_noise_pos(self):
        """
        The probability of [0,0] is 0.1
        noise position(with a normal distribution) is 0.4
        position is 0.5
        Returns:

        """
        noise_pos = []
        random_index = random.random()
        range_index = 1 + np.random.normal(loc=0.0, scale=0.1, size=None)
        if random_index < 0.4:
            noise_pos = self.current_pos
        else:
            for pos in self.current_pos:
                _pos = pos + random.randint(-300, 300)
                noise_pos.append(_pos)

        return noise_pos

    def calculate_rect(self, current_pos):

        top = current_pos[1] - self.size[1]/2
        bottom = current_pos[1] + self.size[1]/2
        left = current_pos[0] - self.size[0]/2
        right = current_pos[0] + self.size[0]/2

        return top, bottom, left, right


class Bullet(object):
    def __init__(self, initial_pos, angle, speed, legal_pos_range=None):
        if legal_pos_range is None:
            self.legal_range = [0, 0, 1000, 1000]
        else:
            self.legal_range = legal_pos_range
        self.ini_pos = initial_pos
        self.current_pos = initial_pos
        self.past_pos = initial_pos
        self.angle = angle
        self.speed = speed

    def update(self):
        x = self.current_pos[0] - self.speed * math.cos(self.angle/180 * math.pi)
        y = self.current_pos[1] - self.speed * math.sin(self.angle/180 * math.pi)
        self.past_pos = self.current_pos
        self.current_pos = [x, y]

        if y < self.legal_range[1] or y > self.legal_range[3] or x < self.legal_range[0] or x > self.legal_range[2]:
            return False

        return True

    def kill(self):
        del self


class Detector(object):
    def __init__(self, initial_pos, initial_angle, detect_distance, detect_angle_range, legal_angle_range=None):
        self.current_pos = initial_pos
        self.ini_angle = initial_angle
        self.radar_dist = detect_distance
        self.radar_angle_range = detect_angle_range
        self.current_angle = initial_angle
        if not legal_angle_range:
            self.legal_range = [-180, 180]
        else:
            self.legal_range = legal_angle_range

    def radar_range(self):
        remote_medium = self.remote_point(0)
        remote_min = self.remote_point(-0.5 * self.radar_angle_range)
        remote_max = self.remote_point(0.5 * self.radar_angle_range)
        return remote_medium, remote_min, remote_max

    def radar_angle_edit(self, action=None):
        if action >= 0.33 and self.current_angle + 1 <= self.legal_range[1]:
            self.current_angle += 1
        elif action <= -0.33 and self.current_angle - 1 >= self.legal_range[0]:
            self.current_angle -= 1

    def update(self, station_pos):
        self.current_pos = station_pos

    def remote_point(self, angle_offset):
        return [self.current_pos[0] - self.radar_dist * math.cos((self.current_angle + angle_offset) / 180 * math.pi),
                self.current_pos[1] + self.radar_dist * math.sin((self.current_angle + angle_offset) / 180 * math.pi)]
