# -*- coding:utf-8 -*-
# @time: 2023/3/8 16:34
# @author: Metalope
import numpy as np


class BufferPool:

    def __init__(self, nums, momentum=0.2, update_freq=100):
        self._pool = [[] for _ in range(nums)]

        self._old_mean = [0 for _ in range(nums)]
        self._old_std = [0 for _ in range(nums)]
        self._old_nums = [0 for _ in range(nums)]

        self._new_mean = [0 for _ in range(nums)]
        self._new_std = [0 for _ in range(nums)]
        self._new_nums = [0 for _ in range(nums)]

        self.running_mean = [0 for _ in range(nums)]
        self.running_std = [0 for _ in range(nums)]
        self._momentum = momentum
        self._update_freq = update_freq

    def push(self, _key, value):
        self._pool[_key].append(value)

        if len(self._pool[_key]) % self._update_freq == 0:
            self.cal(_key)

    def get(self, _key):
        return self.running_mean[_key], self.running_std[_key]

    def cal(self, _key):

        add_mean = np.array(self._pool[_key]).mean()
        add_std = np.array(self._pool[_key]).std()
        add_nums = np.array(self._pool[_key]).size

        self._new_mean[_key] = (self._old_nums[_key] * self._old_mean[_key] + add_nums * add_mean) / \
                               (self._old_nums[_key] + add_nums)

        self._new_std[_key] = np.sqrt(
            (self._old_nums[_key] *
             (self._old_std[_key] ** 2 + (self._new_mean[_key] - self._old_mean[_key]) ** 2) +
             (add_std ** 2 + (self._new_mean[_key] - add_mean) ** 2)) /
            (self._old_nums[_key] + add_nums)
        )

        self.running_mean[_key] = (1 - self._momentum) * self._old_mean[_key] + self._momentum * self._new_mean[_key]
        self.running_std[_key] = (1 - self._momentum) * self._old_std[_key] + self._momentum * self._new_std[_key]

        # self._old_mean[_key] = self._new_mean[_key]
        # self._old_std[_key] = self._new_std[_key]
        self._old_mean[_key] = self.running_mean[_key]
        self._old_std[_key] = self.running_std[_key]
        self._old_nums[_key] = self._old_nums[_key] + add_nums

        self._pool[_key].clear()


if __name__ == "__main__":
    x = BufferPool(2)
    for i in range(0, 200):
        x.push(0, i)
        x.push(1, i)
