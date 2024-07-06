from collections import deque
import numpy as np
import random
import time
import os


RESTORE_SIZE = 3000


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(* batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def restore(self):
        source = []
        for i in range(-RESTORE_SIZE, 0):
            source.append(self.memory[i])
        replay_buffer_save(source)

    def load(self):
        for items in replay_buffer_load():
            self.memory.append(items)

    def __len__(self):
        return len(self.memory)


def replay_buffer_save(source):
    time_stamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    file_path = './replay_buffer/'+time_stamp
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    np.save(file_path + '/rebuff.npy', source)


def replay_buffer_load():
    file_path = './replay_buffer_source/rebuff.npy'
    return np.load(file_path, allow_pickle=True)