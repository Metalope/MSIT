import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np


class PltRender:
    def __init__(self, target_size, station_size, x_lim=1100, y_lim=1100):
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.set_figheight(10)
        self.fig.set_figwidth(10)
        self.target_size = target_size
        self.station_size = station_size
        self.bullets_group = []
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.sta_pos = []
        self.count = 0

    def update(self, bullets, sta_pos, tar_pos, det_pos, random_pos, interval=0.01, info=""):
        self.ax.clear()
        station = mpatch.Rectangle((sta_pos[0], sta_pos[1]), self.station_size[0], self.station_size[1], color="y", alpha=1)
        target = mpatch.Rectangle((tar_pos[0], tar_pos[1]), self.target_size[0], self.target_size[1], color="r", alpha=1)
        random = mpatch.Rectangle((random_pos[0], random_pos[1]), self.target_size[0], self.target_size[1], color="g", alpha=1) # target with random pos
        detector = mpatch.Polygon(([det_pos[0][0], det_pos[0][1]],
                                   [det_pos[1][0], det_pos[1][1]],
                                   [det_pos[2][0], det_pos[2][1]]), color="blue", alpha=0.5)

        self.ax.add_patch(station)
        self.ax.add_patch(target)
        self.ax.add_patch(random)
        self.ax.add_patch(detector)

        self.bullets_group = []
        for bullet in bullets:
            self.bullets_group.append(bullet.current_pos)

        for item in self.bullets_group:
            bullet = mpatch.Rectangle((item[0], item[1]), 6, 6, color="black", alpha=1)
            self.ax.add_patch(bullet)

        self.ax.set_xlim(-100, self.x_lim)
        self.ax.set_ylim(-100, self.y_lim)

        for item in range(len(info)):
            self.ax.text(10, 1000+25*(item+1), info[item])

        self.ax.text(10, 1000, f"Targt_pos:[{tar_pos[0]}, {tar_pos[1]}]")
        self.ax.text(10, 1075, f"Station_pos:[{sta_pos[0]}], [{sta_pos[1]}]")
        self.ax.plot()

        self.sta_pos.append(0.1*(sta_pos[0]-500))

        # if self.count >= 100:
        #     self.ax[1].set_xlim(self.count-100, self.count+20)
        # else:
        #     self.ax[1].set_xlim(0, 120)
        #
        # self.ax[1].set_ylim(-20, 20)

        self.count += 1
        if self.count == 1000:
            self.count = 0
            self.sta_pos = []
        plt.pause(interval)

    @staticmethod
    def close_plt():
        plt.close("all")
