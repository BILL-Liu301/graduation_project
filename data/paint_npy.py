import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import colors

white_line = np.load("230415/npys/white_line.npy")
lane = np.load("230415/npys/lane.npy")

plt.figure()

plt.title(label="White Line & Lane")
plt.plot(lane[:, 1], lane[:, 2], ".", label='Lane', color="k")
plt.plot(white_line[:, 1], white_line[:, 2], "*", label='White Line', color="k")
plt.legend(loc='upper right')
plt.show()

for lane_no in range(1, 3, 1):
    plt.title(label="Vehicle Trajectory")
    for index in range(0, 30):
        npy = np.load("230507/npys/" + str(lane_no) + "/" + str(index) + ".npy")
        if index == 0:
            plt.plot(npy[:, 1], npy[:, 2],
                     "c--" * int(2 - lane_no) + "y--" * int(lane_no - 1), label='Vehicle Trajectory')
        else:
            plt.plot(npy[:, 1], npy[:, 2],
                     "c--" * int(2 - lane_no) + "y--" * int(lane_no - 1))

plt.plot(lane[:, 1], lane[:, 2], ".", label='Lane', color="k")
plt.plot(white_line[:, 1], white_line[:, 2], "*", label='White Line', color="k")
plt.legend(loc='upper right')
plt.show()

