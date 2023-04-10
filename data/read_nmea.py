import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import colors

file_path = '230410/0410_nmea_sentence.txt'
data_source = open(file_path, 'r').readlines()

# stamp x y Vx Vy
data = np.array([[0, 0.0, 0.0, 0.0, 0.0]])

for each_line in data_source:
    # print(each_line[50:56])
    if each_line[50:56] == '$GPAGM':
        each_line_separate = each_line.split(',')
        # print(each_line_separate)
        data[-1, 0] = data.shape[0]
        data[-1, 1] = float(each_line_separate[17])  # 经度
        data[-1, 2] = float(each_line_separate[16])  # 纬度
        data[-1, 3] = float(each_line_separate[20])  # 东向速度
        data[-1, 4] = float(each_line_separate[19])  # 北向速度
        data = np.r_[data, np.array([[0, 0.0, 0.0, 0.0, 0.0]])]

data = np.delete(data, data.shape[0] - 1, 0)

k_lat = m.pi * 6371393 * m.cos(data[0, 1] * m.pi / 180) / 180
k_lon = m.pi * 6371393 / 180
data[:, 1] = data[:, 1] * k_lon
data[:, 2] = data[:, 2] * k_lat
data[:, 1] = data[:, 1] - data[0, 1]
data[:, 2] = data[:, 2] - data[0, 2]

# plt.figure("数据源")
# plt.plot(data[:, 1], data[:, 2])
# plt.show()

np.savetxt('source.txt', data, fmt='%f')
np.save('source.npy', data)
