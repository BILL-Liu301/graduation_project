import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import colors

for lane in range(1, 3, 1):
    for index in range(0, 30):
        file_path = "230507/" + str(lane) + "/" + str(index) + ".txt"
        data_source = open(file_path, 'r').readlines()

        # stamp x y Vx Vy heading
        data = np.array([[0, 0.0, 0.0, 0.0, 0.0, 0.0]])

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
                data[-1, 5] = float(each_line_separate[7])  # 与正北偏航
                data = np.r_[data, np.array([[0, 0.0, 0.0, 0.0, 0.0, 0.0]])]

        data = np.delete(data, data.shape[0] - 1, 0)

        init_lon = 113.3363841
        init_lat = 23.1690645
        k_lat = m.pi * 6371393 * m.cos(init_lat * m.pi / 180) / 180
        k_lon = m.pi * 6371393 / 180
        init_heading = 184.71
        data[:, 1] = data[:, 1] - init_lon
        data[:, 2] = data[:, 2] - init_lat
        # data[:, 5] = data[:, 5] - init_heading
        data[:, 1] = data[:, 1] * k_lon
        data[:, 2] = data[:, 2] * k_lat

        # plt.figure("数据源")
        # plt.plot(data[:, 1], "*")
        # plt.plot(data[:, 3], ".")
        # plt.show()

        np.save("230507/npys/" + str(lane) + "/" + str(index) + ".npy", data)
