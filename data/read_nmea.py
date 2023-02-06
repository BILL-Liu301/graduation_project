import numpy as np
import math as m
import matplotlib.pyplot as plt

file_path = '230203/0203_nmea_sentence.txt'
data_source = open(file_path, 'r').readlines()

data_output = np.zeros((1, 2))
k_lat = m.pi * 6371393 * m.cos(data_output[0, 1] * m.pi / 360) / 180
k_lon = m.pi * 6371393 / 180

for each_line in data_source:
    if each_line[0:8] == 'sentence':
        each_line_separate = each_line.split(',')
        for i in range(len(data_output[-1])):
            data_output[-1, i] = float(each_line_separate[12+i])
        temp = np.zeros((len(data_output)+1, len(data_output[-1])))
        temp[0:len(data_output), :] = data_output[0:len(data_output), :]
        data_output = temp


data_output = np.delete(data_output, [len(data_output)-1], 0)

data_output[:, :] = data_output[:, :] - data_output[0, :]
data_output[:, 0] = data_output[:, 0] * k_lat
data_output[:, 1] = data_output[:, 1] * k_lon

plt.figure("数据源")
plt.plot(data_output[:, 1], data_output[:, 0])
plt.show()
