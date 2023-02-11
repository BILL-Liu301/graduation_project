import numpy as np
import math as m
import matplotlib.pyplot as plt

file_path = '230203/0203_nmea_sentence.txt'
data_source = open(file_path, 'r').readlines()

lat_lon = np.zeros((1, 2))
k_lat = m.pi * 6371393 * m.cos(lat_lon[0, 1] * m.pi / 360) / 180
k_lon = m.pi * 6371393 / 180

for each_line in data_source:
    if each_line[0:8] == 'sentence':
        each_line_separate = each_line.split(',')
        for i in range(len(lat_lon[-1])):
            lat_lon[-1, i] = float(each_line_separate[12+i])
        temp = np.zeros((len(lat_lon)+1, len(lat_lon[-1])))
        temp[0:len(lat_lon), :] = lat_lon[0:len(lat_lon), :]
        lat_lon = temp


lat_lon = np.delete(lat_lon, [len(lat_lon)-1], 0)

lat_lon[:, :] = lat_lon[:, :] - lat_lon[0, :]
lat_lon[:, 0] = lat_lon[:, 0] * k_lat
lat_lon[:, 1] = lat_lon[:, 1] * k_lon

# plt.figure("数据源")
# plt.plot(lat_lon[:, 1], lat_lon[:, 0])
# plt.show()

# stamp、acc、position、velocity
data_output = np.zeros((len(lat_lon), 4))
data_output[:, 2] = lat_lon[:, 0]
data_output[:, 0] = np.linspace(1, len(data_output), len(data_output))
np.savetxt('source.txt', data_output, fmt='%f')
# plt.figure("data_output")
# plt.stairs(data_output[:, 2])
# plt.show()
