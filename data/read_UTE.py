import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as t
import math as m

# frame,time,car_id,lane_id,longitude,latitude,speed<km/h>,acceleration<m/s2>,length,width
path = "Ubiquitous_Traffic_Eyes/06/KZM6frenet.csv"
data_source = pd.read_csv(path, header=0, sep=",")
data_source = np.array(data_source)
car_ids = 100

# time x y V car_id
init_t = data_source[0, 1]
init_x = data_source[0, 4]
init_y = data_source[0, 5]
k_x = 10
k_y = 10
index = 0
for car_id in range(1, car_ids):
    start = index
    while True:
        if int(data_source[index, 2]) == car_id:
            index = index + 1
        else:
            break
    data_temp = np.array(data_source[start:index, [1, 4, 5, 6, 2]])

    data_temp[:, 0] = data_temp[:, 0] - init_t
    data_temp[:, 1] = (data_temp[:, 1] - init_x) * k_x
    data_temp[:, 2] = (data_temp[:, 2] - init_y) * k_y
    np.save("Ubiquitous_Traffic_Eyes/datas/" + str(car_id) + ".npy", data_temp)

# plt.figure()
# for car_id in range(1, car_ids):
#     data_temp = np.load("Ubiquitous_Traffic_Eyes/datas/" + str(car_id) + ".npy")
#     plt.xlim(-800, 150)
#     plt.ylim(-300, 300)
#     for i in range(data_temp.shape[0]):
#         plt.plot(data_temp[i, 1], data_temp[i, 2], "*")
#         if i == int(data_temp.shape[0] - 1):
#             print("--------------")
#             plt.pause(data_temp[i, 0] - data_temp[i - 1, 0])
#             break
#         plt.pause(data_temp[i + 1, 0] - data_temp[i, 0])
#     # plt.clf()

