import math
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

# stamp x y Vx Vy
file_name = 'source.txt'
file_data = open(file_name, 'r')
file_data_lines = file_data.readlines()
data_size = 3
data = np.zeros((len(file_data_lines), data_size))
temp = 0
for row in file_data_lines:
    find_data = row.split(' ')
    data[temp, 0] = float(find_data[0])
    data[temp, 1] = float(find_data[1])
    data[temp, 2] = float(find_data[2])
    temp += 1

seq_size = 50
jump_size = int(seq_size / 2)
split_size = 10
# # 滑窗重组
# data_reshape = np.array([[0, 0.0, 0.0]])
# for i in range(0, data.shape[0] - jump_size - seq_size*split_size, jump_size):
#     for j in range(0, seq_size*split_size, split_size):
#         data_reshape[-1, :] = np.array(data[i+j, :])
#         data_reshape = np.r_[data_reshape, np.array([[0, 0.0, 0.0]])]
# np.save("data_reshape.npy", data_reshape)

# 加载滑窗重组的结果
data_reshape = np.load("data_reshape.npy")
data_reshape = np.delete(data_reshape, [-1], 0)
data_reshape = np.array(data_reshape.reshape(-1, seq_size, data_size))

# 设置边界格，两行三列，1m*1m
side_length_x = 2
side_length_x_center = 3
side_length_y = 4
row = 3
column = 3

# 分割input和output
input_size = int(0.5 * seq_size)
output_size = seq_size - input_size
train_size = int(0.8 * data_reshape.shape[0])
test_size = data_reshape.shape[0] - train_size

training_data_input = np.array(data_reshape[0:train_size, 0:input_size, :])
# training_data_output_origin = np.array(data_reshape[0:train_size, (input_size + 1):(input_size + 2), :])
training_data_output_origin = np.array(data_reshape[0:train_size, input_size:data_reshape.shape[1], :])

testing_data_input = np.array(data_reshape[train_size:data_reshape.shape[0], 0:input_size, :])
# testing_data_output_origin = np.array(data_reshape[train_size:data_reshape.shape[0], (input_size + 1):(input_size + 2), :])
testing_data_output_origin = np.array(
    data_reshape[train_size:data_reshape.shape[0], input_size:data_reshape.shape[1], :])

# 采用相对坐标，以训练集最后一个点为原点
for i in range(training_data_input.shape[0]):
    for j in range(training_data_output_origin.shape[1]):
        training_data_output_origin[i, j, 1] = training_data_output_origin[i, j, 1] - training_data_input[i, -1, 1]
        training_data_output_origin[i, j, 2] = training_data_output_origin[i, j, 2] - training_data_input[i, -1, 2]
    for j in range(training_data_input.shape[1]):
        training_data_input[i, j, 1] = training_data_input[i, j, 1] - training_data_input[i, -1, 1]
        training_data_input[i, j, 2] = training_data_input[i, j, 2] - training_data_input[i, -1, 2]

for i in range(testing_data_input.shape[0]):
    for j in range(testing_data_output_origin.shape[1]):
        testing_data_output_origin[i, j, 1] = testing_data_output_origin[i, j, 1] - testing_data_input[i, -1, 1]
        testing_data_output_origin[i, j, 2] = testing_data_output_origin[i, j, 2] - testing_data_input[i, -1, 2]
    for j in range(testing_data_input.shape[1]):
        testing_data_input[i, j, 1] = testing_data_input[i, j, 1] - testing_data_input[i, -1, 1]
        testing_data_input[i, j, 2] = testing_data_input[i, j, 2] - testing_data_input[i, -1, 2]

# 旋转数据组
for i in range(training_data_input.shape[0]):
    theda = math.atan2(training_data_input[i, -5, 2],
                       training_data_input[i, -5, 1])  # 与x轴的夹角
    theda = 3 * math.pi / 2 - theda

    for j in range(training_data_input.shape[1]):
        x_temp = training_data_input[i, j, 1]
        y_temp = training_data_input[i, j, 2]
        training_data_input[i, j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
        training_data_input[i, j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)

    for j in range(training_data_output_origin.shape[1]):
        x_temp = training_data_output_origin[i, j, 1]
        y_temp = training_data_output_origin[i, j, 2]
        training_data_output_origin[i, j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
        training_data_output_origin[i, j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)

for i in range(testing_data_input.shape[0]):
    theda = math.atan2(testing_data_input[i, -5, 2],
                       testing_data_input[i, -5, 1])  # 与x轴的夹角
    theda = 3 * math.pi / 2 - theda

    for j in range(testing_data_input.shape[1]):
        x_temp = testing_data_input[i, j, 1]
        y_temp = testing_data_input[i, j, 2]
        testing_data_input[i, j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
        testing_data_input[i, j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)

    for j in range(testing_data_output_origin.shape[1]):
        x_temp = testing_data_output_origin[i, j, 1]
        y_temp = testing_data_output_origin[i, j, 2]
        testing_data_output_origin[i, j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
        testing_data_output_origin[i, j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)

# 分类
# 6 7 8
# 3 4 5
# 0 1 2
training_data_output = np.zeros([training_data_output_origin.shape[0],
                                 1,
                                 row * column])
testing_data_output = np.zeros([testing_data_output_origin.shape[0],
                                1,
                                row * column])
for i in range(training_data_output.shape[0]):
    for j in range(training_data_output_origin.shape[1]):
        if abs(training_data_output_origin[i, j, 1]) <= (side_length_x_center / 2):
            # 在1，4，7内
            if training_data_output_origin[i, j, 2] <= side_length_y:
                # 在1内
                training_data_output[i, 0, 1] = training_data_output[i, 0, 1] + 1
            elif (training_data_output_origin[i, j, 2] > side_length_y) and \
                 (training_data_output_origin[i, j, 2] <= side_length_y * 2):
                # 在4内
                training_data_output[i, 0, 4] = training_data_output[i, 0, 4] + 1
            elif (training_data_output_origin[i, j, 2] > side_length_y * 2) and \
                 (training_data_output_origin[i, j, 2] <= side_length_y * 3):
                # 在7内
                training_data_output[i, 0, 7] = training_data_output[i, 0, 7] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}，y = {training_data_output_origin[i, j, 2]}]")
                raise IndexError
        elif (training_data_output_origin[i, j, 1] > side_length_x_center / 2) and \
             (training_data_output_origin[i, j, 1] <= (side_length_x_center + side_length_x)):
            # 在2，5，8内
            if training_data_output_origin[i, j, 2] <= side_length_y:
                # 在2内
                training_data_output[i, 0, 2] = training_data_output[i, 0, 2] + 1
            elif (training_data_output_origin[i, j, 2] > side_length_y) and \
                 (training_data_output_origin[i, j, 2] <= side_length_y * 2):
                # 在5内
                training_data_output[i, 0, 5] = training_data_output[i, 0, 5] + 1
            elif (training_data_output_origin[i, j, 2] > side_length_y * 2) and \
                 (training_data_output_origin[i, j, 2] <= side_length_y * 3):
                # 在8内
                training_data_output[i, 0, 8] = training_data_output[i, 0, 8] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {training_data_output_origin[i, j, 2]}")
                raise IndexError
        elif (training_data_output_origin[i, j, 1] > -(side_length_x_center + side_length_x)) and \
             (training_data_output_origin[i, j, 1] <= -(side_length_x_center / 2)):
            # 在0，3，6内
            if training_data_output_origin[i, j, 2] <= side_length_y:
                # 在0内
                training_data_output[i, 0, 0] = training_data_output[i, 0, 0] + 1
            elif (training_data_output_origin[i, j, 2] > side_length_y) and \
                 (training_data_output_origin[i, j, 2] <= side_length_y * 2):
                # 在3内
                training_data_output[i, 0, 3] = training_data_output[i, 0, 3] + 1
            elif (training_data_output_origin[i, j, 2] > side_length_y * 2) and \
                 (training_data_output_origin[i, j, 2] <= side_length_y * 3):
                # 在6内
                training_data_output[i, 0, 6] = training_data_output[i, 0, 6] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {training_data_output_origin[i, j, 2]}")
                raise IndexError
        else:
            print(f"超出范围，请增大范围，当前i={i}，x = {training_data_output_origin[i, j, 1]}")
            raise IndexError
    training_data_output[i, 0, :] = training_data_output[i, 0, :] / training_data_output[i, 0, :].sum()

for i in range(testing_data_output.shape[0]):
    for j in range(testing_data_output_origin.shape[1]):
        if abs(testing_data_output_origin[i, j, 1]) <= (side_length_x_center / 2):
            # 在1，4，7内
            if testing_data_output_origin[i, j, 2] <= side_length_y:
                # 在1内
                testing_data_output[i, 0, 1] = testing_data_output[i, 0, 1] + 1
            elif (testing_data_output_origin[i, j, 2] > side_length_y) and \
                 (testing_data_output_origin[i, j, 2] <= side_length_y * 2):
                # 在4内
                testing_data_output[i, 0, 4] = testing_data_output[i, 0, 4] + 1
            elif (testing_data_output_origin[i, j, 2] > side_length_y * 2) and \
                 (testing_data_output_origin[i, j, 2] <= side_length_y * 3):
                # 在7内
                testing_data_output[i, 0, 7] = testing_data_output[i, 0, 7] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}，y = {testing_data_output_origin[i, j, 2]}]")
                raise IndexError
        elif (testing_data_output_origin[i, j, 1] > side_length_x_center / 2) and \
             (testing_data_output_origin[i, j, 1] <= (side_length_x_center + side_length_x)):
            # 在2，5，8内
            if testing_data_output_origin[i, j, 2] <= side_length_y:
                # 在2内
                testing_data_output[i, 0, 2] = testing_data_output[i, 0, 2] + 1
            elif (testing_data_output_origin[i, j, 2] > side_length_y) and \
                 (testing_data_output_origin[i, j, 2] <= side_length_y * 2):
                # 在5内
                testing_data_output[i, 0, 5] = testing_data_output[i, 0, 5] + 1
            elif (testing_data_output_origin[i, j, 2] > side_length_y * 2) and \
                 (testing_data_output_origin[i, j, 2] <= side_length_y * 3):
                # 在8内
                testing_data_output[i, 0, 8] = testing_data_output[i, 0, 8] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {testing_data_output_origin[i, j, 2]}")
                raise IndexError
        elif (testing_data_output_origin[i, j, 1] > -(side_length_x_center + side_length_x)) and \
             (testing_data_output_origin[i, j, 1] <= -(side_length_x_center / 2)):
            # 在0，3，6内
            if testing_data_output_origin[i, j, 2] <= side_length_y:
                # 在0内
                testing_data_output[i, 0, 0] = testing_data_output[i, 0, 0] + 1
            elif (testing_data_output_origin[i, j, 2] > side_length_y) and \
                 (testing_data_output_origin[i, j, 2] <= side_length_y * 2):
                # 在3内
                testing_data_output[i, 0, 3] = testing_data_output[i, 0, 3] + 1
            elif (testing_data_output_origin[i, j, 2] > side_length_y * 2) and \
                 (testing_data_output_origin[i, j, 2] <= side_length_y * 3):
                # 在6内
                testing_data_output[i, 0, 6] = testing_data_output[i, 0, 6] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {testing_data_output_origin[i, j, 2]}")
                raise IndexError
        else:
            print(f"超出范围，请增大范围，当前i={i}，x = {testing_data_output_origin[i, j, 1]}")
            raise IndexError
    testing_data_output[i, 0, :] = testing_data_output[i, 0, :] / testing_data_output[i, 0, :].sum()


# plt.figure()
# for i in range(training_data_input.shape[0]):
#     plt.clf()
#     plt.plot([0.0, 0.0], [-1.0, 1.0], "r--")
#     lim = 5
#     plt.xlim(-lim, lim)
#     plt.ylim(-lim, lim)
#     plt.plot(training_data_input[i, :, 1], training_data_input[i, :, 2], ".")
#     plt.plot(training_data_output_origin[i, :, 1], training_data_output_origin[i, :, 2], ".")
#
#     lim = 8
#     plt.xlim(-lim, lim)
#     plt.ylim(0.0, row * side_length_y)
#     plt.plot([side_length_x_center / 2, side_length_x_center / 2], [0.0, side_length_y * 3], "r--")
#     plt.plot([-side_length_x_center / 2, -side_length_x_center / 2], [0.0, side_length_y * 3], "r--")
#     plt.plot([(side_length_x_center + side_length_x / 2), -(side_length_x_center + side_length_x / 2)],
#              [side_length_y, side_length_y], "r--")
#     plt.plot([(side_length_x_center + side_length_x / 2), -(side_length_x_center + side_length_x / 2)],
#              [side_length_y * 2, side_length_y * 2], "r--")
#     plt.text(-(side_length_x + side_length_x_center) / 2, side_length_y / 2,
#              training_data_output[i, 0, 0], fontsize=10)
#     plt.text(0.0, side_length_y / 2,
#              training_data_output[i, 0, 1], fontsize=10)
#     plt.text((side_length_x + side_length_x_center) / 2, side_length_y / 2,
#              training_data_output[i, 0, 2], fontsize=10)
#     plt.text(-(side_length_x + side_length_x_center) / 2, side_length_y + side_length_y / 2,
#              training_data_output[i, 0, 3], fontsize=10)
#     plt.text(0.0, side_length_y + side_length_y / 2,
#              training_data_output[i, 0, 4], fontsize=10)
#     plt.text((side_length_x + side_length_x_center) / 2, side_length_y + side_length_y / 2,
#              training_data_output[i, 0, 5], fontsize=10)
#     plt.text(-(side_length_x + side_length_x_center) / 2, 2 * side_length_y + side_length_y / 2,
#              training_data_output[i, 0, 6], fontsize=10)
#     plt.text(0.0, 2 * side_length_y + side_length_y / 2,
#              training_data_output[i, 0, 7], fontsize=10)
#     plt.text((side_length_x + side_length_x_center) / 2, 2 * side_length_y + side_length_y / 2,
#              training_data_output[i, 0, 8], fontsize=10)
#
#     # plt.plot([training_data_input[i, -1, 1], training_data_input[i, -2, 1]],
#     #          [training_data_input[i, -1, 2], training_data_input[i, -2, 2]],
#     #          "r--")
#     # plt.plot(training_data_input[i, -2:training_data_input.shape[1], 1],
#     #          training_data_input[i, -2:training_data_input.shape[1], 2], ".")
#     # if i == 100:
#     #     print()
#     # theda = math.atan2(training_data_input[i, -5, 2],
#     #                    training_data_input[i, -5, 1])
#     # print(theda * 180 / math.pi)
#     plt.pause(0.001)


# # 移除index参数
training_data_input = np.array(training_data_input[:, :, 1:data_size])
# training_data_output = np.array(training_data_output_origin[:, :, 1:3])
testing_data_input = np.array(testing_data_input[:, :, 1:data_size])
# testing_data_output = np.array(testing_data_output_origin[:, :, 1:3])
