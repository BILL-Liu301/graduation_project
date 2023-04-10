import math
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

seq_size = 50
jump_size = int(seq_size / 4)
split_size = 8
data_size = 3

# # stamp x y Vx Vy
# data = np.load("source.npy")[:, 0:data_size]
#
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

# 设置边界格，行三列，1m*1m
side_length_x = 2
side_length_x_center = 1.5
side_length_y = 4
row = 5
column = 5

# 分割input和output
input_size = int(0.5 * seq_size)
output_size = seq_size - input_size
train_size = int(0.5 * data_reshape.shape[0])
test_size = data_reshape.shape[0] - train_size

training_data_input = np.array(data_reshape[0:train_size, 0:input_size, :])
training_data_output_origin = np.array(data_reshape[0:train_size,
                                       input_size:data_reshape.shape[1], :])

testing_data_input = np.array(data_reshape[train_size:data_reshape.shape[0], 0:input_size, :])
testing_data_output_origin = np.array(data_reshape[train_size:data_reshape.shape[0],
                                      input_size:data_reshape.shape[1], :])

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
# 20 21 22 23 24
# 15 16 17 18 19
# 10 11 12 13 14
# 5  6  7  8  9
# 0  1  2  3  4
training_data_output = np.zeros([training_data_output_origin.shape[0],
                                 1,
                                 row * column])
testing_data_output = np.zeros([testing_data_output_origin.shape[0],
                                1,
                                row * column])
reshape = training_data_output
reshape_origin = training_data_output_origin
for i in range(reshape.shape[0]):
    for j in range(int(reshape_origin.shape[1] - 1), reshape_origin.shape[1]):
        if abs(reshape_origin[i, j, 1]) <= (side_length_x_center / 2):
            # 在2， 7， 12， 17， 22内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在2内
                reshape[i, 0, 2] = reshape[i, 0, 2] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在7内
                reshape[i, 0, 7] = reshape[i, 0, 7] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在12内
                reshape[i, 0, 12] = reshape[i, 0, 12] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在17内
                reshape[i, 0, 17] = reshape[i, 0, 17] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在22内
                reshape[i, 0, 22] = reshape[i, 0, 22] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}，y = {reshape_origin[i, j, 2]}]")
                raise IndexError
        elif (reshape_origin[i, j, 1] > side_length_x_center / 2) and \
             (reshape_origin[i, j, 1] <= (side_length_x_center / 2 + side_length_x)):
            # 在3，8，13，18，23内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在3内
                reshape[i, 0, 3] = reshape[i, 0, 3] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在8内
                reshape[i, 0, 8] = reshape[i, 0, 8] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在13内
                reshape[i, 0, 13] = reshape[i, 0, 23] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在18内
                reshape[i, 0, 18] = reshape[i, 0, 18] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在23内
                reshape[i, 0, 23] = reshape[i, 0, 23] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {reshape_origin[i, j, 2]}")
                raise IndexError
        elif (reshape_origin[i, j, 1] > (side_length_x_center / 2 + side_length_x)) and \
             (reshape_origin[i, j, 1] <= (side_length_x_center / 2 + side_length_x * 2)):
            # 在4，9，14，19，24内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在4内
                reshape[i, 0, 4] = reshape[i, 0, 4] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在9内
                reshape[i, 0, 9] = reshape[i, 0, 9] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在14内
                reshape[i, 0, 14] = reshape[i, 0, 14] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在19内
                reshape[i, 0, 19] = reshape[i, 0, 19] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在24内
                reshape[i, 0, 24] = reshape[i, 0, 24] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {reshape_origin[i, j, 2]}")
                raise IndexError
        elif (reshape_origin[i, j, 1] > -(side_length_x_center / 2 + side_length_x)) and \
             (reshape_origin[i, j, 1] <= -(side_length_x_center / 2)):
            # 在1，6，11，16，21内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在1内
                reshape[i, 0, 1] = reshape[i, 0, 1] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在6内
                reshape[i, 0, 6] = reshape[i, 0, 6] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在11内
                reshape[i, 0, 11] = reshape[i, 0, 11] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在16内
                reshape[i, 0, 16] = reshape[i, 0, 16] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在21内
                reshape[i, 0, 21] = reshape[i, 0, 21] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {reshape_origin[i, j, 2]}")
                raise IndexError
        elif (reshape_origin[i, j, 1] > -(side_length_x_center / 2 + side_length_x * 2)) and \
             (reshape_origin[i, j, 1] <= -(side_length_x_center / 2 + side_length_x)):
            # 在0，5，10，15，20内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在0内
                reshape[i, 0, 0] = reshape[i, 0, 0] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在5内
                reshape[i, 0, 5] = reshape[i, 0, 5] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在10内
                reshape[i, 0, 10] = reshape[i, 0, 10] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在15内
                reshape[i, 0, 15] = reshape[i, 0, 15] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在20内
                reshape[i, 0, 20] = reshape[i, 0, 20] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {reshape_origin[i, j, 2]}")
                raise IndexError
        else:
            print(f"超出范围，请增大范围，当前i={i}，x = {reshape_origin[i, j, 1]}")
            raise IndexError
    reshape[i, 0, :] = reshape[i, 0, :] / reshape[i, 0, :].sum()
    # reshape[i, 0, :] = np.exp(reshape[i, 0, :]) / np.exp(reshape[i, 0, :]).sum()
    training_data_output = np.array(reshape)

reshape = testing_data_output
reshape_origin = testing_data_output_origin
for i in range(reshape.shape[0]):
    for j in range(int(reshape_origin.shape[1] - 1), reshape_origin.shape[1]):
        if abs(reshape_origin[i, j, 1]) <= (side_length_x_center / 2):
            # 在2， 7， 12， 17， 22内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在2内
                reshape[i, 0, 2] = reshape[i, 0, 2] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在7内
                reshape[i, 0, 7] = reshape[i, 0, 7] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在12内
                reshape[i, 0, 12] = reshape[i, 0, 12] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在17内
                reshape[i, 0, 17] = reshape[i, 0, 17] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在22内
                reshape[i, 0, 22] = reshape[i, 0, 22] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}，y = {reshape_origin[i, j, 2]}]")
                raise IndexError
        elif (reshape_origin[i, j, 1] > side_length_x_center / 2) and \
             (reshape_origin[i, j, 1] <= (side_length_x_center / 2 + side_length_x)):
            # 在3，8，13，18，23内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在3内
                reshape[i, 0, 3] = reshape[i, 0, 3] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在8内
                reshape[i, 0, 8] = reshape[i, 0, 8] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在13内
                reshape[i, 0, 13] = reshape[i, 0, 23] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在18内
                reshape[i, 0, 18] = reshape[i, 0, 18] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在23内
                reshape[i, 0, 23] = reshape[i, 0, 23] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {reshape_origin[i, j, 2]}")
                raise IndexError
        elif (reshape_origin[i, j, 1] > (side_length_x_center / 2 + side_length_x)) and \
             (reshape_origin[i, j, 1] <= (side_length_x_center / 2 + side_length_x * 2)):
            # 在4，9，14，19，24内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在4内
                reshape[i, 0, 4] = reshape[i, 0, 4] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在9内
                reshape[i, 0, 9] = reshape[i, 0, 9] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在14内
                reshape[i, 0, 14] = reshape[i, 0, 14] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在19内
                reshape[i, 0, 19] = reshape[i, 0, 19] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                 (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在24内
                reshape[i, 0, 24] = reshape[i, 0, 24] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {reshape_origin[i, j, 2]}")
                raise IndexError
        elif (reshape_origin[i, j, 1] > -(side_length_x_center / 2 + side_length_x)) and \
             (reshape_origin[i, j, 1] <= -(side_length_x_center / 2)):
            # 在1，6，11，16，21内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在1内
                reshape[i, 0, 1] = reshape[i, 0, 1] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在6内
                reshape[i, 0, 6] = reshape[i, 0, 6] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在11内
                reshape[i, 0, 11] = reshape[i, 0, 11] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在16内
                reshape[i, 0, 16] = reshape[i, 0, 16] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在21内
                reshape[i, 0, 21] = reshape[i, 0, 21] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {reshape_origin[i, j, 2]}")
                raise IndexError
        elif (reshape_origin[i, j, 1] > -(side_length_x_center / 2 + side_length_x * 2)) and \
             (reshape_origin[i, j, 1] <= -(side_length_x_center / 2 + side_length_x)):
            # 在0，5，10，15，20内
            if reshape_origin[i, j, 2] <= side_length_y:
                # 在0内
                reshape[i, 0, 0] = reshape[i, 0, 0] + 1
            elif (reshape_origin[i, j, 2] > side_length_y) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 2):
                # 在5内
                reshape[i, 0, 5] = reshape[i, 0, 5] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 2) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 3):
                # 在10内
                reshape[i, 0, 10] = reshape[i, 0, 10] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 3) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 4):
                # 在15内
                reshape[i, 0, 15] = reshape[i, 0, 15] + 1
            elif (reshape_origin[i, j, 2] > side_length_y * 4) and \
                    (reshape_origin[i, j, 2] <= side_length_y * 5):
                # 在20内
                reshape[i, 0, 20] = reshape[i, 0, 20] + 1
            else:
                print(f"超出范围，请增大范围，当前[{i}, {j}]，y = {reshape_origin[i, j, 2]}")
                raise IndexError
        else:
            print(f"超出范围，请增大范围，当前i={i}，x = {reshape_origin[i, j, 1]}")
            raise IndexError
    reshape[i, 0, :] = reshape[i, 0, :] / reshape[i, 0, :].sum()
    # reshape[i, 0, :] = np.exp(reshape[i, 0, :]) / np.exp(reshape[i, 0, :]).sum()
    testing_data_output = np.array(reshape)

# plt.figure()
# for i in range(training_data_input.shape[0]):
#     plt.clf()
#     plt.plot([0.0, 0.0], [-1.0, 1.0], "r--")
#     plt.plot(training_data_input[i, :, 1], training_data_input[i, :, 2], ".")
#     plt.plot(training_data_output_origin[i, :, 1], training_data_output_origin[i, :, 2], ".")
#
#     lim = 10
#     plt.xlim(-(int(column / 2) * side_length_x + side_length_x_center / 2),
#              (int(column / 2) * side_length_x + side_length_x_center / 2))
#     plt.ylim(0.0, row * side_length_y)
#
#     plt.plot([side_length_x_center / 2, side_length_x_center / 2], [0.0, side_length_y * row], "r--")
#     plt.plot([-side_length_x_center / 2, -side_length_x_center / 2], [0.0, side_length_y * row], "r--")
#     plt.plot([-(side_length_x_center / 2 + side_length_x), -(side_length_x_center / 2 + side_length_x)],
#              [0.0, side_length_y * row], "r--")
#     plt.plot([(side_length_x_center / 2 + side_length_x), (side_length_x_center / 2 + side_length_x)],
#              [0.0, side_length_y * row], "r--")
#
#     plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
#              [side_length_y, side_length_y], "r--")
#     plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
#              [side_length_y * 2, side_length_y * 2], "r--")
#     plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
#              [side_length_y * 3, side_length_y * 3], "r--")
#     plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
#              [side_length_y * 4, side_length_y * 4], "r--")
#
#     plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2,
#              training_data_output[i, 0, 0], fontsize=10)
#     plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2,
#              training_data_output[i, 0, 1], fontsize=10)
#     plt.text(0.0, side_length_y / 2,
#              training_data_output[i, 0, 2], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2,
#              training_data_output[i, 0, 3], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2,
#              training_data_output[i, 0, 4], fontsize=10)
#
#     plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y,
#              training_data_output[i, 0, 5], fontsize=10)
#     plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y,
#              training_data_output[i, 0, 6], fontsize=10)
#     plt.text(0.0, side_length_y / 2 + side_length_y,
#              training_data_output[i, 0, 7], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y,
#              training_data_output[i, 0, 8], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y,
#              training_data_output[i, 0, 9], fontsize=10)
#
#     plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
#              training_data_output[i, 0, 10], fontsize=10)
#     plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
#              training_data_output[i, 0, 11], fontsize=10)
#     plt.text(0.0, side_length_y / 2 + side_length_y * 2,
#              training_data_output[i, 0, 12], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
#              training_data_output[i, 0, 13], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
#              training_data_output[i, 0, 14], fontsize=10)
#
#     plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
#              training_data_output[i, 0, 15], fontsize=10)
#     plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
#              training_data_output[i, 0, 16], fontsize=10)
#     plt.text(0.0, side_length_y / 2 + side_length_y * 3,
#              training_data_output[i, 0, 17], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
#              training_data_output[i, 0, 18], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
#              training_data_output[i, 0, 19], fontsize=10)
#
#     plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
#              training_data_output[i, 0, 20], fontsize=10)
#     plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
#              training_data_output[i, 0, 21], fontsize=10)
#     plt.text(0.0, side_length_y / 2 + side_length_y * 4,
#              training_data_output[i, 0, 22], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
#              training_data_output[i, 0, 23], fontsize=10)
#     plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
#              training_data_output[i, 0, 24], fontsize=10)
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


# 移除index参数
training_data_input = np.array(training_data_input[:, :, 1:data_size])
# training_data_output = np.array(training_data_output_origin[:, :, 1:3])
testing_data_input = np.array(testing_data_input[:, :, 1:data_size])
# testing_data_output = np.array(testing_data_output_origin[:, :, 1:3])
