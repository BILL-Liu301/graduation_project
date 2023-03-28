import math
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

Qw = 15
Ql = 25
start_x = 20
start_y = 10
grid_w = 0.1  # y
grid_l = 0.1  # x
print(f"Qw * Ql = {Qw * grid_w * Ql * grid_l}m^2")

# stamp x y Vx Vy
file_name = 'source.txt'
file_data = open(file_name, 'r')
file_data_lines = file_data.readlines()
data = np.zeros((len(file_data_lines), 5))
temp = 0
for row in file_data_lines:
    find_data = row.split(' ')
    data[temp, 0] = float(find_data[0])
    data[temp, 1] = float(find_data[1])
    data[temp, 2] = float(find_data[2])
    data[temp, 3] = float(find_data[3])
    data[temp, 4] = float(find_data[4])
    temp += 1

seq_size = 50
hidden_length = 5
jump_size = int(seq_size / 2)
split_size = 1
# 滑窗重组
# data_reshape = np.array([[0, 0.0, 0.0, 0.0, 0.0]])
# for i in range(0, data.shape[0] - jump_size, jump_size):
#     for j in range(0, seq_size*split_size, split_size):
#         data_reshape[-1, :] = np.array(data[i+j, :])
#         data_reshape = np.r_[data_reshape, np.array([[0, 0.0, 0.0, 0.0, 0.0]])]
# np.save("data_reshape.npy", data_reshape)

# 加载滑窗重组的结果
data_reshape = np.load("data_reshape.npy")
data_reshape = np.delete(data_reshape, [-1], 0)
data_reshape = np.array(data_reshape.reshape(-1, seq_size, 5))
dis = np.zeros(data_reshape.shape[0])
print(f"data_reshape:{data_reshape.shape}")

# plt.figure()
# for i in range(dis.shape[0]):
#     dis[i] = math.sqrt(
#         (data_reshape[i, 0, 1] - data_reshape[i, -1, 1]) ** 2 + (data_reshape[i, 0, 2] - data_reshape[i, -1, 2]) ** 2)
# plt.clf()
# plt.plot(data_reshape[i, :, 1], data_reshape[i, :, 2], "*", label='trajectory')
# # plt.plot(data_reshape[i, :, 1], data_reshape[i, :, 3], label='Vx')
# # plt.plot(data_reshape[i, :, 1], data_reshape[i, :, 4], label='Vy')
# plt.legend(loc='upper right')
# plt.pause(0.001)
# print(f"{dis[i]:.4f}")
# print(dis.max())

rate = 0.5
training_data_input = np.array(data_reshape[0:int(data_reshape.shape[0] * rate), :, :])
testing_data_input = np.array(
    data_reshape[int(data_reshape.shape[0] * rate):(data_reshape.shape[0] - data_reshape.shape[1]), :, :])
# training_data_output = np.zeros([training_data_input.shape[0], training_data_input.shape[1], 3])
# testing_data_output = np.zeros([testing_data_input.shape[0], testing_data_input.shape[1], 3])
training_data_output = np.zeros([training_data_input.shape[0], 1, Ql * Qw])
testing_data_output = np.zeros([testing_data_input.shape[0], 1, Ql * Qw])

for i in range(training_data_output.shape[0]):
    for j in range(training_data_output.shape[1]):
        training_data_output[i, j, 0] = training_data_input[i, -1, 0] + split_size + j * split_size
        training_data_output[i, j, 1] = data[int(training_data_output[i, j, 0]) - 1, 1] - training_data_input[i, 0, 1]
        training_data_output[i, j, 2] = data[int(training_data_output[i, j, 0]) - 1, 2] - training_data_input[i, 0, 2]
    training_data_input[i, :, 1] = training_data_input[i, :, 1] - training_data_input[i, 0, 1]
    training_data_input[i, :, 2] = training_data_input[i, :, 2] - training_data_input[i, 0, 2]

for i in range(testing_data_output.shape[0]):
    for j in range(testing_data_output.shape[1]):
        testing_data_output[i, j, 0] = testing_data_input[i, -1, 0] + split_size + j * split_size
        testing_data_output[i, j, 1] = data[int(testing_data_output[i, j, 0] - 1), 1] - testing_data_input[i, 0, 1]
        testing_data_output[i, j, 2] = data[int(testing_data_output[i, j, 0] - 1), 2] - testing_data_input[i, 0, 2]
    testing_data_input[i, :, 1] = testing_data_input[i, :, 1] - testing_data_input[i, 0, 1]
    testing_data_input[i, :, 2] = testing_data_input[i, :, 2] - testing_data_input[i, 0, 2]
# training_data_output = training_data_output[:, :, 1:3]
# testing_data_output = testing_data_output[:, :, 1:3]

# 栅格化,观测车辆在最初点的左下（5m，5m）的位置，车辆原点在Qw*Ql这个矩形的最下边中心
training_data_input[:, :, 1] = training_data_input[:, :, 1] // grid_l + start_x
training_data_input[:, :, 2] = training_data_input[:, :, 2] // grid_w + start_y
# training_data_output[:, :, 0] = training_data_output[:, :, 0] // grid_l + start_x
# training_data_output[:, :, 1] = training_data_output[:, :, 1] // grid_w + start_y
training_data_output[:, :, 1] = training_data_output[:, :, 1] // grid_l + start_x
training_data_output[:, :, 2] = training_data_output[:, :, 2] // grid_w + start_y

testing_data_input[:, :, 1] = testing_data_input[:, :, 1] // grid_l + start_x
testing_data_input[:, :, 2] = testing_data_input[:, :, 2] // grid_w + start_y
# testing_data_output[:, :, 0] = testing_data_output[:, :, 0] // grid_l + start_x
# testing_data_output[:, :, 1] = testing_data_output[:, :, 1] // grid_w + start_y
testing_data_output[:, :, 0] = testing_data_output[:, :, 1] // grid_l + start_x
testing_data_output[:, :, 1] = testing_data_output[:, :, 2] // grid_w + start_y

# print(training_data_input[:, :, 1].min(), training_data_input[:, :, 2].min())
# print(training_data_output[:, :, 0].min(), training_data_output[:, :, 1].min())
# print(testing_data_input[:, :, 1].min(), testing_data_input[:, :, 2].min())
# print(testing_data_output[:, :, 0].min(), testing_data_output[:, :, 1].min())
#
# print(training_data_input[:, :, 1].max(), training_data_input[:, :, 2].max())
# print(training_data_output[:, :, 0].max(), training_data_output[:, :, 1].max())
# print(testing_data_input[:, :, 1].max(), testing_data_input[:, :, 2].max())
# print(testing_data_output[:, :, 0].max(), testing_data_output[:, :, 1].max())


# plt.figure()
# for i in range(testing_data_input.shape[0]):
#     plt.clf()
#     plt.plot(testing_data_input[i, :, 1], testing_data_input[i, :, 2], "*")
#     plt.plot(testing_data_output[i, :, 0, 0], testing_data_output[i, :, 0, 1], ".")
#     plt.pause(0.001)

for i in range(training_data_output.shape[0]):
    for j in range(training_data_output.shape[1]):
        l = training_data_output[i, j, 1]
        w = training_data_output[i, j, 2]
        training_data_output[i, j, 0:3] = 0
        flag = int((w - 1) * Qw + l)
        training_data_output[i, j, flag] = 1

for i in range(testing_data_output.shape[0]):
    for j in range(testing_data_output.shape[1]):
        l = testing_data_output[i, j, 1]
        w = testing_data_output[i, j, 2]
        testing_data_output[i, j, 0:3] = 0
        flag = int((w - 1) * Qw + l)

        testing_data_output[i, j, flag] = 1

training_data_output = np.squeeze(training_data_output, axis=1)
testing_data_output = np.squeeze(testing_data_output, axis=1)


def criterion(predicted, actual):
    middle = torch.zeros([predicted.shape[0], predicted.shape[1], 1, predicted.shape[3]]).to(torch.device('cuda:0'))
    for i in range(predicted.shape[2]):
        middle = middle + predicted[:, :, 0:1, :]
    middle = middle / torch.tensor(4.0)
    f = nn.MSELoss()
    loss = f(middle, actual)
    return loss


def find_w(index):
    return (index / Ql).floor()


def find_l(index):
    return index % Ql
