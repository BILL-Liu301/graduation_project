import math
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

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
jump_size = int(seq_size / 2)
split_size = 10
data_size = 5
# # 滑窗重组
# data_reshape = np.array([[0, 0.0, 0.0, 0.0, 0.0]])
# for i in range(0, data.shape[0] - jump_size - seq_size*split_size, jump_size):
#     for j in range(0, seq_size*split_size, split_size):
#         data_reshape[-1, :] = np.array(data[i+j, :])
#         data_reshape = np.r_[data_reshape, np.array([[0, 0.0, 0.0, 0.0, 0.0]])]
# np.save("data_reshape.npy", data_reshape)

# 加载滑窗重组的结果
data_reshape = np.load("data_reshape.npy")
data_reshape = np.delete(data_reshape, [-1], 0)
data_reshape = np.array(data_reshape.reshape(-1, seq_size, data_size))

# 分割input和output
input_size = int(0.8 * seq_size)
output_size = seq_size - input_size
train_size = int(0.5 * data_reshape.shape[0])
test_size = data_reshape.shape[0] - train_size

training_data_input = np.array(data_reshape[0:train_size, 0:input_size, :])
# training_data_output = np.array(data_reshape[0:train_size, (input_size + 1):(input_size + 2), :])
training_data_output = np.array(data_reshape[0:train_size, input_size:data_reshape.shape[1], :])

testing_data_input = np.array(data_reshape[train_size:data_reshape.shape[0], 0:input_size, :])
# testing_data_output = np.array(data_reshape[train_size:data_reshape.shape[0], (input_size + 1):(input_size + 2), :])
testing_data_output = np.array(data_reshape[train_size:data_reshape.shape[0], input_size:data_reshape.shape[1], :])

# 采用相对坐标，以训练集最后一个点为原点
for i in range(training_data_input.shape[0]):
    for j in range(training_data_output.shape[1]):
        training_data_output[i, j, 1] = training_data_output[i, j, 1] - training_data_input[i, -1, 1]
        training_data_output[i, j, 2] = training_data_output[i, j, 2] - training_data_input[i, -1, 2]
    for j in range(training_data_input.shape[1]):
        training_data_input[i, j, 1] = training_data_input[i, j, 1] - training_data_input[i, -1, 1]
        training_data_input[i, j, 2] = training_data_input[i, j, 2] - training_data_input[i, -1, 2]

for i in range(testing_data_input.shape[0]):
    for j in range(testing_data_output.shape[1]):
        testing_data_output[i, j, 1] = testing_data_output[i, j, 1] - testing_data_input[i, -1, 1]
        testing_data_output[i, j, 2] = testing_data_output[i, j, 2] - testing_data_input[i, -1, 2]
    for j in range(testing_data_input.shape[1]):
        testing_data_input[i, j, 1] = testing_data_input[i, j, 1] - testing_data_input[i, -1, 1]
        testing_data_input[i, j, 2] = testing_data_input[i, j, 2] - testing_data_input[i, -1, 2]

# plt.figure()
# for i in range(training_data_input.shape[0]):
#     plt.clf()
#     plt.plot(training_data_input[i, :, 1], training_data_input[i, :, 2], ".")
#     plt.plot([training_data_input[i, 0, 1], training_data_input[i, -1, 2]],
#              [training_data_input[i, -1, 2], training_data_input[i, -1, 2]], "r--")
#     plt.plot(training_data_output[i, :, 1], training_data_output[i, :, 2], ".")
#     plt.pause(0.01)

# 移除index参数
training_data_input = np.array(training_data_input[:, :, 1:5])
training_data_output = np.array(training_data_output[:, :, 1:3])
testing_data_input = np.array(testing_data_input[:, :, 1:5])
testing_data_output = np.array(testing_data_output[:, :, 1:3])

