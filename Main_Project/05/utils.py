import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# def reshape_data(origin, batch_size, input_size, seq):
#     if len(origin) % (batch_size * input_size * seq) != 0.0:
#         print("Reshape Filed")
#         sys.exit(0)
#     len_reshaped = len(origin) - (input_feature_num + output_feature_num) + 1
#     reshaped = np.zeros((len_reshaped, rows, input_feature_num))
#     for each_block in range(len_reshaped):
#         for each_row in range(rows):
#             for each_feature_num in range(input_feature_num):
#                 reshaped[each_block, each_row, each_feature_num] = origin[each_block + each_feature_num]
#     return reshaped


# stamp、acc、position、velocity
file_name = 'source.txt'
file_data = open(file_name, 'r')
file_data_lines = file_data.readlines()
data = np.zeros((len(file_data_lines), 4))
temp = 0
for row in file_data_lines:
    find_data = row.split(' ')
    data[temp, 0] = float(find_data[0])
    data[temp, 1] = float(find_data[1])
    data[temp, 2] = float(find_data[2])
    data[temp, 3] = float(find_data[3])
    temp += 1
data[:, 0] = data[:, 0] - data[0, 0]

training_ratio = 0.5
training_len = int(len(data) * training_ratio)

training_data = np.zeros((training_len, 2))
training_data[:, 0] = data[0:training_len, 0]
training_data[:, 1] = data[0:training_len, 2]

training_data_reshaped = training_data[:, 1].reshape(-1, 10)
training_data_t = training_data[:, 0].reshape(-1, 10)

training_data_input = np.zeros((training_data_reshaped.shape[0],
                                training_data_reshaped.shape[1] - 1))
training_data_output = np.zeros((training_data_reshaped.shape[0], 1))
for i in range(len(training_data_input)):
    for j in range(len(training_data_input[i])):
        training_data_input[i, j] = training_data_reshaped[i, j]
    training_data_output[i] = training_data_reshaped[i, -1]
for i in range(len(training_data_input[0]) - 1):
    training_data_input[:, i + 1] = training_data_input[:, i + 1] - training_data_input[:, 0]
np.savetxt('training_data_input.txt', training_data_input, fmt='%f')
np.savetxt('training_data_output.txt', training_data_output, fmt='%f')


testing_len = int(len(data) * (1 - training_ratio))
testing_data = np.zeros((testing_len, 2))
testing_data[:, 0] = data[testing_len:len(data), 0]
testing_data[:, 1] = data[testing_len:len(data), 2]

testing_data_reshaped = testing_data[:, 1].reshape(-1, 10)
testing_data_t = testing_data[:, 0].reshape(-1, 10)

testing_data_input = np.zeros((testing_data_reshaped.shape[0],
                               testing_data_reshaped.shape[1] - 1))
testing_data_output = np.zeros((testing_data_reshaped.shape[0], 1))
for i in range(len(testing_data_input)):
    for j in range(len(testing_data_input[i])):
        testing_data_input[i, j] = testing_data_reshaped[i, j]
    testing_data_output[i] = testing_data_reshaped[i, -1]
for i in range(len(testing_data_input[0]) - 1):
    testing_data_input[:, i + 1] = testing_data_input[:, i + 1] - testing_data_input[:, 0]
np.savetxt('testing_data_input.txt', testing_data_input, fmt='%f')
np.savetxt('testing_data_output.txt', testing_data_output, fmt='%f')


