import math
import numpy as np
import os
import matplotlib.pyplot as plt

white_line = np.load("white_line.npy")
lane = np.load("lane.npy")

white_line[:, 2] = -1.0 * white_line[:, 2]
lane[:, 2] = -1.0 * lane[:, 2]
lane_no = np.array([1, 1, 1, 2, 2])

data_size = 5
seq_size = 50
jump_size = int(seq_size / 5)
split_size = 1

row = 15
column = 15  # 必须为奇数
size_row = 1  # 行高
size_colum = 1  # 列宽

# # time x y Vx Vy heading
# # path = "./npys/"
# path = "./00/"
# files = sorted(os.listdir(path), reverse=False)
# # time x y Vx Vy heading lane_no
# data_reshape = np.array([[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]])
# for index, file in enumerate(files):
#     data = np.load(path + str(file))
#     print(path + str(file), data.shape)
#     data_reshape_temp = np.array([[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]])
#     for i in range(0, data.shape[0] - jump_size - seq_size*split_size, jump_size):
#         for j in range(0, seq_size*split_size, split_size):
#             # data_reshape[-1, :] = np.append(np.array(data[i+j, :]), lane_no[index])
#             data_reshape_temp[-1, :] = np.append(np.array(data[i + j, :]), index)
#             data_reshape_temp = np.r_[data_reshape_temp, np.array([[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]])]
#
#     data_reshape_temp = np.delete(data_reshape_temp, [-1], 0)
#     data_reshape = np.r_[data_reshape, data_reshape_temp]
#     print(index, str(file), data_reshape.shape)
#     print("--------")
# np.save("data_reshape.npy", data_reshape)

data_reshape = np.load("data_reshape.npy")
data_reshape = np.delete(data_reshape, [0], 0)
data_reshape = np.array(data_reshape.reshape(-1, seq_size, 7))
print(data_reshape.shape)

input_size = int(0.5 * seq_size)
output_size = seq_size - input_size
train_size = int(0.1 * data_reshape.shape[0])
test_size = data_reshape.shape[0] - train_size

training_data_input_xy = np.array(data_reshape[0:train_size, 0:input_size, :])
# training_data_output = np.array(data_reshape[0:train_size, (input_size + 1):(input_size + 2), :])
training_data_output = np.array(data_reshape[0:train_size, input_size:data_reshape.shape[1], :])

testing_data_input_xy = np.array(data_reshape[train_size:data_reshape.shape[0], 0:input_size, :])
# testing_data_output = np.array(data_reshape[train_size:data_reshape.shape[0], (input_size + 1):(input_size + 2), :])
testing_data_output = np.array(
    data_reshape[train_size:data_reshape.shape[0], input_size:data_reshape.shape[1], :])

# 采用相对坐标，以训练集最后一个点为原点
for i in range(training_data_input_xy.shape[0]):
    for j in range(training_data_output.shape[1]):
        training_data_output[i, j, 1] = training_data_output[i, j, 1] - training_data_input_xy[i, -1, 1]
        training_data_output[i, j, 2] = training_data_output[i, j, 2] - training_data_input_xy[i, -1, 2]
    for j in range(training_data_input_xy.shape[1]):
        training_data_input_xy[i, j, 1] = training_data_input_xy[i, j, 1] - training_data_input_xy[i, -1, 1]
        training_data_input_xy[i, j, 2] = training_data_input_xy[i, j, 2] - training_data_input_xy[i, -1, 2]

for i in range(testing_data_input_xy.shape[0]):
    for j in range(testing_data_output.shape[1]):
        testing_data_output[i, j, 1] = testing_data_output[i, j, 1] - testing_data_input_xy[i, -1, 1]
        testing_data_output[i, j, 2] = testing_data_output[i, j, 2] - testing_data_input_xy[i, -1, 2]
    for j in range(testing_data_input_xy.shape[1]):
        testing_data_input_xy[i, j, 1] = testing_data_input_xy[i, j, 1] - testing_data_input_xy[i, -1, 1]
        testing_data_input_xy[i, j, 2] = testing_data_input_xy[i, j, 2] - testing_data_input_xy[i, -1, 2]

# 旋转数据组
for i in range(training_data_input_xy.shape[0]):
    theda = math.atan2(training_data_input_xy[i, -2, 2],
                       training_data_input_xy[i, -2, 1])  # 与x轴的夹角
    theda = 3 * math.pi / 2 - theda
    # theda = training_data_input_xy[i, -1, 5] * math.pi / 180
    training_data_input_xy[i, -1, 5] = -theda

    for j in range(training_data_input_xy.shape[1]):
        x_temp = training_data_input_xy[i, j, 1]
        y_temp = training_data_input_xy[i, j, 2]
        training_data_input_xy[i, j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
        training_data_input_xy[i, j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)

    for j in range(training_data_output.shape[1]):
        x_temp = training_data_output[i, j, 1]
        y_temp = training_data_output[i, j, 2]
        training_data_output[i, j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
        training_data_output[i, j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)

for i in range(testing_data_input_xy.shape[0]):
    theda = math.atan2(testing_data_input_xy[i, -2, 2],
                       testing_data_input_xy[i, -2, 1])  # 与x轴的夹角
    theda = 3 * math.pi / 2 - theda
    # theda = testing_data_input_xy[i, -1, 5] * math.pi / 180
    testing_data_input_xy[i, -1, 5] = -theda

    for j in range(testing_data_input_xy.shape[1]):
        x_temp = testing_data_input_xy[i, j, 1]
        y_temp = testing_data_input_xy[i, j, 2]
        testing_data_input_xy[i, j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
        testing_data_input_xy[i, j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)

    for j in range(testing_data_output.shape[1]):
        x_temp = testing_data_output[i, j, 1]
        y_temp = testing_data_output[i, j, 2]
        testing_data_output[i, j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
        testing_data_output[i, j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)

# plt.figure()
# for i in range(training_data_input_xy.shape[0]):
#     plt.xlim(-5, 5)
#     plt.ylim(-5, 5)
#     plt.plot(training_data_input_xy[i, :, 1], training_data_input_xy[i, :, 2], "*", color="r")
#     plt.plot(training_data_output[i, :, 1], training_data_output[i, :, 2], "*", color="b")
#     plt.pause(0.01)
#     plt.clf()

# 创建关于white_line和lane的输入
training_data_input_white_line = np.zeros([training_data_input_xy.shape[0],
                                           1,
                                           int(row * column)])
training_data_input_lane = np.zeros([training_data_input_xy.shape[0],
                                     1,
                                     int(row * column)])
testing_data_input_white_line = np.zeros([testing_data_input_xy.shape[0],
                                          1,
                                          int(row * column)])
testing_data_input_lane = np.zeros([testing_data_input_xy.shape[0],
                                    1,
                                    int(row * column)])

# 索引表，x_min,x_max,y_min,y_max
index_box = np.zeros([row, column, 4])
for i in range(row):
    for j in range(column):
        location = j - (column - 1) / 2
        if location == 0.0:
            index_box[i, j, 0] = abs(location) * size_colum - size_colum / 2
            index_box[i, j, 1] = abs(location) * size_colum + size_colum / 2
        elif location < 0.0:
            index_box[i, j, 0] = -(abs(location) * size_colum + size_colum / 2)
            index_box[i, j, 1] = -(abs(location) * size_colum - size_colum / 2)
        elif location > 0.0:
            index_box[i, j, 0] = (abs(location) * size_colum - size_colum / 2)
            index_box[i, j, 1] = (abs(location) * size_colum + size_colum / 2)
        index_box[i, j, 2] = i * size_row
        index_box[i, j, 3] = (i + 1) * size_row


# 查找宫格范围
def find_index(x, y):
    for r in range(row):
        if index_box[r, 0, 2] <= y <= index_box[r, 0, 3]:
            for c in range(column):
                if index_box[0, c, 0] <= x <= index_box[0, c, 1]:
                    return True, r, c
            return False, 0.0, 0.0
    return False, 0.0, 0.0


# input_white_line = training_data_input_white_line
# input_lane = training_data_input_lane
# # plt.figure()
# for i in range(training_data_input_xy.shape[0]):
#     theda = data_reshape[i, (input_size - 1), 5] * math.pi / 180
#     white_line_temp = white_line.copy()
#     lane_temp = lane.copy()
#     lane_temp = lane_temp[lane_temp[:, 5] == training_data_input_xy[i, -1, 6], :]
#     for j in range(white_line_temp.shape[0]):
#         x_temp = white_line_temp[j, 1] - data_reshape[i, (input_size - 1), 1]
#         y_temp = white_line_temp[j, 2] - data_reshape[i, (input_size - 1), 2]
#         white_line_temp[j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
#         white_line_temp[j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)
#         if white_line_temp[j, 2] < 0:
#             continue
#         flag, index_r, index_c = find_index(white_line_temp[j, 1], white_line_temp[j, 2])
#         if flag:
#             input_white_line[i, -1, index_r * row + index_c] = 1
#
#     for j in range(lane_temp.shape[0]):
#         x_temp = lane_temp[j, 1] - data_reshape[i, (input_size - 1), 1]
#         y_temp = lane_temp[j, 2] - data_reshape[i, (input_size - 1), 2]
#         lane_temp[j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
#         lane_temp[j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)
#         if lane_temp[j, 2] < 0.0:
#             continue
#         flag, index_r, index_c = find_index(lane_temp[j, 1], lane_temp[j, 2])
#         if flag:
#             input_lane[i, -1, index_r * row + index_c] = 1
#
#     # plt.subplot(2, 2, 1)
#     # lim = row * size_row + 1
#     # plt.xlim(-lim/2, lim/2)
#     # plt.ylim(-0, lim)
#     # # plt.plot(training_data_input_xy[i, :, 1], training_data_input_xy[i, :, 2], "*", color="r")
#     # # plt.plot(training_data_output[i, :, 1], training_data_output[i, :, 2], "*", color="b")
#     # plt.plot(white_line_temp[:, 1], white_line_temp[:, 2], "*", color="b")
#     # plt.plot(lane_temp[:, 1], lane_temp[:, 2], ".", color="b")
#     # for r in range(row):
#     #     plt.plot([index_box[r, 0, 0], index_box[r, -1, 1]],
#     #              [index_box[r, 0, 2], index_box[r, -1, 2]], "g--")
#     #     plt.plot([index_box[r, 0, 0], index_box[r, -1, 1]],
#     #              [index_box[r, 0, 3], index_box[r, -1, 3]], "g--")
#     # for c in range(column):
#     #     plt.plot([index_box[0, c, 0], index_box[-1, c, 0]],
#     #              [index_box[0, c, 2], index_box[-1, c, 3]], "g--")
#     #     plt.plot([index_box[0, c, 1], index_box[-1, c, 1]],
#     #              [index_box[0, c, 2], index_box[-1, c, 3]], "g--")
#     #
#     # for r in range(row):
#     #     for c in range(column):
#     #         if input_lane[i, -1, r * row + c] == 1.0:
#     #             plt.plot((index_box[r, c, 0] + index_box[r, c, 1]) / 2,
#     #                      (index_box[r, c, 2] + index_box[r, c, 3]) / 2,
#     #                      's', color="b")
#     #
#     # plt.subplot(2, 2, 2)
#     # plt.imshow(np.flip(input_white_line[i, -1, :].reshape(row, column), axis=0))
#     # plt.subplot(2, 2, 3)
#     # plt.imshow(np.flip(input_lane[i, -1, :].reshape(row, column), axis=0))
#     # plt.pause(0.1)
#     # plt.clf()
# input_white_line[np.nonzero(input_white_line)] = 1.0
# input_lane[np.nonzero(input_lane)] = 1.0
#
# input_white_line = testing_data_input_white_line
# input_lane = testing_data_input_lane
# # plt.figure()
# for i in range(testing_data_input_xy.shape[0]):
#     theda = data_reshape[i + train_size, (input_size - 1), 5] * math.pi / 180
#     white_line_temp = white_line.copy()
#     lane_temp = lane.copy()
#     lane_temp = lane_temp[lane_temp[:, 5] == testing_data_input_xy[i, -1, 6], :]
#     for j in range(white_line_temp.shape[0]):
#         x_temp = white_line_temp[j, 1] - data_reshape[i + train_size, (input_size - 1), 1]
#         y_temp = white_line_temp[j, 2] - data_reshape[i + train_size, (input_size - 1), 2]
#         white_line_temp[j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
#         white_line_temp[j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)
#         flag, index_r, index_c = find_index(white_line_temp[j, 1], white_line_temp[j, 2])
#         if flag:
#             input_white_line[i, -1, index_r * row + index_c] = 1
#
#     for j in range(lane_temp.shape[0]):
#         x_temp = lane_temp[j, 1] - data_reshape[i + train_size, (input_size - 1), 1]
#         y_temp = lane_temp[j, 2] - data_reshape[i + train_size, (input_size - 1), 2]
#         lane_temp[j, 1] = x_temp * math.cos(theda) - y_temp * math.sin(theda)
#         lane_temp[j, 2] = x_temp * math.sin(theda) + y_temp * math.cos(theda)
#         flag, index_r, index_c = find_index(lane_temp[j, 1], lane_temp[j, 2])
#         if flag:
#             input_lane[i, -1, index_r * row + index_c] = 1
#
#     # plt.subplot(2, 2, 1)
#     # lim = row * size_row + 1
#     # plt.xlim(-lim / 2, lim / 2)
#     # plt.ylim(-0, lim)
#     # plt.plot(testing_data_input_xy[i, :, 1], testing_data_input_xy[i, :, 2], "*", color="r")
#     # plt.plot(testing_data_output[i, :, 1], testing_data_output[i, :, 2], "*", color="b")
#     # plt.plot(white_line_temp[:, 1], white_line_temp[:, 2], "*", color="b")
#     # plt.plot(lane_temp[:, 1], lane_temp[:, 2], "*", color="b")
#     #
#     # for r in range(row):
#     #     plt.plot([index_box[r, 0, 0], index_box[r, -1, 1]],
#     #              [index_box[r, 0, 2], index_box[r, -1, 2]], "g--")
#     #     plt.plot([index_box[r, 0, 0], index_box[r, -1, 1]],
#     #              [index_box[r, 0, 3], index_box[r, -1, 3]], "g--")
#     # for c in range(column):
#     #     plt.plot([index_box[0, c, 0], index_box[-1, c, 0]],
#     #              [index_box[0, c, 2], index_box[-1, c, 3]], "g--")
#     #     plt.plot([index_box[0, c, 1], index_box[-1, c, 1]],
#     #              [index_box[0, c, 2], index_box[-1, c, 3]], "g--")
#     #
#     # # for r in range(row):
#     # #     for c in range(column):
#     # #         plt.text((index_box[r, c, 0] + index_box[r, c, 1]) / 2,
#     # #                  (index_box[r, c, 2] + index_box[r, c, 3]) / 2,
#     # #                  input_lane[i, -1, r * row + c],
#     # #                  fontsize=10)
#     #
#     # plt.subplot(2, 2, 2)
#     # plt.imshow(np.flip(input_white_line[i, -1, :].reshape(row, column), axis=0))
#     # plt.subplot(2, 2, 3)
#     # plt.imshow(np.flip(input_lane[i, -1, :].reshape(row, column), axis=0))
#     # plt.pause(0.01)
#     # plt.clf()
# input_white_line[np.nonzero(input_white_line)] = 1.0
# input_lane[np.nonzero(input_lane)] = 1.0

# # 移除index参数
training_data_input_xy = np.array(training_data_input_xy[:, :, :])
training_data_input_white_line = training_data_input_white_line
training_data_input_lane = training_data_input_lane
training_data_output = np.array(training_data_output[:, :, 1:3])

testing_data_input_xy = np.array(testing_data_input_xy[:, :, :])
testing_data_input_white_line = testing_data_input_white_line
testing_data_input_lane = testing_data_input_lane
testing_data_output = np.array(testing_data_output[:, :, 1:3])

# index = np.linspace(0, training_data_input_xy.shape[0] - 1, num=training_data_input_xy.shape[0])
# np.random.shuffle(index)
# index = index.astype(int)
# training_data_input_xy = training_data_input_xy[index]
# training_data_input_white_line = training_data_input_white_line[index]
# training_data_input_lane = training_data_input_lane[index]
# training_data_output = training_data_output[index]
#
# index = np.linspace(0, testing_data_input_xy.shape[0] - 1, num=testing_data_input_xy.shape[0])
# np.random.shuffle(index)
# index = index.astype(int)
# testing_data_input_xy = testing_data_input_xy[index]
# testing_data_input_white_line = testing_data_input_white_line[index]
# testing_data_input_lane = testing_data_input_lane[index]
# testing_data_output = testing_data_output[index]

