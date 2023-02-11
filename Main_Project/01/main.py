import torch
import torch.nn as nn
from utils import training_data_input, training_data_output, testing_data_input, testing_data_output, data, training_len
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置运行设备的环境为GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')

input_size = 1
num_layers = 2
hidden_size = 64
output_size = 1
sequence_length = 9
learning_rate = 1e-3
num_epochs = 100000
show_epoch = 500


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, sequence_length):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        hidden_size_hid = int(hidden_size/2)
        self.fc1 = nn.Linear(hidden_size, hidden_size_hid)
        self.fc2 = nn.Linear(hidden_size_hid, output_size)

        # 打印基本信息
        print(f'模型参数：每次输入{input_size}个点，输入{sequence_length}次，输出{output_size}个点')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# 设置模型基本参数
model_lstm = LSTM(input_size, hidden_size, num_layers, output_size, sequence_length).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adadelta(model_lstm.parameters(), lr=learning_rate)
print(model_lstm)

# 显示数据，并转换格式
training_data_input = torch.from_numpy(training_data_input).to(torch.float32)
training_data_output = torch.from_numpy(training_data_output).to(torch.float32).to(device)
testing_data_input = torch.from_numpy(testing_data_input).to(torch.float32)
testing_data_output = torch.from_numpy(testing_data_output).to(torch.float32).to(device)

training_data_input = training_data_input.reshape(-1, sequence_length, input_size).to(device)
testing_data_input = testing_data_input.reshape(-1, sequence_length, input_size).to(device)

print(f'training_data_input: {training_data_input.shape}')
print(f'training_data_output: {training_data_output.shape}')
print(f'testing_data_input: {testing_data_input.shape}')
print(f'testing_data_output: {testing_data_output.shape}')

# 训练模型
# t_start = time.time()
# for epoch in range(num_epochs):
#     output = model_lstm(training_data_input)
#     loss = criterion(output, training_data_output)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % show_epoch == 0:
#         print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.4f}', end="\r", flush=True)
#     if (loss < 1e-4) | ((epoch + 1) == num_epochs):
#         print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.5f}')
#         break
# t_end = time.time()
# print(f'训练时长为{t_end - t_start}s')

# # 保存模型
path = "model_lstm.pth"
# torch.save(model_lstm, path)

# # 加载模型
model_lstm = torch.load(path)
model_lstm.eval()


# 拓展点
def expend(source, num):
    aim = np.zeros(len(source) * num)
    for i in range(len(source)):
        aim[i * num + num - 1] = source[i]
    return aim


with torch.no_grad():
    # 转变为检测状态
    model_lstm = model_lstm.eval()
    # 检测训练集
    out_training = model_lstm(training_data_input)
    dis_training = (out_training - training_data_output).cpu()
    # 检测检测集
    out_testing = model_lstm(testing_data_input)
    dis_testing = (out_testing - testing_data_output).cpu()
    # 整合数据
    print(f'训练集误差最大值为:{abs(dis_testing).max()}米')
    print(f'训练集误差平均值为:{abs(dis_testing).mean()}米')
    out_training = expend(out_training, 10)
    dis_training = expend(dis_training, 10)
    out_testing = expend(out_testing, 10)
    dis_testing = expend(dis_testing, 10)
    out_all = np.append(out_training, out_testing)
    dis_all = np.append(dis_training, dis_testing)
    # 显示原始数据
    plt.figure()
    plt.plot(data[:, 0], data[:, 2], 'b', label='input')
    plt.plot(data[:, 0], out_all[:], '*', label='output')
    plt.plot(data[:, 0], dis_all[:] * 10, 'g', label='dis*10')
    plt.plot([data[training_len, 0], data[training_len, 0]], [-400, 100], 'r--', label='separation')
    plt.plot([0, len(out_all)], [max(out_training), max(out_training)], 'r--', label='Max')
    plt.legend(loc='upper right')
    plt.show()
