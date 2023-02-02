import torch
import torch.nn as nn
from utils import training_data_input, training_data_output, testing_data_input, testing_data_output, data
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置运行设备的环境为GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'本次程序运行的设备环境为{device}')

input_size = 1
num_layers = 5
hidden_size = 512
output_size = 3
batch_size = 35
sequence_length = 7
learning_rate = 1e-3
num_epochs = 100000
show_epoch = 100


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, sequence_length):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # 打印基本信息
        print(f'模型参数：每次输入{input_size}个点，输入{sequence_length}次，输出{output_size}个点')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 设置模型基本参数
model_lstm = LSTM(input_size, hidden_size, num_layers, output_size, sequence_length).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=learning_rate)

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
t_start = time.time()
for epoch in range(num_epochs):
    output = model_lstm(training_data_input)
    loss = criterion(output, training_data_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % show_epoch == 0:
        print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.5f}')
    if loss < 1e-4:
        print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.5f}')
        break

t_end = time.time()
print(f'训练时长为{t_end - t_start}s')


# 拓展点
def expend(source, num):
    aim = np.zeros(len(source) * num)
    for i in range(len(source)):
        aim[(i * num + num - len(source[i])):(i * num + num - len(source[i]) + 3)] = source[i].cpu()
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
    out_training = expend(out_training, 10)
    dis_training = expend(dis_training, 10)
    out_testing = expend(out_testing, 10)
    dis_testing = expend(dis_testing, 10)
    out_all = np.append(out_training, out_testing)
    dis_all = np.append(dis_training, dis_testing)
    # 显示原始数据
    print(f'误差最大值为:{dis_all.max()}米')
    plt.figure()
    plt.plot(data[:, 0], data[:, 2], 'b', label='input')
    plt.plot(data[:, 0], out_all[:], '*', label='output')
    plt.plot(data[:, 0], dis_all[:] * 10, 'g', label='dis*10')
    plt.legend(loc='upper right')
    plt.show()
