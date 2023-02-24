import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import training_data_input, training_data_output, testing_data_input, testing_data_output, data, \
                  training_len, training_data_output_start_point, testing_data_output_start_point
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import sys

# 定义tensorboard的writer
tensorboard_writer = SummaryWriter("../runs/06")

# 设置运行设备的环境为GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')

input_size = 1
num_layers = 2
hidden_size = 32
output_size = 1
sequence_length = 9
learning_rate = 2
num_epochs = 50000000
show_epoch = 500
basic_var = 1
cur_var = 0.0
add_change_ls = 0
history_loss = np.zeros((20, 1))
path = "model_lstm.pth"
training_or_testing = int(input("请输入模式选择，0代表训练，1代表只作测试："))  # 0:训练 1:测试
if training_or_testing == 0:
    print("本次程序将会进行训练，并测试")
else:
    print("本次程序只作测试")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, sequence_length):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        hidden_size_hid = int(hidden_size / 2)
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

tensorboard_writer.add_graph(model_lstm, training_data_input)


# 旋转打印进度
def print_process(num):
    if (num % 10) % 5 == 0:
        print('\r     ', end="", flush=True)
        print('\r-----', end="", flush=True)
    if (num % 10) % 5 == 1:
        print('\r\\\\\\\\\\', end="", flush=True)
    if (num % 10) % 5 == 2:
        print('\r|||||', end="", flush=True)
    if (num % 10) % 5 == 3:
        print('\r/////', end="", flush=True)
    if (num % 10) % 5 == 4:
        print('\r(0,0)', end="", flush=True)


# 更改学习率和优化器
def lr_optim(epoch, basic_var, cur_loss, lr, optim, add_change_ls):
    if cur_loss <= 1e-1:
        print("loss足够小，不再更改lr和optim")
        return lr, optim, 0.0, 0

    # 加入数据
    history_loss[0:(len(history_loss) - 1), 0] = history_loss[1:len(history_loss), 0]
    history_loss[-1, 0] = cur_loss

    if history_loss[0, 0] == 0.0:
        print("数据不够，函数lr_optim()暂不运行")
        return lr, optim, 0.0, 0

    loss_var = history_loss.var()
    if loss_var <= basic_var:
        # 数据出现震荡，减小学习率，使其收敛下来
        print(f'方差为{loss_var:.8f}<{basic_var}，可能处于震荡状态，当前学习率为{lr:.9f}')
        if add_change_ls >= 100:
            print(f'当前方差过小累计次数为{add_change_ls}，现重头开始累加')
            add_change_ls = 0
        add_change_ls = add_change_ls + 1
        print(f'已检测到方差过小的次数为{add_change_ls}')
        lr = lr - 0.1 * (5 * cur_loss / 70) * lr - 0.00001 * add_change_ls
        if lr <= 0:
            lr = 0.001
        print(f'在第{epoch + 1}个epoch将学习率改为{lr:.9f}')
        print()
        optim = torch.optim.Adadelta(model_lstm.parameters(), lr=lr)
    else:
        add_change_ls = 0

    return lr, optim, loss_var, add_change_ls


# 训练模型
if training_or_testing == 0:
    t_start = time.time()
    for epoch in range(num_epochs):
        output = model_lstm(training_data_input)
        loss = criterion(output, training_data_output)

        if (epoch + 1) % 10 == 0:
            tensorboard_writer.add_scalar('training_loss', loss.item(), epoch)
            tensorboard_writer.add_scalar('training_lr', learning_rate, epoch)
            tensorboard_writer.add_scalar('training_var', cur_var, epoch)
            learning_rate, optimizer, cur_var, add_change_ls = lr_optim(epoch, basic_var, loss.item(),
                                                                        learning_rate, optimizer, add_change_ls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % show_epoch == 0:
            # print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.4f}', end="\r", flush=True)
            print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.4f}')
            if loss.item() <= 1:
                temp_path = "models/" + str(epoch + 1) + "_" + "{:.4f}".format(loss.item()) + ".pth"
                torch.save(model_lstm, temp_path)
                print(f"已将模型文件保存为{temp_path}")
            print("---------------------------------------------------------")
        if (loss < 1) | ((epoch + 1) == num_epochs):
            print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.5f}')
            break
        # print_process(epoch + 1)

    t_end = time.time()
    print(f'训练时长为{t_end - t_start}s')
    torch.save(model_lstm, path)
    print(f"已将模型文件保存为{path}")
    tensorboard_writer.close()
else:
    model_lstm = torch.load(path)
    model_lstm.eval()
    print(f"已读取模型文件{path}")


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
    out_training = np.add(out_training.cpu(), training_data_output_start_point)
    # 检测检测集
    out_testing = model_lstm(testing_data_input)
    dis_testing = (out_testing - testing_data_output).cpu()
    out_testing = np.add(out_testing.cpu(), testing_data_output_start_point)
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
    plt.plot(data[:, 0], dis_all[:], 'g', label='dis')
    plt.plot([data[training_len, 0], data[training_len, 0]], [-400, 100], 'r--', label='separation')
    plt.plot([0, len(out_all)], [max(out_training), max(out_training)], 'r--', label='Max')
    plt.legend(loc='upper right')
    plt.show()
