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
learning_rate = 1e-3
start_loss = 12
end_loss = 10
num_epochs = 50000000
show_epoch = 500
basic_k = 0.01
cur_k = 0.0
add_change_ls = 0
history_loss = np.zeros((30, 1))
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

# training_data_input = training_data_input.reshape(-1, sequence_length, input_size).to(device)
# testing_data_input = testing_data_input.reshape(-1, sequence_length, input_size).to(device)

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
def lr_optim(epoch, basic_k, cur_loss, lr, optim, acl):
    # if cur_loss <= (end_loss * 10):
    #     print("loss足够小，不再更改lr和optim")
    #     return lr, optim, cur_loss, acl

    # 加入数据
    history_loss[0:(len(history_loss) - 1), 0] = history_loss[1:len(history_loss), 0]
    history_loss[-1, 0] = cur_loss

    if history_loss[0, 0] == 0.0:
        # print("数据不够，函数lr_optim()暂不运行")
        return lr, optim, cur_loss, acl

    loss_k = np.polyfit(np.arange(history_loss.shape[0]), history_loss, 1)[0]
    print(f'当前斜率为{loss_k.item():.15f}')

    if abs(loss_k) <= basic_k:
        # 调整学习率，使其收敛下来
        print(f'abs(loss_k)<{basic_k}，数据可能处于震荡状态，当前学习率为{lr:.15f}')
        if loss_k <= 0:
            # loss基本上在减小
            print("loss<0，正在正常减小")
            # if abs(loss_k) >= (4 * basic_k / 5):
            if False:
                # 当前loss正在下降，但是速度明显太慢
                lr = lr + math.cos(loss_k / basic_k * math.pi / 2) * 1e-2 * lr + acl * 1e-4 * lr
                print(f'loss降低速度过慢，已调整学习率为{lr:.15f}')
            else:
                # 数据可能震荡了
                lr = lr - math.cos(loss_k / basic_k * math.pi / 2) * 1e-2 * lr - acl * 1e-4 * lr
                print(f'loss可能震荡，已调整学习率为{lr:.15f}')
            acl = acl + 1
            if acl > 1e2:
                acl = 0
            print(f'add_change_ls = {acl}')
        else:
            # loss可能在增大
            lr = lr * 99 / 100
            print(f'loss可能增大，已调整学习率为{lr:.15f}')
            acl = 0
            print(f'已重置add_change_ls')
        history_loss[int((history_loss.shape[0] / 2)):int(history_loss.shape[0])] = history_loss[
                                                                                    0:int(history_loss.shape[0] / 2)]
        history_loss[0:int((history_loss.shape[0] / 2))] = 0.0
    print()
    return lr, optim, loss_k, acl


# 训练模型
if training_or_testing == 0:
    t_start = time.time()
    for epoch in range(num_epochs):
        output = model_lstm(training_data_input)
        loss = criterion(output, training_data_output) * training_data_input.shape[0]

        if (epoch + 1) % 20 == 0:
            tensorboard_writer.add_scalar('training_loss', loss.item(), epoch)
            tensorboard_writer.add_scalar('training_lr', learning_rate, epoch)
            tensorboard_writer.add_scalar('training_k', cur_k, epoch)
            if loss.item() <= start_loss:
                learning_rate, optimizer, cur_k, add_change_ls = lr_optim(epoch, basic_k, loss.item(),
                                                                          learning_rate, optimizer, add_change_ls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % show_epoch == 0:
            # print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.4f}', end="\r", flush=True)
            print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.4f}')
            if loss.item() <= (1.1 * end_loss):
                temp_path = "models/" + str(epoch + 1) + "_" + "{:.4f}".format(loss.item()) + ".pth"
                torch.save(model_lstm, temp_path)
                print(f"已将模型文件保存为{temp_path}")
            print("---------------------------------------------------------")
        if (loss < end_loss) | ((epoch + 1) == num_epochs):
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
    dis_all = np.append(dis_training, dis_testing) * 100.0
    # 显示原始数据
    plt.figure()
    plt.plot(data[:, 0], data[:, 2], 'b', label='input')
    plt.plot(data[:, 0], out_all[:], '*', label='output')
    plt.plot(data[:, 0], dis_all[:], 'g', label='dis * 100')
    plt.plot([data[training_len, 0], data[training_len, 0]], [-400, 100], 'r--', label='separation')
    plt.plot([0, len(out_all)], [max(out_training), max(out_training)], 'r--', label='Max')
    plt.legend(loc='upper right')
    plt.show()
