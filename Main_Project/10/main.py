import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from util import training_data_input_xy, training_data_input_white_line, training_data_input_lane
from util import testing_data_input_xy, testing_data_input_white_line, testing_data_input_lane
from util import training_data_output, testing_data_output, index_box, data_reshape
from util import data_size, input_size, row, column, size_row, theda_train, theda_test
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as scheduler

# 启动前检查
print('CUDA版本:', torch.version.cuda)
print('Pytorch版本:', torch.__version__)
print('显卡是否可用:', '可用' if (torch.cuda.is_available()) else '不可用')
print('显卡数量:', torch.cuda.device_count())
print('当前显卡型号:', torch.cuda.get_device_name())
print('当前显卡的总显存:', torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 'GB')

# 清空缓存，固定随即种子
torch.manual_seed(1)
torch.cuda.empty_cache()

# 设置运行设备的环境为GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')

# 更改数据类型
# training_data_input_xy = torch.from_numpy(training_data_input_xy).to(torch.float32).to(device)
# training_data_input_white_line = torch.from_numpy(training_data_input_white_line).to(torch.float32).to(device)
# training_data_input_lane = torch.from_numpy(training_data_input_lane).to(torch.float32).to(device)
# training_data_output = torch.from_numpy(training_data_output).to(torch.float32).to(device)
# testing_data_input_xy = torch.from_numpy(testing_data_input_xy).to(torch.float32).to(device)
# testing_data_input_white_line = torch.from_numpy(testing_data_input_white_line).to(torch.float32).to(device)
# testing_data_input_lane = torch.from_numpy(testing_data_input_lane).to(torch.float32).to(device)
# testing_data_output = torch.from_numpy(testing_data_output).to(torch.float32).to(device)

print(f'training_data_input_xy: {training_data_input_xy.shape}')
print(f'training_data_input_white_line: {training_data_input_white_line.shape}')
print(f'training_data_input_lane: {training_data_input_lane.shape}')
print(f'training_data_output: {training_data_output.shape}')
print(f'testing_data_input_xy: {testing_data_input_xy.shape}')
print(f'testing_data_input_white_line: {testing_data_input_white_line.shape}')
print(f'testing_data_input_lane: {testing_data_input_lane.shape}')
print(f'testing_data_output: {testing_data_output.shape}')

# 模式选取
mode_switch = np.array([1, 1, 1, 1, 1])
vector_map_switch = 1
check_source_switch = 0

# 定义基本参数
size_basic = 64
size_encoder_fc_input = data_size - 1  # 去除索引
size_encoder_fc_middle = size_basic
size_encoder_fc_output = size_basic
size_encoder_lstm_input = size_encoder_fc_output
size_encoder_lstm_hidden = size_basic
size_encoder_activate_num_parameters = input_size

size_decoder_lstm_input = 2 * size_encoder_lstm_hidden + 1 * row * column * vector_map_switch
size_decoder_lstm_hidden = 2 * size_encoder_lstm_hidden
size_decoder_fc_input = size_decoder_lstm_hidden
size_decoder_fc_middle = size_basic
size_decoder_fc_output = 2

size_connector_fc_input = size_decoder_fc_output
size_connector_fc_middle = size_basic
size_connector_fc_output = 2 * size_encoder_lstm_hidden

learning_rate_init = 1e-3
learning_rate = learning_rate_init
max_epoch = 200
batch_ratio = 0.2


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, encoder_fc_input_size, encoder_fc_middle_size, encoder_fc_output_size,
                 encoder_lstm_input_size, encoder_lstm_hidden_size,
                 encoder_activate_num_parameters_size):
        super(Encoder, self).__init__()
        self.encoder_activate_init = 1.0
        self.encoder_lstm_hidden_size = encoder_lstm_hidden_size
        self.encoder_bias = False
        self.encoder_lstm_num_layers = 1

        self.encoder_fc1 = nn.Linear(encoder_fc_input_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc2 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc3 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc4 = nn.Linear(encoder_fc_middle_size, encoder_fc_output_size, bias=self.encoder_bias)
        self.encoder_lstm_front = nn.LSTM(encoder_lstm_input_size, encoder_lstm_hidden_size,
                                          num_layers=self.encoder_lstm_num_layers, batch_first=True)
        self.encoder_lstm_back = nn.LSTM(encoder_lstm_input_size, encoder_lstm_hidden_size,
                                         num_layers=self.encoder_lstm_num_layers, batch_first=True)
        self.encoder_activate1 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        self.encoder_activate2 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        self.encoder_activate3 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        self.encoder_activate4 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        # self.encoder_normalization = nn.BatchNorm1d(encoder_activate_num_parameters_size, affine=False)
        self.encoder_normalization = nn.LayerNorm([encoder_activate_num_parameters_size, encoder_fc_middle_size])

    def forward(self, x):
        h0 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        c0 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        h1 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        c1 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)

        out = self.encoder_fc1(self.encoder_activate1(x))
        # out = self.encoder_normalization(out)
        # out = self.encoder_fc2(self.encoder_activate2(out))
        # out = self.encoder_normalization(out)
        # out = self.encoder_fc3(self.encoder_activate3(out))
        # out = self.encoder_normalization(out)
        out = self.encoder_fc4(self.encoder_activate4(out))
        # out = self.encoder_normalization(out)
        out_front, (h_front, c_front) = self.encoder_lstm_front(out, (h0, c0))
        out_back, (h_back, c_back) = self.encoder_lstm_back(out.flip(dims=[1]), (h1, c1))
        h = torch.cat([h_front, h_back], 2)
        c = torch.cat([c_front, c_back], 2)
        out = torch.cat([out_front[:, -1, :].unsqueeze(1), out_back[:, -1, :].unsqueeze(1)], 2)
        return out, (h, c)


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, decoder_lstm_input_size, decoder_lstm_hidden_size,
                 decoder_fc_input_size, decoder_fc_middle_size, decoder_fc_output_size):
        super(Decoder, self).__init__()
        self.decoder_lstm_hidden_size = decoder_lstm_hidden_size
        self.decoder_bias = False
        self.decoder_activate_init = 1
        self.decoder_lstm_num_layers = 1

        self.decoder_lstm = nn.LSTM(decoder_lstm_input_size, decoder_lstm_hidden_size,
                                    num_layers=self.decoder_lstm_num_layers, batch_first=True)
        self.decoder_fc1 = nn.Linear(decoder_fc_input_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc2 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc3 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc4 = nn.Linear(decoder_fc_middle_size, decoder_fc_output_size, bias=self.decoder_bias)
        self.decoder_activate1 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_activate2 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_activate3 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_activate4 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        # self.decoder_normalization = nn.BatchNorm1d(1, affine=False)
        self.decoder_normalization = nn.LayerNorm([1, decoder_fc_middle_size])

    def forward(self, x, h1, c1):
        out, (h2, c2) = self.decoder_lstm(x, (h1, c1))
        out = self.decoder_fc1(self.decoder_activate1(out))
        # out = self.decoder_normalization(out)
        # out = self.decoder_fc2(self.decoder_activate2(out))
        # out = self.decoder_normalization(out)
        # out = self.decoder_fc3(self.decoder_activate3(out))
        # out = self.decoder_normalization(out)
        out = self.decoder_fc4(self.decoder_activate4(out))
        return out, (h2, c2)


# 定义传递器
class Connector(nn.Module):
    def __init__(self, connector_fc_input_size, connector_fc_middle_size, connector_fc_output_size):
        super(Connector, self).__init__()
        self.connector_bias = False
        self.connector_activate_init = 1

        self.connector_fc1 = nn.Linear(connector_fc_input_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc2 = nn.Linear(connector_fc_middle_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc3 = nn.Linear(connector_fc_middle_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc4 = nn.Linear(connector_fc_middle_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc5 = nn.Linear(connector_fc_middle_size, connector_fc_output_size, bias=self.connector_bias)
        self.connector_activate1 = nn.PReLU(num_parameters=1,
                                            init=self.connector_activate_init)
        self.connector_activate2 = nn.PReLU(num_parameters=1,
                                            init=self.connector_activate_init)
        self.connector_activate3 = nn.PReLU(num_parameters=1,
                                            init=self.connector_activate_init)
        self.connector_activate4 = nn.PReLU(num_parameters=1,
                                            init=self.connector_activate_init)
        self.connector_activate5 = nn.PReLU(num_parameters=1,
                                            init=self.connector_activate_init)
        self.connector_normalization = nn.LayerNorm([1, connector_fc_middle_size])

    def forward(self, x):
        out = self.connector_fc1(self.connector_activate1(x))
        # out = self.connector_normalization(out)
        # out = self.connector_fc2(self.connector_activate2(out))
        # out = self.connector_normalization(out)
        # out = self.connector_fc3(self.connector_activate3(out))
        # out = self.connector_normalization(out)
        # out = self.connector_fc4(self.connector_activate4(out))
        # out = self.connector_normalization(out)
        out = self.connector_fc5(self.connector_activate5(out))
        return out


# 模型实例化
encoder = Encoder(size_encoder_fc_input, size_encoder_fc_middle, size_encoder_fc_output,
                  size_encoder_lstm_input, size_encoder_lstm_hidden, size_encoder_activate_num_parameters).to(device)
decoder = Decoder(size_decoder_lstm_input, size_decoder_lstm_hidden,
                  size_decoder_fc_input, size_decoder_fc_middle, size_decoder_fc_output).to(device)
connector = Connector(size_connector_fc_input, size_connector_fc_middle, size_connector_fc_output).to(device)

# 优化器和损失函数
optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate)
optimizer_connector = optim.Adam(connector.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


# 停止判定
def judge_end(point_num, grad_min, grad_max, loss_item):
    ref = math.sqrt(point_num)
    # print(loss_item < 2, abs(grad_max) <= 2, abs(grad_min) <= 0.0001)
    if loss_item < ref and abs(grad_max) <= ref and abs(grad_min) <= 0.001:
        return True
    return False


# 主要部分
t_start = time.time()
fig = plt.figure()
if mode_switch[0] == 1:
    print("进行单点模型训练")

    scheduler_encoder = scheduler.StepLR(optimizer_encoder, step_size=int(min(max_epoch / 10, 10)), gamma=0.7,
                                         last_epoch=-1)
    scheduler_decoder = scheduler.StepLR(optimizer_decoder, step_size=int(min(max_epoch / 10, 10)), gamma=0.7,
                                         last_epoch=-1)

    all_loss = np.zeros([1])
    for epoch in range(int(max_epoch / 2)):

        batch_size = training_data_input_xy.shape[0] * batch_ratio
        for each_batch in range(int(1 / batch_ratio)):
            print(f"epoch:{epoch + 1}/{max_epoch / 50}, "
                  f"batch:{each_batch + 1}/{int(1 / batch_ratio)}, "
                  f"loss:{all_loss[-1]:.5f}")
            index_start = int(each_batch * batch_size)
            index_end = int((each_batch + 1) * batch_size)
            train_input_xy = training_data_input_xy[index_start:index_end, :, :][:, :, 1:data_size]
            train_input_white_line = training_data_input_white_line[index_start:index_end, :, :]
            train_input_lane = training_data_input_lane[index_start:index_end, :, :]
            train_output = training_data_output[index_start:index_end, :, :]

            train_input_xy = torch.from_numpy(train_input_xy).to(torch.float32).to(device)
            train_input_white_line = torch.from_numpy(train_input_white_line).to(torch.float32).to(device)
            train_input_lane = torch.from_numpy(train_input_lane).to(torch.float32).to(device)
            train_output = torch.from_numpy(train_output).to(torch.float32).to(device)

            encoded, (h_encoded, c_encoded) = encoder(train_input_xy)
            if vector_map_switch == 1:
                encoded = torch.cat([encoded, train_input_lane], 2)
            decoded, _ = decoder(encoded, h_encoded, c_encoded)
            loss = criterion(decoded, train_output[:, 0, :].unsqueeze(1))
            all_loss[epoch] = loss.item()

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

        plt.clf()

        plt.title("Single Point Prediction Training Data", fontsize=15)
        plt.plot(all_loss, label="Loss")
        plt.legend(loc='upper right')
        # plt.text(epoch / 2, (all_loss[:].max() + all_loss[:].min()) / 2,
        #          f"lr:{learning_rate:.10f}, all_loss:{all_loss[-1]}", fontsize=15)
        plt.legend()

        plt.pause(0.001)

        all_loss = np.append(all_loss, [0.0], axis=0)

        scheduler_encoder.step()
        scheduler_decoder.step()
        learning_rate = scheduler_encoder.get_last_lr()[0]

        rand_para = torch.randperm(training_data_input_xy.shape[0])
        training_data_input_xy = training_data_input_xy[rand_para]
        training_data_input_white_line = training_data_input_white_line[rand_para]
        training_data_input_lane = training_data_input_lane[rand_para]
        training_data_output = training_data_output[rand_para]

    torch.save(encoder, "end_encoder.pth")
    torch.save(decoder, "end_decoder.pth")
    fig.savefig("figs/" + "single.png")
    plt.clf()
    print("--------------")
    print(f"当前程序运行时间为：{int((time.time() - t_start) / 60)} min, {(time.time() - t_start) % 60}s")
if mode_switch[1] == 1:
    print("单点模型测试")

    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")

    all_loss = np.zeros([1])
    if check_source_switch == 0:
        batch_size = testing_data_input_xy.shape[0] * batch_ratio
    else:
        batch_size = training_data_input_xy.shape[0] * batch_ratio
    for each_batch in range(int(1 / batch_ratio)):
        torch.cuda.empty_cache()
        index_start = int(each_batch * batch_size)
        index_end = int((each_batch + 1) * batch_size)
        if check_source_switch == 0:
            check_input_xy = testing_data_input_xy[index_start:index_end, :, :][:, :, 1:data_size]
            check_input_white_line = testing_data_input_white_line[index_start:index_end, :, :]
            check_input_lane = testing_data_input_lane[index_start:index_end, :, :]
            check_output = testing_data_output[index_start:index_end, 0:1, :]
        else:
            check_input_xy = training_data_input_xy[index_start:index_end, :, :][:, :, 1:data_size]
            check_input_white_line = training_data_input_white_line[index_start:index_end, :, :]
            check_input_lane = training_data_input_lane[index_start:index_end, :, :]
            check_output = training_data_output[index_start:index_end, 0:1, :]

        check_input_xy = torch.from_numpy(check_input_xy).to(torch.float32).to(device)
        check_input_white_line = torch.from_numpy(check_input_white_line).to(torch.float32).to(device)
        check_input_lane = torch.from_numpy(check_input_lane).to(torch.float32).to(device)
        check_output = torch.from_numpy(check_output).to(torch.float32).to(device)

        encoded, (h_encoded, c_encoded) = encoder(check_input_xy)
        if vector_map_switch == 1:
            encoded = torch.cat([encoded, check_input_lane], 2)
        decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)

        for i in range(check_output.shape[0]):
            loss = criterion(decoded[i], check_output[i])
            all_loss[-1] = loss.item()
            all_loss = np.append(all_loss, [0.0], axis=0)
        all_loss = np.delete(all_loss, [-1], 0)
        plt.clf()
        plt.title("Single Point Prediction Testing Data", fontsize=15)
        plt.plot(all_loss, label="Loss")
        plt.text(all_loss.shape[0] / 2, all_loss.max() * 1.01,
                 f"Max:{all_loss.max():.5f}", fontsize=10)
        plt.plot([0, all_loss.shape[0]],
                 [all_loss.max(), all_loss.max()],
                 "r--", label="Max")
        plt.text(all_loss.shape[0] / 2, all_loss.mean() * 2,
                 f"Mean:{all_loss.mean():.5f}", fontsize=10)
        plt.plot([0, all_loss.shape[0]],
                 [all_loss.mean(), all_loss.mean()],
                 "k:", label="Mean")
        plt.legend(loc='upper right')
        plt.pause(0.01)
    plt.savefig("../result/10/single.png")
    print("--------------")
    print(f"当前程序运行时间为：{int((time.time() - t_start) / 60)} min, {(time.time() - t_start) % 60}s")
if mode_switch[2] == 1:
    print("进行连接模型训练")
    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")

    for points in range(1, training_data_output.shape[1], 1):
        learning_rate = learning_rate_init * points
        optimizer_decoder = optim.Adam(decoder.parameters(), lr=(learning_rate / 10))
        optimizer_connector = optim.Adam(connector.parameters(), lr=learning_rate)
        scheduler_decoder = scheduler.StepLR(optimizer_decoder, step_size=int(min(max_epoch / 10, 10)), gamma=0.7,
                                             last_epoch=-1)
        scheduler_connector = scheduler.StepLR(optimizer_connector, step_size=int(min(max_epoch / 10, 10)), gamma=0.7,
                                               last_epoch=-1)

        all_loss = np.zeros([1])
        all_grad_abs = np.array([[0.0, 10]])
        for epoch in range(max_epoch):
            batch_size = training_data_input_xy.shape[0] * batch_ratio
            for each_batch in range(int(1 / batch_ratio)):
                print(f"epoch:{epoch + 1}/{max_epoch}, "
                      f"points:{points + 1}/{training_data_output.shape[1]}, "
                      f"batch:{each_batch + 1}/{int(1 / batch_ratio)}, "
                      f"loss:{all_loss[-1]:.5f}")
                torch.cuda.empty_cache()
                index_start = int(each_batch * batch_size)
                index_end = int((each_batch + 1) * batch_size)
                train_input_xy = training_data_input_xy[index_start:index_end, :, :][:, :, 1:data_size]
                train_input_white_line = training_data_input_white_line[index_start:index_end, :, :]
                train_input_lane = training_data_input_lane[index_start:index_end, :, :]
                train_output = training_data_output[index_start:index_end, :, :]

                train_input_xy = torch.from_numpy(train_input_xy).to(torch.float32).to(device)
                train_input_white_line = torch.from_numpy(train_input_white_line).to(torch.float32).to(device)
                train_input_lane = torch.from_numpy(train_input_lane).to(torch.float32).to(device)
                train_output = torch.from_numpy(train_output).to(torch.float32).to(device)

                encoded, (h_encoded, c_encoded) = encoder(train_input_xy)
                if vector_map_switch == 1:
                    encoded = torch.cat([encoded, train_input_lane], 2)
                decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
                decoded_clone = decoded.clone()
                for point in range(points):
                    connected = connector(decoded)
                    encoded, h_encoded, c_encoded = connected, h_decoded, c_decoded
                    if vector_map_switch == 1:
                        encoded = torch.cat([encoded, train_input_lane], 2)
                    decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
                    decoded_clone = torch.cat((decoded_clone.clone(), decoded.clone()), 1)

                loss = criterion(decoded_clone, train_output[:, 0:(points + 1), :]) * decoded_clone.shape[0]
                all_loss[epoch] = loss.item()

                optimizer_connector.zero_grad()
                optimizer_decoder.zero_grad()
                loss.backward()
                optimizer_connector.step()
                optimizer_decoder.step()

            for name, param in decoder.named_parameters():
                if param.grad is None:
                    continue
                if abs(param.grad.cpu().numpy().max()) >= all_grad_abs[-1, 0]:
                    all_grad_abs[-1, 0] = param.grad.cpu().numpy().max()

                if abs(param.grad.cpu().numpy().min()) >= all_grad_abs[-1, 0]:
                    all_grad_abs[-1, 0] = param.grad.cpu().numpy().min()

                temp = param.grad.cpu().numpy()
                temp = temp[np.nonzero(temp)]
                temp = temp[np.abs(temp).argmin()]
                if abs(all_grad_abs[-1, 1]) >= abs(temp):
                    all_grad_abs[-1, 1] = temp

            for name, param in connector.named_parameters():
                if param.grad is None:
                    continue
                if abs(param.grad.cpu().numpy().max()) >= all_grad_abs[-1, 0]:
                    all_grad_abs[-1, 0] = param.grad.cpu().numpy().max()

                if abs(param.grad.cpu().numpy().min()) >= all_grad_abs[-1, 0]:
                    all_grad_abs[-1, 0] = param.grad.cpu().numpy().min()

                temp = param.grad.cpu().numpy()
                temp = temp[np.nonzero(temp)]
                if temp.shape[0] == 0.0:
                    temp = 0.0
                else:
                    temp = temp[np.abs(temp).argmin()]
                if abs(all_grad_abs[-1, 1]) >= abs(temp):
                    all_grad_abs[-1, 1] = temp

            plt.clf()

            plt.subplot(3, 1, 1)
            plt.title("Trajectory Prediction Training Data", fontsize=15)
            plt.plot(all_loss, label="Loss")
            plt.plot([0.0, epoch], [0.0, 0.0], "k--", label="Reference")
            plt.text(epoch, all_loss[:].mean(),
                     f"Point:{points + 1}, lr:{learning_rate * 1e4:.3f}*10^-4, Loss:{all_loss[-1]:.5f}",
                     horizontalalignment="right", fontsize=10)
            plt.legend(loc='upper right')

            plt.subplot(3, 1, 2)
            plt.plot(all_grad_abs[:, 0], label="Grad")
            plt.plot([0.0, epoch], [0.0, 0.0], "k--", label="Reference")
            plt.text(epoch, abs(all_grad_abs[-1, 0] * 1.01),
                     f"Point:{points + 1}, Grad:{all_grad_abs[-1, 0]:.5f}", horizontalalignment="right", fontsize=10)
            plt.legend(loc='upper right')

            plt.subplot(3, 1, 3)
            plt.plot(all_grad_abs[:, 1], label="Grad")
            plt.plot([0.0, epoch], [0.0, 0.0], "k--", label="Reference")
            plt.text(epoch, abs(all_grad_abs[-1, 1] * 1.5),
                     f"Point:{points + 1}, Grad:{all_grad_abs[-1, 1] * 1e7:.5f}*10^-7", horizontalalignment="right", fontsize=10)
            plt.legend(loc='upper right')

            plt.pause(0.01)

            if epoch >= 10 and judge_end(points+1, all_grad_abs[-1, 1], all_grad_abs[-1, 0], all_loss[-1]):
                print(f"points:{points + 1},epoch:{epoch + 1},loss:{all_loss[-1]}")
                break

            all_loss = np.append(all_loss, [0.0], axis=0)
            all_grad_abs = np.append(all_grad_abs, np.array([[0.0, 10]]), axis=0)

            scheduler_decoder.step()
            scheduler_connector.step()
            learning_rate = scheduler_connector.get_last_lr()[0]

            rand_para = torch.randperm(training_data_input_xy.shape[0])
            training_data_input_xy = training_data_input_xy[rand_para]
            training_data_input_white_line = training_data_input_white_line[rand_para]
            training_data_input_lane = training_data_input_lane[rand_para]
            training_data_output = training_data_output[rand_para]

        fig.savefig("figs/" + str(points + 1) + ".png")
        print(f"points:{points + 1},loss:{all_loss[-2]}")
        print("------------------------------------")

        torch.save(decoder, "end_decoder.pth")
        torch.save(connector, "end_connector.pth")
    plt.clf()
    print("--------------")
    print(f"当前程序运行时间为：{int((time.time() - t_start) / 60)} min, {(time.time() - t_start) % 60}s")
if mode_switch[3] == 1:
    print("循环预测模型测试")
    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")
    connector = torch.load("end_connector.pth")

    all_loss = np.zeros([1])
    if check_source_switch == 0:
        batch_size = testing_data_input_xy.shape[0] * batch_ratio
    else:
        batch_size = training_data_input_xy.shape[0] * batch_ratio
    for each_batch in range(int(1 / batch_ratio)):
        torch.cuda.empty_cache()
        print(f"batch:{each_batch + 1}/{int(1 / batch_ratio)}")
        index_start = int(each_batch * batch_size)
        index_end = int((each_batch + 1) * batch_size)
        if check_source_switch == 0:
            check_input_xy = testing_data_input_xy[index_start:index_end, :, :][:, :, 1:data_size]
            check_input_white_line = testing_data_input_white_line[index_start:index_end, :, :]
            check_input_lane = testing_data_input_lane[index_start:index_end, :, :]
            check_output = testing_data_output[index_start:index_end, :, :]
        else:
            check_input_xy = training_data_input_xy[index_start:index_end, :, :][:, :, 1:data_size]
            check_input_white_line = training_data_input_white_line[index_start:index_end, :, :]
            check_input_lane = training_data_input_lane[index_start:index_end, :, :]
            check_output = training_data_output[index_start:index_end, :, :]

        check_input_xy = torch.from_numpy(check_input_xy).to(torch.float32).to(device)
        check_input_white_line = torch.from_numpy(check_input_white_line).to(torch.float32).to(device)
        check_input_lane = torch.from_numpy(check_input_lane).to(torch.float32).to(device)
        check_output = torch.from_numpy(check_output).to(torch.float32).to(device)

        encoded, (h_encoded, c_encoded) = encoder(check_input_xy)
        if vector_map_switch == 1:
            encoded = torch.cat([encoded, check_input_lane], 2)
        decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
        output = decoded.clone()

        for i in range(check_output.shape[1] - 1):
            connected = connector(decoded)
            encoded, h_encoded, c_encoded = connected, h_decoded, c_decoded
            if vector_map_switch == 1:
                encoded = torch.cat([encoded, check_input_lane], 2)
            decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
            output = torch.cat((output.clone(), decoded.clone()), 1)

        for i in range(check_output.shape[0]):
            loss = criterion(output[i], check_output[i])
            all_loss[-1] = loss.item()
            all_loss = np.append(all_loss, [0.0], axis=0)

        for i in range(0, check_output.shape[0]):
            if all_loss[(index_start + i)] >= 1.0:
                fig.savefig("../result/10/" + str(each_batch) + "_" + str(i) + ".png")
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.title("Trajectories", fontsize=15)
            lim = row * size_row + 1
            plt.xlim(-lim / 2, lim / 2)
            plt.ylim(-lim + 2, lim)

            for r in range(row):
                plt.plot([index_box[r, 0, 0], index_box[r, -1, 1]],
                         [index_box[r, 0, 2], index_box[r, -1, 2]], "y-")
                plt.plot([index_box[r, 0, 0], index_box[r, -1, 1]],
                         [index_box[r, 0, 3], index_box[r, -1, 3]], "y-")
            for c in range(column):
                plt.plot([index_box[0, c, 0], index_box[-1, c, 0]],
                         [index_box[0, c, 2], index_box[-1, c, 3]], "y-")
                plt.plot([index_box[0, c, 1], index_box[-1, c, 1]],
                         [index_box[0, c, 2], index_box[-1, c, 3]], "y-")
            plt.plot([index_box[0, 0, 0], index_box[0, -1, 1]],
                     [index_box[0, 0, 2], index_box[0, -1, 2]], "y-", label="Grid Map")

            flag = True
            for r in range(row):
                for c in range(column):
                    if check_input_lane[i, -1, r * row + c] == 1.0:
                        if flag:
                            plt.plot((index_box[r, c, 0] + index_box[r, c, 1]) / 2,
                                     (index_box[r, c, 2] + index_box[r, c, 3]) / 2,
                                     's', color="b", label="Occupied")
                            flag = False
                        else:
                            plt.plot((index_box[r, c, 0] + index_box[r, c, 1]) / 2,
                                     (index_box[r, c, 2] + index_box[r, c, 3]) / 2,
                                     's', color="b")

            plt.plot(check_input_xy.cpu().detach().numpy()[i, :, 0],
                     check_input_xy.cpu().detach().numpy()[i, :, 1], "-", color="k", label="Input")
            plt.plot(output.cpu().detach().numpy()[i, :, 0],
                     output.cpu().detach().numpy()[i, :, 1], ".", color="r", label="Output")
            plt.plot(check_output.cpu().detach().numpy()[i, :, 0],
                     check_output.cpu().detach().numpy()[i, :, 1], "--", color="k", label="Reference")
            plt.legend(loc='upper right')

            plt.subplot(1, 2, 2)
            plt.title("Loss", fontsize=15)
            plt.plot(all_loss[0:(index_start + i + 1)], label="Loss")
            plt.plot([0, index_start + i],
                     [all_loss[0:(index_start + i + 1)].max(), all_loss[0:(index_start + i + 1)].max()],
                     "r--", label="Max")
            plt.text((index_start + i) / 2, all_loss[0:(index_start + i + 1)].max() * 1.001,
                     f"Max:{all_loss[0:(index_start + i + 1)].max():.5f}", fontsize=10)
            plt.plot([0, index_start + i],
                     [all_loss[0:(index_start + i + 1)].mean(), all_loss[0:(index_start + i + 1)].mean()],
                     "k:", label="Mean")
            plt.text((index_start + i) / 2, all_loss[0:(index_start + i + 1)].mean() * 1.001,
                     f"Mean:{all_loss[0:(index_start + i + 1)].mean():.5f}", fontsize=10)
            plt.legend(loc='upper right')

            plt.pause(0.01)
        fig.savefig("../result/10/all.png")
        np.save("all_loss.npy", all_loss)
    plt.clf()
    print("--------------")
    print(f"当前程序运行时间为：{int((time.time() - t_start) / 60)} min, {(time.time() - t_start) % 60}s")
if mode_switch[4] == 1:
    print("分析偏差")
    all_loss = np.load("all_loss.npy")
    print(all_loss.shape)
    num = 17
    loss_area = np.linspace(0, 4, num=num)
    loss_num = np.zeros([num-1, 1])

    for i in range(all_loss.shape[0]):
        for j in range(num-1):
            if loss_area[j] <= all_loss[i] < loss_area[j + 1]:
                loss_num[j] = loss_num[j] + 1
                break

    loss_num_rate = loss_num / loss_num.sum()

    plt.title("Probability Distributions", fontsize=15)
    x = np.zeros([num-1])
    y = np.zeros([num-1])
    for i in range(num-1):
        x[i] = (loss_area[i] + loss_area[i + 1]) / 2
        y[i] = loss_num_rate[i, 0]
        if i == 0:
            plt.bar(x[i], y[i],
                    color="c", edgecolor='k', width=(loss_area[i + 1] - loss_area[i]), label="Proportion")
        else:
            plt.bar(x[i], y[i],
                    color="c", edgecolor='k', width=(loss_area[i+1] - loss_area[i]))
        plt.text(x[i], y[i], int(loss_num[i, 0]),
                 horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    plt.plot(x, y, "r", label="Trend")
    plt.legend(loc='upper right')
    plt.pause(2)
    fig.savefig("../result/10/distributions.png")

    print(f"当前程序运行时间为：{int((time.time() - t_start) / 60)} min, {(time.time() - t_start) % 60}s")
