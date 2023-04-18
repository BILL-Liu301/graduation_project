import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from util import training_data_input_xy, training_data_input_white_line, training_data_input_lane
from util import testing_data_input_xy, testing_data_input_white_line, testing_data_input_lane
from util import training_data_output, testing_data_output, index_box
from util import data_size, input_size, row, column
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as scheduler

# 启动前检查
print('CUDA版本:', torch.version.cuda)
print('Pytorch版本:', torch.__version__)
print('显卡是否可用:', '可用' if (torch.cuda.is_available()) else '不可用')
print('显卡数量:', torch.cuda.device_count())
print('是否支持BF16数字格式:', '支持' if (torch.cuda.is_bf16_supported()) else '不支持')
print('当前显卡型号:', torch.cuda.get_device_name())
print('当前显卡的CUDA算力:', torch.cuda.get_device_capability())
print('当前显卡的总显存:', torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 'GB')
print('是否支持TensorCore:', '支持' if (torch.cuda.get_device_properties(0).major >= 7) else '不支持')
print('当前显卡的显存使用率:', torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100,
      '%')

# 清空缓存，固定随即种子
torch.manual_seed(1)
torch.cuda.empty_cache()

# 设置运行设备的环境为GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')

# 更改数据类型
training_data_input_xy = torch.from_numpy(training_data_input_xy).to(torch.float32).to(device)
training_data_input_white_line = torch.from_numpy(training_data_input_white_line).to(torch.float32).to(device)
training_data_input_lane = torch.from_numpy(training_data_input_lane).to(torch.float32).to(device)
training_data_output = torch.from_numpy(training_data_output).to(torch.float32).to(device)
testing_data_input_xy = torch.from_numpy(testing_data_input_xy).to(torch.float32).to(device)
testing_data_input_white_line = torch.from_numpy(testing_data_input_white_line).to(torch.float32).to(device)
testing_data_input_lane = torch.from_numpy(testing_data_input_lane).to(torch.float32).to(device)
testing_data_output = torch.from_numpy(testing_data_output).to(torch.float32).to(device)

print(f'training_data_input_xy: {training_data_input_xy.shape}')
print(f'training_data_input_white_line: {training_data_input_white_line.shape}')
print(f'training_data_input_lane: {training_data_input_lane.shape}')
print(f'training_data_output: {training_data_output.shape}')
print(f'testing_data_input_xy: {testing_data_input_xy.shape}')
print(f'testing_data_input_white_line: {testing_data_input_white_line.shape}')
print(f'testing_data_input_lane: {testing_data_input_lane.shape}')
print(f'testing_data_output: {testing_data_output.shape}')

# 模式选取
mode_switch = int(input("请进行模式选择："))
vector_map_switch = int(input("请进行是否启用vector_map："))
# mode_switch = 0
# vector_map_switch = 0

# 定义基本参数
size_basic = 256
size_encoder_fc_input = data_size - 1  # 减去index
size_encoder_fc_middle = size_basic
size_encoder_fc_output = size_basic
size_encoder_lstm_input = size_encoder_fc_output
size_encoder_lstm_hidden = size_basic
size_encoder_activate_num_parameters = input_size

size_decoder_lstm_input = size_encoder_lstm_hidden + 2 * row * column * vector_map_switch
size_decoder_lstm_hidden = size_basic
size_decoder_fc_input = size_decoder_lstm_hidden
size_decoder_fc_middle = size_basic
size_decoder_fc_output = 2

size_connector_fc_input = size_decoder_fc_output
size_connector_fc_middle = size_basic
size_connector_fc_output = size_encoder_lstm_hidden

learning_rate_init = 1e-4
learning_rate = learning_rate_init
max_epoch = 50
batch_ratio = 0.2


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, encoder_fc_input_size, encoder_fc_middle_size, encoder_fc_output_size,
                 encoder_lstm_input_size, encoder_lstm_hidden_size,
                 encoder_activate_num_parameters_size):
        super(Encoder, self).__init__()
        self.encoder_activate_init = 1.0
        self.encoder_lstm_hidden_size = encoder_lstm_hidden_size
        self.encoder_bias = True
        self.encoder_lstm_num_layers = 2

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

    def forward(self, x):
        h0 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        c0 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        h1 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        c1 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)

        out = self.encoder_fc1(x)
        out = self.encoder_fc2(out)
        out = self.encoder_fc3(out)
        out = self.encoder_fc4(out)
        out_front, (h_front, c_front) = self.encoder_lstm_front(out, (h0, c0))
        out_back, (h_back, c_back) = self.encoder_lstm_back(out.flip(dims=[1]), (h1, c1))
        h = torch.add(h_front, h_back)
        c = torch.add(c_front, c_back)
        out = torch.add(out_front, out_back)
        out = out[:, -1, :].unsqueeze(1)
        return out, (h, c)


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, decoder_lstm_input_size, decoder_lstm_hidden_size,
                 decoder_fc_input_size, decoder_fc_middle_size, decoder_fc_output_size):
        super(Decoder, self).__init__()
        self.decoder_lstm_hidden_size = decoder_lstm_hidden_size
        self.decoder_bias = True
        self.decoder_activate_init = 1.0
        self.decoder_lstm_num_layers = 2

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

    def forward(self, x, h1, c1):
        out, (h2, c2) = self.decoder_lstm(x, (h1, c1))
        out = self.decoder_fc1(out)
        out = self.decoder_fc2(out)
        out = self.decoder_fc3(out)
        out = self.decoder_fc4(out)
        return out, (h2, c2)


# 定义传递器
class Connector(nn.Module):
    def __init__(self, connector_fc_input_size, connector_fc_middle_size, connector_fc_output_size):
        super(Connector, self).__init__()
        self.connector_bias = True

        self.connector_fc1 = nn.Linear(connector_fc_input_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc2 = nn.Linear(connector_fc_middle_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc3 = nn.Linear(connector_fc_middle_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc4 = nn.Linear(connector_fc_middle_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc5 = nn.Linear(connector_fc_middle_size, connector_fc_output_size, bias=self.connector_bias)

    def forward(self, x):
        out = self.connector_fc1(x)
        out = self.connector_fc2(out)
        out = self.connector_fc3(out)
        out = self.connector_fc4(out)
        out = self.connector_fc5(out)
        return out


# 模型实例化
encoder = Encoder(size_encoder_fc_input, size_encoder_fc_middle, size_encoder_fc_output,
                  size_encoder_lstm_input, size_encoder_lstm_hidden, size_encoder_activate_num_parameters).to(device)
decoder = Decoder(size_decoder_lstm_input, size_decoder_lstm_hidden,
                  size_decoder_fc_input, size_decoder_fc_middle, size_decoder_fc_output).to(device)
connector = Connector(size_connector_fc_input, size_connector_fc_middle, size_connector_fc_output).to(device)

print(decoder)

# 优化器和损失函数
optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate)
optimizer_connector = optim.Adam(connector.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 主要部分
t_start = time.time()
fig = plt.figure()
if mode_switch == 0:
    print("进行单点模型训练")

    scheduler_encoder = scheduler.StepLR(optimizer_encoder, step_size=100, gamma=0.7, last_epoch=-1)
    scheduler_decoder = scheduler.StepLR(optimizer_decoder, step_size=100, gamma=0.7, last_epoch=-1)

    all_loss = np.zeros([1])
    for epoch in range(max_epoch):
        torch.cuda.empty_cache()
        encoded, (h_encoded, c_encoded) = encoder(training_data_input_xy)
        if vector_map_switch == 1:
            encoded = torch.cat([encoded, training_data_input_white_line, training_data_input_lane], 2)
        decoded, _ = decoder(encoded, h_encoded, c_encoded)
        loss = criterion(decoded, training_data_output[:, 0, :].unsqueeze(1))
        print('当前显卡的显存使用率:',
              torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100, '%')

        plt.clf()

        plt.subplot(2, 1, 1)
        all_loss[epoch] = loss.item()
        plt.plot(all_loss)
        plt.text(epoch / 2, (all_loss[:].max() + all_loss[:].min()) / 2,
                 f"lr:{learning_rate:.10f}", fontsize=10)

        plt.subplot(2, 1, 2)
        for i in range(20):
            plt.plot(np.append(training_data_output.cpu().detach().numpy()[i, 0, 0],
                               decoded.cpu().detach().numpy()[i, :, 0]),
                     np.append(training_data_output.cpu().detach().numpy()[i, 0, 1],
                               decoded.cpu().detach().numpy()[i, :, 1]))

        all_loss = np.append(all_loss, [0.0], axis=0)

        plt.pause(0.001)

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()

        if (epoch + 1) % 50 == 0:
            print(f"epoch:{epoch + 1},loss:{loss.item()}")

        optimizer_encoder.step()
        optimizer_decoder.step()

        # scheduler_encoder.step()
        # scheduler_decoder.step()
        # learning_rate = scheduler_encoder.get_last_lr()[0]

        rand_para = torch.randperm(training_data_input_xy.shape[0])
        training_data_input_xy = training_data_input_xy[rand_para]
        training_data_input_white_line = training_data_input_white_line[rand_para]
        training_data_input_lane = training_data_input_lane[rand_para]
        training_data_output = training_data_output[rand_para]

    torch.save(encoder, "end_encoder.pth")
    torch.save(decoder, "end_decoder.pth")
    fig.savefig("figs/" + "single.png")
if mode_switch == 1:
    print("单点模型测试")

    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")

    check_input_xy = training_data_input_xy
    check_input_white_line = training_data_input_white_line
    check_input_lane = training_data_input_lane
    check_output = training_data_output

    encoded, (h_encoded, c_encoded) = encoder(check_input_xy)
    if vector_map_switch == 1:
        encoded = torch.cat([encoded, check_input_white_line, check_input_lane], 2)
    decoded, _ = decoder(encoded, h_encoded, c_encoded)
    loss = criterion(decoded, check_output[:, 0, :].unsqueeze(1)) * check_output.shape[0]
    print(loss.item())

    for i in range(check_output.shape[0]):
        plt.plot(np.append(check_output.cpu().detach().numpy()[i, 0, 0],
                           decoded.cpu().detach().numpy()[i, :, 0]),
                 np.append(check_output.cpu().detach().numpy()[i, 0, 1],
                           decoded.cpu().detach().numpy()[i, :, 1]))
    plt.show()
if mode_switch == 0:
    print("进行连接模型训练")
    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")

    for points in range(40, training_data_output.shape[1], 1):
        learning_rate = learning_rate_init * points * 0.1
        optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate)
        optimizer_connector = optim.Adam(connector.parameters(), lr=learning_rate)
        scheduler_decoder = scheduler.StepLR(optimizer_decoder, step_size=100, gamma=0.7, last_epoch=-1)
        scheduler_connector = scheduler.StepLR(optimizer_connector, step_size=100, gamma=0.7, last_epoch=-1)

        all_loss = np.zeros([1])
        all_grad_abs = np.array([[0.0, 10]])
        for epoch in range(max_epoch):
            torch.cuda.empty_cache()
            batch_size = training_data_input_xy.shape[0] * batch_ratio
            for each_batch in range(int(1 / batch_ratio)):
                train_input_xy = training_data_input_xy[int(each_batch * batch_size):int((each_batch + 1) * batch_size), :, :]
                train_input_white_line = training_data_input_white_line[int(each_batch * batch_size):int((each_batch + 1) * batch_size), :, :]
                train_input_lane = training_data_input_lane[int(each_batch * batch_size):int((each_batch + 1) * batch_size), :, :]
                train_output = training_data_output[int(each_batch * batch_size):int((each_batch + 1) * batch_size), :, :]

                encoded, (h_encoded, c_encoded) = encoder(train_input_xy)
                if vector_map_switch == 1:
                    encoded = torch.cat([encoded, train_input_white_line, train_input_lane], 2)
                decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
                decoded_clone = decoded.clone()
                for point in range(points):
                    connected = connector(decoded)
                    encoded, h_encoded, c_encoded = connected, h_decoded, c_decoded
                    if vector_map_switch == 1:
                        encoded = torch.cat([encoded, train_input_white_line, train_input_lane], 2)
                    decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
                    decoded_clone = torch.cat((decoded_clone.clone(), decoded.clone()), 1)

                loss = criterion(decoded_clone, train_output[:, 0:(points + 1), :]) * decoded_clone.shape[0]
                all_loss[epoch] = loss.item()

                optimizer_connector.zero_grad()
                optimizer_decoder.zero_grad()
                loss.backward()
                optimizer_connector.step()
                optimizer_decoder.step()

            # print('当前显卡的显存使用率:',
            #       torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100, '%')

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

            plt.clf()
            show = 150
            cal = 100

            plt.subplot(3, 1, 1)
            plt.plot(all_loss)
            plt.text(epoch / 2, (all_loss[:].max() + all_loss[:].min()) / 2,
                     f"point:{points}, lr:{learning_rate:.10f}", fontsize=10)

            plt.subplot(3, 1, 2)
            plt.plot(all_grad_abs[:, 0])
            plt.plot([0.0, epoch], [0.0, 0.0], "r--")

            plt.subplot(3, 1, 3)
            plt.plot(all_grad_abs[:, 1])
            plt.plot([0.0, epoch], [0.0, 0.0], "r--")

            plt.pause(0.01)

            all_loss = np.append(all_loss, [0.0], axis=0)
            all_grad_abs = np.append(all_grad_abs, np.array([[0.0, 10]]), axis=0)

            if (epoch + 1) % 50 == 0:
                print(f"points:{points},epoch:{epoch + 1},loss:{all_loss[-2]}")

            # scheduler_decoder.step()
            # scheduler_connector.step()
            # learning_rate = scheduler_connector.get_last_lr()[0]

            rand_para = torch.randperm(training_data_input_xy.shape[0])
            training_data_input_xy = training_data_input_xy[rand_para]
            training_data_input_white_line = training_data_input_white_line[rand_para]
            training_data_input_lane = training_data_input_lane[rand_para]
            training_data_output = training_data_output[rand_para]

        fig.savefig("figs/" + str(points + 1) + ".png")

        torch.save(decoder, "end_decoder.pth")
        torch.save(connector, "end_connector.pth")
if mode_switch == 0:
    print("循环预测模型测试")
    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")
    connector = torch.load("end_connector.pth")

    check_input_xy = testing_data_input_xy
    check_input_white_line = testing_data_input_white_line
    check_input_lane = testing_data_input_lane
    check_output = testing_data_output

    encoded, (h_encoded, c_encoded) = encoder(check_input_xy)
    if vector_map_switch == 1:
        encoded = torch.cat([encoded, check_input_white_line, check_input_lane], 2)
    decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
    output = decoded.clone()

    for i in range(check_output.shape[1] - 1):
        connected = connector(decoded)
        encoded, h_encoded, c_encoded = connected, h_decoded, c_decoded
        if vector_map_switch == 1:
            encoded = torch.cat([encoded, check_input_white_line, check_input_lane], 2)
        decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
        output = torch.cat((output.clone(), decoded.clone()), 1)

    all_loss = np.zeros([1])
    for i in range(check_output.shape[0]):
        plt.subplot(1, 2, 1)
        lim = 20
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.plot(check_input_xy.cpu().detach().numpy()[i, :, 0], check_input_xy.cpu().detach().numpy()[i, :, 1])
        plt.plot(check_output.cpu().detach().numpy()[i, :, 0], check_output.cpu().detach().numpy()[i, :, 1])
        plt.plot(output.cpu().detach().numpy()[i, :, 0], output.cpu().detach().numpy()[i, :, 1], "*")

        for r in range(row):
            plt.plot([index_box[r, 0, 0], index_box[r, -1, 1]],
                     [index_box[r, 0, 2], index_box[r, -1, 2]], "g--")
            plt.plot([index_box[r, 0, 0], index_box[r, -1, 1]],
                     [index_box[r, 0, 3], index_box[r, -1, 3]], "g--")
        for c in range(column):
            plt.plot([index_box[0, c, 0], index_box[-1, c, 0]],
                     [index_box[0, c, 2], index_box[-1, c, 3]], "g--")
            plt.plot([index_box[0, c, 1], index_box[-1, c, 1]],
                     [index_box[0, c, 2], index_box[-1, c, 3]], "g--")

        plt.subplot(1, 2, 2)
        loss = criterion(output[i], check_output[i])
        all_loss[-1] = loss.item()
        plt.plot(all_loss)
        all_loss = np.append(all_loss, [0.0], axis=0)

        if (i+1) % 50 == 0:
            fig.savefig("../result/10/" + str(i) + ".png")
        plt.pause(0.01)
        plt.clf()
print(f"本次程序运行时间为：{int((time.time() - t_start) / 60)} min, {(time.time() - t_start) % 60}s")

