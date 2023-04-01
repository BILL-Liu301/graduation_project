import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import training_data_input, training_data_output, testing_data_input, testing_data_output
from utils import data_size, input_size
import matplotlib.pyplot as plt

# 清空缓存，固定随即种子
torch.manual_seed(1)
torch.cuda.empty_cache()

# 设置运行设备的环境为GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')

# 更改数据类型
training_data_input = torch.from_numpy(training_data_input).to(torch.float32).to(device)
training_data_output = torch.from_numpy(training_data_output).to(torch.float32).to(device)
testing_data_input = torch.from_numpy(testing_data_input).to(torch.float32).to(device)
testing_data_output = torch.from_numpy(testing_data_output).to(torch.float32).to(device)

print(f'training_data_input: {training_data_input.shape}')
print(f'training_data_output: {training_data_output.shape}')
print(f'testing_data_input: {testing_data_input.shape}')
print(f'testing_data_output: {testing_data_output.shape}')

# 定义基本参数
size_basic = 128
size_encoder_fc_input = data_size - 1  # 减去index
size_encoder_fc_middle = size_basic
size_encoder_fc_output = size_basic
size_encoder_lstm_input = size_encoder_fc_output
size_encoder_lstm_hidden = size_basic
size_encoder_activate_num_parameters = input_size

size_decoder_lstm_input = size_encoder_lstm_hidden
size_decoder_lstm_hidden = size_basic
size_decoder_fc_input = size_decoder_lstm_hidden
size_decoder_fc_middle = size_basic
size_decoder_fc_output = 2

size_connector_fc_input = size_decoder_fc_output
size_connector_fc_middle = size_basic
size_connector_fc_output = size_decoder_lstm_input

learning_rate = 1e-4
max_epoch = 50000


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, encoder_fc_input_size, encoder_fc_middle_size, encoder_fc_output_size,
                 encoder_lstm_input_size, encoder_lstm_hidden_size,
                 encoder_activate_num_parameters_size):
        super(Encoder, self).__init__()
        self.encoder_activate_init = 1.0
        self.encoder_lstm_hidden_size = encoder_lstm_hidden_size
        self.encoder_bias = True

        self.encoder_bias = False
        self.encoder_fc1 = nn.Linear(encoder_fc_input_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc2 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc3 = nn.Linear(encoder_fc_middle_size, encoder_fc_output_size, bias=self.encoder_bias)
        self.encoder_lstm = nn.LSTM(encoder_lstm_input_size, encoder_lstm_hidden_size, num_layers=1, batch_first=True)
        self.encoder_activate1 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        self.encoder_activate2 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        self.encoder_activate3 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)

    def forward(self, x):
        h0 = torch.ones(1, x.size(0), self.encoder_lstm_hidden_size).to(device)
        c0 = torch.ones(1, x.size(0), self.encoder_lstm_hidden_size).to(device)

        # out = self.encoder_fc1(self.encoder_activate1(x))
        # out = self.encoder_fc2(self.encoder_activate2(out))
        # out = self.encoder_fc3(self.encoder_activate2(out))
        out = self.encoder_fc1(x)
        out = self.encoder_fc2(out)
        out = self.encoder_fc3(out)
        out, (h1, c1) = self.encoder_lstm(out, (h0, c0))
        out = out[:, -1, :].unsqueeze(1)
        return out, (h1, c1)


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, decoder_lstm_input_size, decoder_lstm_hidden_size,
                 decoder_fc_input_size, decoder_fc_middle_size, decoder_fc_output_size):
        super(Decoder, self).__init__()
        self.decoder_lstm_hidden_size = decoder_lstm_hidden_size
        self.decoder_bias = True
        self.decoder_activate_init = 1.0

        self.decoder_lstm = nn.LSTM(decoder_lstm_input_size, decoder_lstm_hidden_size, num_layers=1, batch_first=True)
        self.decoder_fc1 = nn.Linear(decoder_fc_input_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc2 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc3 = nn.Linear(decoder_fc_middle_size, decoder_fc_output_size, bias=self.decoder_bias)
        self.decoder_activate1 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_activate2 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_activate3 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)

    def forward(self, x, h1, c1):
        out, (h2, c2) = self.decoder_lstm(x, (h1, c1))
        # out = self.decoder_fc1(self.decoder_activate1(out))
        # out = self.decoder_fc2(self.decoder_activate2(out))
        # out = self.decoder_fc3(self.decoder_activate2(out))
        out = self.decoder_fc1(out)
        out = self.decoder_fc2(out)
        out = self.decoder_fc3(out)
        return out, (h2, c2)


# 定义传递器
class Connector(nn.Module):
    def __init__(self, connector_fc_input_size, connector_fc_middle_size, connector_fc_output_size):
        super(Connector, self).__init__()
        self.connector_bias = True

        self.connector_fc1 = nn.Linear(connector_fc_input_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc2 = nn.Linear(connector_fc_middle_size, connector_fc_middle_size, bias=self.connector_bias)
        self.connector_fc3 = nn.Linear(connector_fc_middle_size, connector_fc_output_size, bias=self.connector_bias)

    def forward(self, x):
        out = self.connector_fc1(x)
        out = self.connector_fc2(out)
        out = self.connector_fc3(out)
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
criterion = nn.L1Loss()

# 模式选取
mode_switch = 2

# 主要部分
if mode_switch == 0:
    print("进行单点模型训练")
    plt.figure()
    all_loss = np.zeros([1])
    for epoch in range(max_epoch):
        encoded, (h_encoded, c_encoded) = encoder(training_data_input)
        decoded, _ = decoder(encoded, h_encoded, c_encoded)
        loss = criterion(decoded, training_data_output) * training_data_output.shape[0]

        plt.clf()
        show = 150
        cal = 100

        plt.subplot(1, 2, 1)
        all_loss[epoch] = loss.item()
        plt.plot(all_loss[max([epoch - show, 0]):(epoch + 1)])
        plt.plot([show, show], [all_loss[max([epoch - show, 0]):(epoch + 1)].min(),
                                all_loss[max([epoch - show, 0]):(epoch + 1)].max()], "r--")
        plt.plot([show - cal, show - cal], [all_loss[max([epoch - show, 0]):(epoch + 1)].min(),
                                            all_loss[max([epoch - show, 0]):(epoch + 1)].max()], "r--")

        plt.subplot(1, 2, 2)
        for i in range(training_data_output.shape[0]):
            plt.plot(np.append(training_data_output.cpu().detach().numpy()[i, :, 0],
                               decoded.cpu().detach().numpy()[i, :, 0]),
                     np.append(training_data_output.cpu().detach().numpy()[i, :, 1],
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

        rand_para = torch.randperm(training_data_input.shape[0])
        training_data_input = training_data_input[rand_para]
        training_data_output = training_data_output[rand_para]

    torch.save(encoder, "end_encoder.pth")
    torch.save(decoder, "end_decoder.pth")
    plt.show()
elif mode_switch == 1:
    print("单点模型测试")
    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")

    encoded, (h_encoded, c_encoded) = encoder(testing_data_input)
    decoded, _ = decoder(encoded, h_encoded, c_encoded)
    loss = criterion(decoded, testing_data_output) * testing_data_output.shape[0]
    print(loss.item())

    plt.figure()
    for i in range(testing_data_output.shape[0]):
        plt.plot(np.append(testing_data_output.cpu().detach().numpy()[i, :, 0],
                           decoded.cpu().detach().numpy()[i, :, 0]),
                 np.append(testing_data_output.cpu().detach().numpy()[i, :, 1],
                           decoded.cpu().detach().numpy()[i, :, 1]))
    plt.show()
elif mode_switch == 2:
    print("进行连接模型训练")
    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")

    plt.figure()
    all_loss = np.zeros([1])
    for points in range(1, training_data_output.shape[1], 1):
        for epoch in range(max_epoch):
            encoded, (h_encoded, c_encoded) = encoder(training_data_input)
            decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
            for point in range(points):
                connected = connector(decoded)
                encoded, h_encoded, c_encoded = connected, h_decoded, c_decoded
                decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
            loss = criterion(decoded, training_data_output[:, points, :].unsqueeze(1)) * training_data_output.shape[0]

            plt.clf()
            show = 150
            cal = 100

            plt.subplot(1, 2, 1)
            all_loss[epoch] = loss.item()
            plt.plot(all_loss[max([epoch - show, 0]):(epoch + 1)])
            plt.plot([show, show], [all_loss[max([epoch - show, 0]):(epoch + 1)].min(),
                                    all_loss[max([epoch - show, 0]):(epoch + 1)].max()], "r--")
            plt.plot([show - cal, show - cal], [all_loss[max([epoch - show, 0]):(epoch + 1)].min(),
                                                all_loss[max([epoch - show, 0]):(epoch + 1)].max()], "r--")

            plt.subplot(1, 2, 2)
            for i in range(training_data_output.shape[0]):
                plt.plot(np.append(training_data_output.cpu().detach().numpy()[i, points, 0],
                                   decoded.cpu().detach().numpy()[i, :, 0]),
                         np.append(training_data_output.cpu().detach().numpy()[i, points, 1],
                                   decoded.cpu().detach().numpy()[i, :, 1]))

            all_loss = np.append(all_loss, [0.0], axis=0)

            plt.pause(0.001)

            optimizer_connector.zero_grad()
            loss.backward()

            if (epoch + 1) % 50 == 0:
                print(f"points:{points},epoch:{epoch + 1},loss:{loss.item()}")

            optimizer_connector.step()

            rand_para = torch.randperm(training_data_input.shape[0])
            training_data_input = training_data_input[rand_para]
            training_data_output = training_data_output[rand_para]

    torch.save(connector, "end_connector.pth")
    plt.show()
elif mode_switch == 3:
    print("循环预测模型测试")
    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")
    connector = torch.load("end_connector.pth")

    check_input = training_data_input
    check_output = training_data_output
    output = np.zeros(check_output.shape)

    encoded, (h_encoded, c_encoded) = encoder(training_data_input)
    for i in range(output.shape[1]):
        decoded, (h_decoded, c_decoded) = decoder(encoded, h_encoded, c_encoded)
        connected = connector(decoded)
        encoded, h_encoded, c_encoded = connected, h_decoded, c_decoded
        for j in range(output.shape[0]):
            output[j, i, 0] = decoded[j, 0, 0]
            output[j, i, 1] = decoded[j, 0, 1]

    plt.figure()
    for i in range(output.shape[0]):
        plt.clf()
        plt.plot(output[i, :, 0], output[i, :, 0], "*")
        plt.plot(check_input.cpu().detach().numpy()[i, :, 0], check_input.cpu().detach().numpy()[i, :, 1])
        plt.plot(check_output.cpu().detach().numpy()[i, :, 0], check_output.cpu().detach().numpy()[i, :, 1])
        plt.pause(0.01)


