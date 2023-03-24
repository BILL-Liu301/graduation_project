import math
from utils import training_data_input, training_data_output, testing_data_input, testing_data_output, Qw, Ql
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gc

torch.cuda.empty_cache()
gc.collect()

# 定义tensorboard的writer
tensorboard_writer = SummaryWriter("../runs/07")

# 清空缓存
torch.cuda.empty_cache()

# 设置运行设备的环境为GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')

# 设定工作模式
# training_or_testing = int(input("请输入模式选择，0代表训练，1代表只作测试："))  # 0:训练 1:测试
# if training_or_testing == 0:
#     print("本次程序将会进行训练，并测试模型")
# else:
#     print("本次程序只作测试")

# 可调参数
size_basic = 128
size_encoder_input = 5
size_encoder_hidden_fc = size_basic
size_encoder_hidden_lstm = size_basic
size_encoder_output_lstm = size_basic
size_decoder_input = size_basic
size_decoder_hidden_lstm = size_basic
size_decoder_hidden_fc = size_basic
size_decoder_output_fc = Ql * Qw
size_transition_input = Ql * Qw
size_transition_hidden_fc = size_basic
size_transition_output_fc = size_decoder_input

size_K = 4
size_delta = training_data_output.shape[1]
learning_rate = 1e-6
max_epochs = 500000

# 更改数据类型
training_data_input = torch.from_numpy(training_data_input).to(torch.float32).to(device)
training_data_output = torch.from_numpy(training_data_output).to(torch.float32).to(device)
testing_data_input = torch.from_numpy(testing_data_input).to(torch.float32).to(device)
testing_data_output = torch.from_numpy(testing_data_output).to(torch.float32).to(device)
print(f'training_data_input: {training_data_input.shape}')
print(f'training_data_output: {training_data_output.shape}')
print(f'testing_data_input: {testing_data_input.shape}')
print(f'testing_data_output: {testing_data_output.shape}')


# 编码器
class Encoder(nn.Module):
    def __init__(self, encoder_input_size, encoder_hidden_size_fc, encoder_hidden_size_lstm, encoder_output_size_lstm):
        super(Encoder, self).__init__()
        self.encoder_hidden_size_lstm = encoder_hidden_size_lstm
        self.encoder_fc1 = nn.Linear(encoder_input_size, encoder_hidden_size_fc)
        self.encoder_fc2 = nn.Linear(encoder_hidden_size_fc, encoder_hidden_size_fc)
        self.encoder_fc3 = nn.Linear(encoder_hidden_size_fc, encoder_hidden_size_fc)
        self.encoder_lstm1 = nn.LSTM(encoder_hidden_size_fc, encoder_hidden_size_lstm, num_layers=1, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(encoder_hidden_size_fc, encoder_output_size_lstm, num_layers=1, batch_first=True)

    def encoder(self, x):
        h0 = torch.zeros(1, x.size(0), self.encoder_hidden_size_lstm).to(device)
        c0 = torch.zeros(1, x.size(0), self.encoder_hidden_size_lstm).to(device)

        y = self.encoder_fc1(x)
        y = self.encoder_fc2(y)
        y = self.encoder_fc3(y)
        y, (h1, c1) = self.encoder_lstm1(y, (h0, c0))
        y, (h2, c2) = self.encoder_lstm2(y, (h0, c0))
        y = y[:, -1, :].unsqueeze(1)

        return y, (h1, c1), (h2, c2)


# 解码器
class Decoder(nn.Module):
    def __init__(self, decoder_input_size, decoder_hidden_size_fc, decoder_hidden_size_lstm, decoder_output_size_fc):
        super(Decoder, self).__init__()
        self.decoder_lstm1 = nn.LSTM(decoder_input_size, decoder_hidden_size_lstm, num_layers=1, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(decoder_hidden_size_lstm, decoder_hidden_size_lstm, num_layers=1, batch_first=True)
        self.decoder_fc1 = nn.Linear(decoder_hidden_size_lstm, decoder_hidden_size_fc)
        self.decoder_fc2 = nn.Linear(decoder_hidden_size_fc, decoder_hidden_size_fc)
        self.decoder_fc3 = nn.Linear(decoder_hidden_size_fc, decoder_output_size_fc)
        self.decoder_softmax = nn.Softmax(dim=2)
        self.decoder_fc4 = nn.Linear(decoder_output_size_fc, decoder_input_size)
        self.decoder_fc5 = nn.Linear(decoder_input_size, decoder_input_size)

    def decoder(self, x, h1, c1, h2, c2):
        y, (h1, c1) = self.decoder_lstm1(x, (h1, c1))
        y, (h2, c2) = self.decoder_lstm2(y, (h2, c2))
        y = self.decoder_fc1(y)
        y = self.decoder_fc2(y)
        y = self.decoder_fc3(y)
        # y = self.decoder_softmax(y)
        y = y.squeeze(1)

        return y, (h1, c1), (h2, c2)


# 过渡器
class Transition(nn.Module):
    def __init__(self, transition_input_size, transition_hidden_size_fc, transition_output_size_fc):
        super(Transition, self).__init__()
        self.transition_fc1 = nn.Linear(transition_input_size, transition_hidden_size_fc)
        self.transition_fc2 = nn.Linear(transition_hidden_size_fc, transition_output_size_fc)

    def transition(self, x):
        y = self.transition_fc1(x)
        y = self.transition_fc2(y)
        return y


# 整合编码器和解码器
class EncoderDecoder(nn.Module):
    def __init__(self, encoder_input_size, encoder_hidden_size_fc, encoder_hidden_size_lstm, encoder_output_size_lstm,
                 decoder_input_size, decoder_hidden_size_fc, decoder_hidden_size_lstm, decoder_output_size_fc,
                 decoder_k, decoder_qw, decoder_ql, decoder_delta):
        super(EncoderDecoder, self).__init__()
        self.encoder_hidden_size_lstm = encoder_hidden_size_lstm
        self.encoder_fc1 = nn.Linear(encoder_input_size, encoder_hidden_size_fc)
        self.encoder_fc2 = nn.Linear(encoder_hidden_size_fc, encoder_hidden_size_fc)
        self.encoder_fc3 = nn.Linear(encoder_hidden_size_fc, encoder_hidden_size_fc)
        self.encoder_lstm1 = nn.LSTM(encoder_hidden_size_fc, encoder_hidden_size_lstm, num_layers=1, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(encoder_hidden_size_fc, encoder_output_size_lstm, num_layers=1, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(decoder_input_size, decoder_hidden_size_lstm, num_layers=1, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(decoder_hidden_size_lstm, decoder_hidden_size_lstm, num_layers=1, batch_first=True)
        self.decoder_fc1 = nn.Linear(decoder_hidden_size_lstm, decoder_hidden_size_fc)
        self.decoder_fc2 = nn.Linear(decoder_hidden_size_fc, decoder_hidden_size_fc)
        self.decoder_fc3 = nn.Linear(decoder_hidden_size_fc, decoder_output_size_fc)
        self.decoder_softmax = nn.Softmax(dim=2)
        self.decoder_fc4 = nn.Linear(decoder_output_size_fc, decoder_input_size)
        self.decoder_fc5 = nn.Linear(decoder_input_size, decoder_input_size)

        self.decoder_fc_ql = nn.Linear(decoder_ql, int(decoder_input_size / 2))
        self.decoder_fc_qw = nn.Linear(decoder_qw, int(decoder_input_size / 2))
        self.decoder_K = decoder_k
        self.decoder_Qw = decoder_qw
        self.decoder_Ql = decoder_ql
        self.decoder_delta = decoder_delta

    def find_w(self, index):
        return (index / self.decoder_Ql).floor()

    def find_l(self, index):
        return index % self.decoder_Ql

    def encoder(self, x):
        h0 = torch.zeros(1, x.size(0), self.encoder_hidden_size_lstm).to(device)
        c0 = torch.zeros(1, x.size(0), self.encoder_hidden_size_lstm).to(device)

        encoded = self.encoder_fc1(x)
        encoded = self.encoder_fc2(encoded)
        encoded = self.encoder_fc3(encoded)
        encoded, (h1, c1) = self.encoder_lstm1(encoded, (h0, c0))
        encoded, (h2, c2) = self.encoder_lstm2(encoded, (h0, c0))
        encoded = encoded[:, -1, :].unsqueeze(1)

        return encoded, (h1, c1), (h2, c2)

    def decoder(self, x, h1, c1, h2, c2):
        decoded = torch.zeros(x.shape[0], self.decoder_delta, self.decoder_Qw * self.decoder_Ql, requires_grad=True).to(
            device)
        for i in range(self.decoder_delta):
            z, (h1, c1) = self.decoder_lstm1(x, (h1, c1))
            z, (h2, c2) = self.decoder_lstm2(z, (h2, c2))
            z = self.decoder_fc1(z)
            z = self.decoder_fc2(z)
            z = self.decoder_fc3(z)
            z = self.decoder_softmax(z)
            decoded[:, i, :] = z[:, 0, :].clone()
            z = self.decoder_fc4(z)
            z = self.decoder_fc5(z)
            x = z.clone()

        return z, decoded

    def forward(self, x):
        encoded, (h1, c1), (h2, c2) = self.encoder(x)
        decoded, average = self.decoder(encoded, h1, c1, h2, c2)
        return decoded, average


# 设置模型基本参数
encoder = Encoder(size_encoder_input, size_encoder_hidden_fc, size_encoder_hidden_lstm, size_encoder_output_lstm).to(device)
decoder = Decoder(size_decoder_input, size_decoder_hidden_fc, size_decoder_hidden_lstm, size_decoder_output_fc).to(device)
transition = Transition(size_transition_input, size_transition_hidden_fc, size_transition_output_fc)

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
optimizer_transition = torch.optim.Adam(transition.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
# 单点训练
for epoch in range(max_epochs):
    encoded, (h1, c1), (h2, c2) = encoder.encoder(training_data_input)
    decoded, (h1, c1), (h2, c2) = decoder.decoder(encoded, h1, c1, h2, c2)
    loss = criterion(decoded, training_data_output)
    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()
    loss.backward()
    if (epoch+1) % 100 == 0:
        print(f"epoch:{epoch+1},loss:{loss.item()}")
        tensorboard_writer.add_scalar("loss", loss.item(), epoch)
    optimizer_encoder.step()
    optimizer_decoder.step()
tensorboard_writer.close()





