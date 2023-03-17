import math

from utils import training_data_input, training_data_output, testing_data_input, testing_data_output, Qw, Ql
import numpy as np
import torch
import torch.nn as nn

# 清空缓存
torch.cuda.empty_cache()

# 设置运行设备的环境为GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')
# device = torch.device('cpu')
# print(f'本次程序运行的设备环境为{device}')

# 可调参数
size_basic = 256
size_encoder_input = 5
size_encoder_hidden_fc = size_basic
size_encoder_hidden_lstm = size_basic
size_encoder_output_lstm = size_basic
size_decoder_input = size_basic
size_decoder_hidden_lstm = size_basic
size_decoder_hidden_fc = size_basic
size_decoder_output_fc = Ql * Qw
size_K = 4
size_delta = training_data_output.shape[1]

# 更改数据类型
training_data_input = torch.from_numpy(training_data_input).to(torch.float32).to(device)
training_data_output = torch.from_numpy(training_data_output).to(torch.float32).to(device)
testing_data_input = torch.from_numpy(testing_data_input).to(torch.float32).to(device)
testing_data_output = torch.from_numpy(testing_data_output).to(torch.float32).to(device)
print(f'training_data_input: {training_data_input.shape}')
print(f'training_data_output: {training_data_output.shape}')
print(f'testing_data_input: {testing_data_input.shape}')
print(f'testing_data_output: {testing_data_output.shape}')


# 定义LSTM模型——编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size_fc, hidden_size_lstm, output_size_lstm):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size_lstm
        self.fc1 = nn.Linear(input_size, hidden_size_fc)
        self.fc2 = nn.Linear(hidden_size_fc, hidden_size_fc)
        self.fc3 = nn.Linear(hidden_size_fc, hidden_size_fc)
        self.lstm1 = nn.LSTM(hidden_size_fc, hidden_size_lstm, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_fc, output_size_lstm, num_layers=1, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)

        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out, (h1, c1) = self.lstm1(out, (h0, c0))
        out, (h2, c2) = self.lstm2(out, (h0, c0))
        out = out[:, -1, :].reshape([out.shape[0], 1, out.shape[2]])

        return out, (h1, c1), (h2, c2)


# 定义LSTM模型——解码器
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size_fc, hidden_size_lstm, output_size_fc, k, qw, ql, delta):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size_lstm, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_lstm, hidden_size_lstm, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size_lstm, hidden_size_fc)
        self.fc2 = nn.Linear(hidden_size_fc, hidden_size_fc)
        self.fc3 = nn.Linear(hidden_size_fc, output_size_fc)
        self.softmax = nn.Softmax(dim=1)
        self.fc_ql = nn.Linear(ql, int(input_size / 2))
        self.fc_qw = nn.Linear(qw, int(input_size / 2))
        self.K = k
        self.Qw = qw
        self.Ql = ql
        self.delta = delta

    def find_w(self, index):
        return (index / self.Ql).floor()

    def find_l(self, index):
        return index % self.Ql

    def beam_search(self, x):
        _, index = torch.sort(x[:, 0, :], descending=True)
        index = index[:, 0:self.K]
        # index_w = torch.from_numpy(np.zeros([x.shape[0], self.K, self.Qw])).to(torch.float32).to(device)
        # index_l = torch.from_numpy(np.zeros([x.shape[0], self.K, self.Ql])).to(torch.float32).to(device)
        index_w = torch.zeros([x.shape[0], self.K, self.Qw])
        index_l = torch.zeros([x.shape[0], self.K, self.Ql])
        for i in range(self.K):
            fw = self.find_w(index[:, i])
            fl = self.find_l(index[:, i])
            for j in range(index.shape[0]):
                index_w[j, i, int(fw[j])] = 1
                index_l[j, i, int(fl[j])] = 1
        return index_w, index_l

    def embedding(self, index_w, index_l):
        return torch.cat([index_w, index_l], 1)

    def once(self, x, h1, c1, h2, c2):
        x, (h1, c1) = self.lstm1(x, (h1, c1))
        x, (h2, c2) = self.lstm2(x, (h2, c2))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x, h1, c1, h2, c2

    def forward(self,x, h1, c1, h2, c2):
        # embedded = torch.tensor([x.shape[0], self.K, x.shape[2]])
        x, h1, c1, h2, c2 = self.once(x, h1, c1, h2, c2)
        index_w, index_l = self.beam_search(x)
        index_w = self.fc_qw(index_w.to(device))
        index_l = self.fc_ql(index_l.to(device))
        trajectory_points = torch.zeros([x.shape[0], self.delta, self.K, 2])
        lstm_init = torch.zeros([self.K, 4, h1.shape[0], h1.shape[1], h1.shape[2]])
        for i in range(self.K):
            lstm_init[i, 0] = h1
            lstm_init[i, 1] = c1
            lstm_init[i, 2] = h2
            lstm_init[i, 3] = c2
        lstm_init.to(device)
        print(index_w.shape, index_l.shape)
        
        for i in range(self.delta):
            for j in range(self.K):
                embedded = self.embedding(index_w[:, j, :], index_l[:, j, :]).unsqueeze(1).to(device)
                embedded, _, _, _, _ = self.once(embedded,
                                                 lstm_init[j, 0], lstm_init[j, 1], lstm_init[j, 2], lstm_init[j, 3])
                print()
        return x


# 设置模型基本参数
encoder = Encoder(size_encoder_input, size_encoder_hidden_fc, size_encoder_hidden_lstm, size_encoder_output_lstm).to(
    device)
decoder = Decoder(size_decoder_input, size_decoder_hidden_fc, size_decoder_hidden_lstm, size_decoder_output_fc,
                  size_K, Qw, Ql, size_delta).to(device)

print(encoder, decoder)

out, (h1, c1), (h2, c2) = encoder(training_data_input)
print(out.shape)
out = decoder.forward(out, h1, c1, h2, c2)
print(out.shape)
