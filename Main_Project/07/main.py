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
learning_rate_encoder = 1e-2
learning_rate_decoder = 1e-2

# 更改数据类型
training_data_input = torch.from_numpy(training_data_input).to(torch.float32).to(device)
training_data_output = torch.from_numpy(training_data_output).to(torch.float32).to(device)
testing_data_input = torch.from_numpy(testing_data_input).to(torch.float32).to(device)
testing_data_output = torch.from_numpy(testing_data_output).to(torch.float32).to(device)
print(f'training_data_input: {training_data_input.shape}')
print(f'training_data_output: {training_data_output.shape}')
print(f'testing_data_input: {testing_data_input.shape}')
print(f'testing_data_output: {testing_data_output.shape}')


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
        self.decoder_softmax = nn.Softmax(dim=1)
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

    def beam_search(self, x, decoded, delta):
        _, index = torch.sort(x[:, 0, :], descending=True)
        index = index[:, 0:self.decoder_K]
        index_w = torch.zeros([x.shape[0], self.decoder_K, self.decoder_Qw])
        index_l = torch.zeros([x.shape[0], self.decoder_K, self.decoder_Ql])

        for i in range(self.decoder_K):
            fw = self.find_w(index[:, i])
            fl = self.find_l(index[:, i])
            for j in range(index.shape[0]):
                index_w[j, i, int(fw[j])] = 1
                index_l[j, i, int(fl[j])] = 1
                decoded[j, delta, i, 0] = fl[j]
                decoded[j, delta, i, 1] = fw[j]

        return index_w, index_l, decoded

    def embedding(self, index_w, index_l):
        return torch.cat([index_w, index_l], 1)

    def once(self, x, h1, c1, h2, c2):
        x, (h1, c1) = self.decoder_lstm1(x, (h1, c1))
        x, (h2, c2) = self.decoder_lstm2(x, (h2, c2))
        x = self.decoder_fc1(x)
        x = self.decoder_fc2(x)
        x = self.decoder_fc3(x)
        # x = self.decoder_softmax(x)
        return x, h1, c1, h2, c2

    def encoder(self, x):
        h0 = torch.zeros(1, x.size(0), self.encoder_hidden_size_lstm).to(device)
        c0 = torch.zeros(1, x.size(0), self.encoder_hidden_size_lstm).to(device)

        encoded = self.encoder_fc1(x)
        encoded = self.encoder_fc2(encoded)
        encoded = self.encoder_fc3(encoded)
        encoded, (h1, c1) = self.encoder_lstm1(encoded, (h0, c0))
        encoded, (h2, c2) = self.encoder_lstm2(encoded, (h0, c0))
        encoded = encoded[:, -1, :].reshape([encoded.shape[0], 1, encoded.shape[2]])

        return encoded, h1, c1, h2, c2

    def decoder(self, x, h1, c1, h2, c2):
        x, h1, c1, h2, c2 = self.once(x, h1, c1, h2, c2)

        decoded = torch.zeros([x.shape[0], self.decoder_delta, self.decoder_K, 2]).to(device)
        index_w, index_l, decoded = self.beam_search(x, decoded, 0)

        index_w = self.decoder_fc_qw(index_w.to(device))
        index_l = self.decoder_fc_ql(index_l.to(device))
        lstm_init = torch.zeros([self.decoder_K, 4, h1.shape[0], h1.shape[1], h1.shape[2]]).to(device)
        for i in range(self.decoder_K):
            lstm_init[i, 0] = h1
            lstm_init[i, 1] = c1
            lstm_init[i, 2] = h2
            lstm_init[i, 3] = c2

        self.decoder_K = 1
        for i in range(self.decoder_delta - 1):
            for j in range(self.decoder_K):
                embedded = self.embedding(index_w[:, j, :], index_l[:, j, :]).unsqueeze(1).to(device)
                embedded, lstm_init[j, 0], lstm_init[j, 1], lstm_init[j, 2], lstm_init[j, 3] = \
                    self.once(embedded, lstm_init[j, 0], lstm_init[j, 1], lstm_init[j, 2], lstm_init[j, 3])
                index_w, index_l, decoded = self.beam_search(embedded, decoded, i + 1)
                index_w = self.decoder_fc_qw(index_w.to(device))
                index_l = self.decoder_fc_ql(index_l.to(device))
        return decoded

    def forward(self, x):
        encoded, h1, c1, h2, c2 = self.encoder(x)
        decoded = self.decoder(encoded, h1, c1, h2, c2)
        return decoded


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

    def beam_search(self, x, trajectory_points, delta):
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
                trajectory_points[j, delta, i, 0] = fl[j]
                trajectory_points[j, delta, i, 1] = fw[j]

        return index_w, index_l, trajectory_points

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

    def forward(self, x, h1, c1, h2, c2):
        x, h1, c1, h2, c2 = self.once(x, h1, c1, h2, c2)

        trajectory_points = torch.zeros([x.shape[0], self.delta, self.K, 2])
        index_w, index_l, trajectory_points = self.beam_search(x, trajectory_points, 0)

        index_w = self.fc_qw(index_w.to(device))
        index_l = self.fc_ql(index_l.to(device))
        lstm_init = torch.zeros([self.K, 4, h1.shape[0], h1.shape[1], h1.shape[2]]).to(device)
        for i in range(self.K):
            lstm_init[i, 0] = h1
            lstm_init[i, 1] = c1
            lstm_init[i, 2] = h2
            lstm_init[i, 3] = c2

        self.K = 1
        for i in range(self.delta - 1):
            for j in range(self.K):
                embedded = self.embedding(index_w[:, j, :], index_l[:, j, :]).unsqueeze(1).to(device)
                embedded, lstm_init[j, 0], lstm_init[j, 1], lstm_init[j, 2], lstm_init[j, 3] = self.once(embedded,
                                                                                                         lstm_init[
                                                                                                             j, 0],
                                                                                                         lstm_init[
                                                                                                             j, 1],
                                                                                                         lstm_init[
                                                                                                             j, 2],
                                                                                                         lstm_init[
                                                                                                             j, 3])
                index_w, index_l, trajectory_points = self.beam_search(embedded, trajectory_points, i + 1)
                index_w = self.fc_qw(index_w.to(device))
                index_l = self.fc_ql(index_l.to(device))
        return trajectory_points


# 设置模型基本参数
model_predict = EncoderDecoder(size_encoder_input, size_encoder_hidden_fc, size_encoder_hidden_lstm,
                               size_encoder_output_lstm,
                               size_decoder_input, size_decoder_hidden_fc, size_decoder_hidden_lstm,
                               size_decoder_output_fc,
                               size_K, Qw, Ql, size_delta).to(device)
trajectory_points = model_predict.forward(training_data_input)
