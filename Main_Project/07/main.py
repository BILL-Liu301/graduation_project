from utils import training_data_input, training_data_output, testing_data_input, testing_data_output, Qw, Ql
import numpy as np
import torch
import torch.nn as nn

# 设置运行设备的环境为GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')
# device = torch.device('cpu')
# print(f'本次程序运行的设备环境为{device}')

# 可调参数
size_encoder_input = 5
size_encoder_hidden_fc = 256
size_encoder_hidden_lstm = 256
size_encoder_output_lstm = 256
size_decoder_input = 256
size_decoder_hidden_lstm = 256
size_decoder_hidden_fc = 256
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
        return index // self.Ql

    def find_l(self, index):
        return index % self.Ql

    def beam_search(self, x):
        index = x.argsort()[-self.K:]
        index_w = np.zeros([self.K, self.Qw])
        index_l = np.zeros([self.K, self.Ql])
        for i in range(self.K):
            index_w[i, self.find_w(index[i])] = 1
            index_l[i, self.find_l(index[i])] = 1
        return index_w, index_l

    def embedding(self, index_w, index_l):
        return torch.cat([index_w, index_l], 1)

    def forward(self, x, h1, c1, h2, c2):
        for i in range(self.delta):
            x, _ = self.lstm1(x, (h1, c1))
            x, _ = self.lstm2(x, (h2, c2))
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.softmax(x)
            index_w, index_l = self.beam_search(out)
            index_w = self.fc_qw(index_w)
            index_l = self.fc_ql(index_l)
            x = self.embedding(index_w, index_l)
        return x


# 设置模型基本参数
encoder = Encoder(size_encoder_input, size_encoder_hidden_fc, size_encoder_hidden_lstm, size_encoder_output_lstm).to(
    device)
decoder = Decoder(size_decoder_input, size_decoder_hidden_fc, size_decoder_hidden_lstm, size_decoder_output_fc,
                  size_K, Qw, Ql, size_delta).to(device)

out, (h1, c1), (h2, c2) = encoder(training_data_input)
print(out.shape)
out = decoder(out, h1, c1, h2, c2)
print(out.shape)
