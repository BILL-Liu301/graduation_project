import math
from utils import training_data_input, training_data_output, testing_data_input, testing_data_output, Qw, Ql
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 定义tensorboard的writer
tensorboard_writer = SummaryWriter("../runs/07")

# 清空缓存
torch.cuda.empty_cache()

# 设置运行设备的环境为GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}，{torch.cuda.get_device_name(device)}')

# 设定工作模式
training_or_testing = int(input("请输入模式选择，0代表训练，1代表只作测试："))  # 0:训练 1:测试
if training_or_testing == 0:
    print("本次程序将会进行训练，并测试模型")
else:
    print("本次程序只作测试")

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
learning_rate = 1e-2
max_epochs = 500

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

        decoded = torch.zeros([x.shape[0], self.decoder_delta, self.decoder_K, 2])
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

        average = decoded.sum(dim=2).to(device)
        # average = torch.zeros([decoded.shape[0], decoded.shape[1], 1, decoded.shape[3]]).to(device)
        # for i in range(decoded.shape[2]):
        #     average = average + decoded[:, :, 0:1, :]
        # average = average / torch.tensor(4.0)
        return decoded, average

    def forward(self, x):
        encoded, h1, c1, h2, c2 = self.encoder(x)
        decoded = self.decoder(encoded, h1, c1, h2, c2)
        return decoded


# 设置模型基本参数
model_predict = EncoderDecoder(size_encoder_input, size_encoder_hidden_fc, size_encoder_hidden_lstm,
                               size_encoder_output_lstm,
                               size_decoder_input, size_decoder_hidden_fc, size_decoder_hidden_lstm,
                               size_decoder_output_fc,
                               size_K, Qw, Ql, size_delta).to(device)
optimizer = torch.optim.Adam(model_predict.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

if training_or_testing == 0:
    for epoch in range(max_epochs):
        trajectory_prediction, middle = model_predict(training_data_input)
        loss = criterion(middle, training_data_output)
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            tensorboard_writer.add_scalar("loss", loss.item(), epoch)
        if loss.item() <= 1:
            break
else:
    print("程序结束")
