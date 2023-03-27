import math
from utils import training_data_input, training_data_output, testing_data_input, testing_data_output, Qw, Ql
from utils import find_l, find_w
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gc
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as scheduler

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
training_or_testing = 0  # 0:训练 1:测试
if training_or_testing == 0:
    print("本次程序将会进行训练，并测试模型")
else:
    print("本次程序只作测试")

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
learning_rate_init = 5e-4
learning_rate = learning_rate_init
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
        h0 = torch.ones(1, x.size(0), self.encoder_hidden_size_lstm).to(device)
        c0 = torch.ones(1, x.size(0), self.encoder_hidden_size_lstm).to(device)

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
        y = self.decoder_softmax(y)
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


# 设置模型基本参数
encoder = Encoder(size_encoder_input, size_encoder_hidden_fc, size_encoder_hidden_lstm, size_encoder_output_lstm).to(device)
decoder = Decoder(size_decoder_input, size_decoder_hidden_fc, size_decoder_hidden_lstm, size_decoder_output_fc).to(device)
transition = Transition(size_transition_input, size_transition_hidden_fc, size_transition_output_fc)

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
scheduler_encoder = scheduler.StepLR(optimizer_encoder, step_size=50, gamma=0.99, last_epoch=-1)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
scheduler_decoder = scheduler.StepLR(optimizer_decoder, step_size=50, gamma=0.99, last_epoch=-1)
optimizer_transition = torch.optim.Adam(transition.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
# 单点训练
all_loss = np.zeros([1])
lr = np.zeros([1])
fig = plt.figure()
if training_or_testing == 0:
    for epoch in range(max_epochs):
        encoded, (h1, c1), (h2, c2) = encoder.encoder(training_data_input)
        decoded, (h1, c1), (h2, c2) = decoder.decoder(encoded, h1, c1, h2, c2)
        loss = criterion(decoded, training_data_output)

        all_loss[epoch] = loss.item()
        plt.clf()
        show = 400
        cal = 100
        k = 0.0
        plt.subplot(2, 1, 1)
        if np.linspace(0, show, show + 1).shape[0] == all_loss[max([epoch - show, 0]):(epoch + 1)].shape[0]:
            x = np.linspace(0, cal, cal+1)
            y = all_loss[max([epoch - cal, 0]):(epoch+1)]
            k, _ = np.polyfit(x, y, 1)

            if abs(k) <= 1e-4 and k <= 0:
                scheduler_encoder.step()
                scheduler_decoder.step()
                learning_rate = scheduler_encoder.get_last_lr()[0]
                plt.text(show / 5, (all_loss[max([epoch - show, 0]):(epoch+1)].max() + all_loss[max([epoch - show, 0]):(epoch+1)].min()) / 2,
                         "k is too low", fontsize=10)
            # if learning_rate <= 5e-5:
            #     learning_rate = learning_rate_init
            #     optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
            #     scheduler_encoder = scheduler.StepLR(optimizer_encoder, step_size=50, gamma=0.9, last_epoch=-1)
            #     optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
            #     scheduler_decoder = scheduler.StepLR(optimizer_decoder, step_size=50, gamma=0.9, last_epoch=-1)
        plt.plot(all_loss[max([epoch - show, 0]):(epoch+1)])
        plt.plot([show, show], [all_loss[max([epoch - show, 0]):(epoch+1)].min(), all_loss[max([epoch - show, 0]):(epoch+1)].max()], "r--")
        plt.plot([show - cal, show - cal], [all_loss[max([epoch - show, 0]):(epoch+1)].min(), all_loss[max([epoch - show, 0]):(epoch+1)].max()], "r--")
        plt.text(show / 2,
                 (all_loss[max([epoch - show, 0]):(epoch+1)].max() + all_loss[max([epoch - show, 0]):(epoch+1)].min()) / 2,
                 f"k:{k:.10f}", fontsize=10)
        plt.subplot(2, 1, 2)
        lr[epoch] = learning_rate
        plt.text(show / 2,
                 (lr[max([epoch - show, 0]):(epoch + 1)].max() + lr[max([epoch - show, 0]):(epoch + 1)].min()) / 2,
                 f"lr:{learning_rate:.10f}", fontsize=10)
        plt.plot(lr[max([epoch - show, 0]):(epoch+1)])
        plt.pause(0.001)

        all_loss = np.append(all_loss, [0.0], axis=0)
        lr = np.append(lr, [0.0], axis=0)
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        if (epoch+1) % 50 == 0:
            fig.savefig("figs/" + str(epoch+1) + ".png")
            print(f"epoch:{epoch+1},loss:{loss.item()}")
            tensorboard_writer.add_scalar("loss", loss.item(), epoch)
        if loss.item() <= 1e-2:
            torch.save(encoder, str(epoch + 1) + "_encoder_" + str(loss.item()))
            torch.save(decoder, str(epoch + 1) + "_decoder_" + str(loss.item()))
        optimizer_encoder.step()
        optimizer_decoder.step()
    torch.save(encoder, "end_encoder")
    torch.save(decoder, "end_decoder")
    tensorboard_writer.close()
elif training_or_testing == 1:
    encoder = torch.load("end_encoder")
    decoder = torch.load("end_decoder")
    with torch.no_grad():
        # encoded, (h1, c1), (h2, c2) = encoder.encoder(testing_data_input)
        # decoded, (h1, c1), (h2, c2) = decoder.decoder(encoded, h1, c1, h2, c2)
        # loss = criterion(decoded, testing_data_output)
        # print(loss.item())
        encoded, (h1, c1), (h2, c2) = encoder.encoder(training_data_input)
        decoded, (h1, c1), (h2, c2) = decoder.decoder(encoded, h1, c1, h2, c2)
        points_training = torch.zeros([training_data_input.shape[0], training_data_input.shape[1] + 1, 2])
        points_training[:, 0:10, 0] = training_data_input[:, :, 0]
        points_training[:, 0:10, 1] = training_data_input[:, :, 1]
        for i in range(points_training.shape[0]):
            points_training[i, 10, 0] = find_l(decoded[i, :].argmax())
            points_training[i, 10, 1] = find_w(decoded[i, :].argmax())
        plt.figure()
        for i in range(points_training.shape[0]):
            loss = criterion(decoded[i].unsqueeze(dim=0), training_data_output[i].unsqueeze(dim=0))
            print(loss.item())
            plt.clf()
            plt.plot(points_training[i, :, 0], points_training[i, :, 1], "*")
            plt.pause(0.01)





