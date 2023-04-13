import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import training_data_input, training_data_output, testing_data_input, testing_data_output
from utils import data_size, input_size, row, column, side_length_y, side_length_x_center, side_length_x
from utils import training_data_output_origin, testing_data_output_origin
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
size_decoder_fc_output = int(row * column)

learning_rate_init = 1e-2
learning_rate = learning_rate_init
max_epoch = 10000
batch_ratio = 1


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, encoder_fc_input_size, encoder_fc_middle_size, encoder_fc_output_size,
                 encoder_lstm_input_size, encoder_lstm_hidden_size,
                 encoder_activate_num_parameters_size):
        super(Encoder, self).__init__()
        self.encoder_activate_init = 0.5
        self.encoder_lstm_hidden_size = encoder_lstm_hidden_size
        self.encoder_bias = True
        self.encoder_lstm_num_layers = 1

        self.encoder_fc1 = nn.Linear(encoder_fc_input_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc2 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc3 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc4 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc5 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc6 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc7 = nn.Linear(encoder_fc_middle_size, encoder_fc_middle_size, bias=self.encoder_bias)
        self.encoder_fc8 = nn.Linear(encoder_fc_middle_size, encoder_fc_output_size, bias=self.encoder_bias)
        self.encoder_lstm_front = nn.LSTM(encoder_lstm_input_size, encoder_lstm_hidden_size,
                                          num_layers=self.encoder_lstm_num_layers, batch_first=True)
        self.encoder_lstm_back = nn.LSTM(encoder_lstm_input_size, encoder_lstm_hidden_size,
                                         num_layers=self.encoder_lstm_num_layers, batch_first=True)
        self.encoder_activate = nn.LeakyReLU(negative_slope=0.1)
        self.encoder_activate1 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        self.encoder_activate2 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        self.encoder_activate3 = nn.PReLU(num_parameters=encoder_activate_num_parameters_size,
                                          init=self.encoder_activate_init)
        self.encoder_batch_normalization = nn.BatchNorm1d(encoder_activate_num_parameters_size, affine=False)

    def forward(self, x):
        h0 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        c0 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        h1 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)
        c1 = torch.ones(self.encoder_lstm_num_layers, x.size(0), self.encoder_lstm_hidden_size).to(device)

        out = self.encoder_fc1(x)
        out = self.encoder_batch_normalization(out)
        out = self.encoder_fc2(self.encoder_activate(out))
        out = self.encoder_batch_normalization(out)
        out = self.encoder_fc3(self.encoder_activate(out))
        out = self.encoder_batch_normalization(out)
        # out = self.encoder_fc4(self.encoder_activate(out))
        # out = self.encoder_batch_normalization(out)
        # out = self.encoder_fc5(self.encoder_activate(out))
        # out = self.encoder_batch_normalization(out)
        # out = self.encoder_fc6(self.encoder_activate(out))
        # out = self.encoder_fc7(self.encoder_activate(out))
        out = self.encoder_fc8(self.encoder_activate(out))
        out = self.encoder_batch_normalization(out)
        out_front, _ = self.encoder_lstm_front(out, (h0, c0))
        out_back, _ = self.encoder_lstm_back(out.flip(dims=[1]), (h1, c1))
        out = torch.add(out_front, out_back)
        out = out[:, -1, :].unsqueeze(1)
        return out


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, decoder_lstm_input_size, decoder_lstm_hidden_size,
                 decoder_fc_input_size, decoder_fc_middle_size, decoder_fc_output_size):
        super(Decoder, self).__init__()
        self.decoder_lstm_hidden_size = decoder_lstm_hidden_size
        self.decoder_bias = True
        self.decoder_activate_init = 0.5
        self.decoder_lstm_num_layers = 1

        self.decoder_lstm = nn.LSTM(decoder_lstm_input_size, decoder_lstm_hidden_size,
                                    num_layers=self.decoder_lstm_num_layers, batch_first=True)
        self.decoder_fc1 = nn.Linear(decoder_fc_input_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc2 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc3 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc4 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc5 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc6 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc7 = nn.Linear(decoder_fc_middle_size, decoder_fc_middle_size, bias=self.decoder_bias)
        self.decoder_fc8 = nn.Linear(decoder_fc_middle_size, decoder_fc_output_size, bias=self.decoder_bias)
        self.decoder_softmax = nn.Softmax(dim=2)
        self.decoder_activate = nn.LeakyReLU(negative_slope=0.1)
        self.decoder_activate1 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_activate2 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_activate3 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_activate4 = nn.PReLU(num_parameters=1,
                                          init=self.decoder_activate_init)
        self.decoder_batch_normalization = nn.BatchNorm1d(1, affine=False)

    def forward(self, x):
        out = self.decoder_fc1(self.decoder_activate(x))
        out = self.decoder_batch_normalization(out)
        out = self.decoder_fc2(self.decoder_activate(out))
        out = self.decoder_batch_normalization(out)
        out = self.decoder_fc3(self.decoder_activate(out))
        out = self.decoder_batch_normalization(out)
        # out = self.decoder_fc4(self.decoder_activate(out))
        # out = self.decoder_batch_normalization(out)
        # out = self.decoder_fc5(self.decoder_activate(out))
        # out = self.decoder_batch_normalization(out)
        # out = self.decoder_fc6(self.decoder_activate(out))
        # out = self.decoder_fc7(self.decoder_activate(out))
        out = self.decoder_fc8(self.decoder_activate(out))
        return out


# 模型实例化
encoder = Encoder(size_encoder_fc_input, size_encoder_fc_middle, size_encoder_fc_output,
                  size_encoder_lstm_input, size_encoder_lstm_hidden, size_encoder_activate_num_parameters).to(device)
decoder = Decoder(size_decoder_lstm_input, size_decoder_lstm_hidden,
                  size_decoder_fc_input, size_decoder_fc_middle, size_decoder_fc_output).to(device)
softmax = nn.Softmax(dim=2)
print('当前显卡的显存使用率:',
      torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100, '%')

# 优化器和损失函数
optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 模式选取
mode_switch = int(input("请进行模式选择："))

# 主要部分
fig = plt.figure()
if mode_switch == 0:
    print("进行模型训练")

    scheduler_encoder = scheduler.StepLR(optimizer_encoder, step_size=50, gamma=0.9, last_epoch=-1)
    scheduler_decoder = scheduler.StepLR(optimizer_decoder, step_size=50, gamma=0.9, last_epoch=-1)

    all_loss = np.zeros([1, int(1 / batch_ratio)])
    all_grad_abs = np.array([[0.0, 10]])
    for epoch in range(max_epoch):
        batch_size = training_data_input.shape[0] * batch_ratio
        for each_batch in range(int(1 / batch_ratio)):
            train_input = training_data_input[int(each_batch * batch_size):int((each_batch + 1) * batch_size), :, :]
            train_output = training_data_output[int(each_batch * batch_size):int((each_batch + 1) * batch_size), :, :]
            encoded = encoder(train_input)
            decoded = decoder(encoded)
            loss = criterion(decoded[:, 0, :], train_output[:, 0, :])
            # print('当前显卡的显存使用率:',
            #       torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100, '%')

            all_loss[epoch, each_batch] = loss.item()

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()

            if (epoch + 1) % 50 == 0:
                print(f"epoch:{epoch + 1},loss:{loss.item()}")

            optimizer_encoder.step()
            optimizer_decoder.step()

        for name, param in encoder.named_parameters():
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

        plt.subplot(3, 1, 1)
        plt.plot(all_loss[:, -1])
        plt.text(epoch / 2, (all_loss[:, -1].max() + all_loss[:, -1].min()) / 2,
                 f"lr:{learning_rate:.10f}", fontsize=10)

        plt.subplot(3, 1, 2)
        plt.plot(all_grad_abs[:, 0])
        plt.plot([0.0, epoch], [0.0, 0.0], "r--")

        plt.subplot(3, 1, 3)
        plt.plot(all_grad_abs[:, 1])
        plt.plot([0.0, epoch], [0.0, 0.0], "r--")

        plt.pause(0.01)

        all_loss = np.append(all_loss, np.zeros([1, int(1 / batch_ratio)]), axis=0)
        all_grad_abs = np.append(all_grad_abs, np.array([[0.0, 10]]), axis=0)

        scheduler_encoder.step()
        scheduler_decoder.step()
        learning_rate = scheduler_encoder.get_last_lr()[0]

        rand_para = torch.randperm(training_data_input.shape[0])
        training_data_input = training_data_input[rand_para]
        training_data_output = training_data_output[rand_para]

    torch.save(encoder, "end_encoder.pth")
    torch.save(decoder, "end_decoder.pth")
    fig.savefig("figs/finish.png")
if mode_switch == 1:
    print("模型测试")

    encoder = torch.load("end_encoder.pth")
    decoder = torch.load("end_decoder.pth")
    encoder.eval()
    decoder.eval()

    check_input = training_data_input
    check_output = training_data_output
    check_output_origin = training_data_output_origin

    encoded = encoder(check_input)
    decoded = decoder(encoded)
    loss = criterion(decoded[:, 0, :], check_output[:, 0, :])
    print(loss.item())

    check_input = check_input.cpu().detach().numpy()
    check_output = check_output.cpu().detach().numpy()
    decoded = decoded.cpu().detach().numpy()

    all_loss = np.zeros([1])
    for i in range(check_output.shape[0]):
        plt.clf()

        plt.subplot(1, 2, 2)
        all_loss[-1] = criterion(torch.tensor(decoded[i, 0, :]), torch.tensor(check_output[i, 0, :])).item()
        plt.plot(all_loss)
        all_loss = np.append(all_loss, np.zeros([1]), axis=0)

        plt.subplot(1, 2, 1)
        softmax = nn.Softmax(dim=0)
        decoded[i, 0, :] = softmax(torch.tensor(decoded[i, 0, :])).numpy()
        plt.plot([0.0, 0.0], [-1.0, 1.0], "r--")
        # plt.plot(check_input[i, :, 0], check_input[i, :, 1], ".")
        plt.plot(check_output_origin[i, :, 1], check_output_origin[i, :, 2], ".")

        lim = 8
        plt.xlim(-(int(column / 2) * side_length_x + side_length_x_center / 2),
                 (int(column / 2) * side_length_x + side_length_x_center / 2))
        plt.ylim(0.0, row * side_length_y)

        plt.plot([side_length_x_center / 2, side_length_x_center / 2], [0.0, side_length_y * row], "r--")
        plt.plot([-side_length_x_center / 2, -side_length_x_center / 2], [0.0, side_length_y * row], "r--")
        plt.plot([-(side_length_x_center / 2 + side_length_x), -(side_length_x_center / 2 + side_length_x)],
                 [0.0, side_length_y * row], "r--")
        plt.plot([(side_length_x_center / 2 + side_length_x), (side_length_x_center / 2 + side_length_x)],
                 [0.0, side_length_y * row], "r--")

        plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
                 [side_length_y, side_length_y], "r--")
        plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
                 [side_length_y * 2, side_length_y * 2], "r--")
        plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
                 [side_length_y * 3, side_length_y * 3], "r--")
        plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
                 [side_length_y * 4, side_length_y * 4], "r--")

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2,
                 f"{decoded[i, 0, 0]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2,
                 f"{decoded[i, 0, 1]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2,
                 f"{decoded[i, 0, 2]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2,
                 f"{decoded[i, 0, 3]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2,
                 f"{decoded[i, 0, 4]:.1f}", fontsize=10)

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 5]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 6]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 7]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 8]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 9]:.1f}", fontsize=10)

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 10]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 11]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 12]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 13]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 14]:.1f}", fontsize=10)

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 15]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 16]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 17]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 18]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 19]:.1f}", fontsize=10)

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 20]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 21]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 22]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 23]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 24]:.1f}", fontsize=10)

        if (i + 1) % 300 == 0:
            fig.savefig("../result/09/0_" + str(i + 1) + ".png")
        plt.pause(0.001)
    fig.savefig("../result/09/train.png")

    check_input = testing_data_input
    check_output = testing_data_output
    check_output_origin = testing_data_output_origin

    encoded = encoder(check_input)
    decoded = decoder(encoded)
    loss = criterion(decoded[:, 0, :], check_output[:, 0, :])
    print(loss.item())

    check_input = check_input.cpu().detach().numpy()
    check_output = check_output.cpu().detach().numpy()
    decoded = decoded.cpu().detach().numpy()

    all_loss = np.zeros([1])
    for i in range(check_output.shape[0]):
        plt.clf()

        plt.subplot(1, 2, 2)
        all_loss[-1] = criterion(torch.tensor(decoded[i, 0, :]), torch.tensor(check_output[i, 0, :])).item()
        plt.plot(all_loss)
        all_loss = np.append(all_loss, np.zeros([1]), axis=0)

        plt.subplot(1, 2, 1)
        softmax = nn.Softmax(dim=0)
        decoded[i, 0, :] = softmax(torch.tensor(decoded[i, 0, :])).numpy()
        plt.plot([0.0, 0.0], [-1.0, 1.0], "r--")
        # plt.plot(check_input[i, :, 0], check_input[i, :, 1], ".")
        plt.plot(check_output_origin[i, :, 1], check_output_origin[i, :, 2], ".")

        lim = 8
        plt.xlim(-(int(column / 2) * side_length_x + side_length_x_center / 2),
                 (int(column / 2) * side_length_x + side_length_x_center / 2))
        plt.ylim(0.0, row * side_length_y)

        plt.plot([side_length_x_center / 2, side_length_x_center / 2], [0.0, side_length_y * row], "r--")
        plt.plot([-side_length_x_center / 2, -side_length_x_center / 2], [0.0, side_length_y * row], "r--")
        plt.plot([-(side_length_x_center / 2 + side_length_x), -(side_length_x_center / 2 + side_length_x)],
                 [0.0, side_length_y * row], "r--")
        plt.plot([(side_length_x_center / 2 + side_length_x), (side_length_x_center / 2 + side_length_x)],
                 [0.0, side_length_y * row], "r--")

        plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
                 [side_length_y, side_length_y], "r--")
        plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
                 [side_length_y * 2, side_length_y * 2], "r--")
        plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
                 [side_length_y * 3, side_length_y * 3], "r--")
        plt.plot([(side_length_x_center / 2 + side_length_x * 2), -(side_length_x_center / 2 + side_length_x * 2)],
                 [side_length_y * 4, side_length_y * 4], "r--")

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2,
                 f"{decoded[i, 0, 0]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2,
                 f"{decoded[i, 0, 1]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2,
                 f"{decoded[i, 0, 2]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2,
                 f"{decoded[i, 0, 3]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2,
                 f"{decoded[i, 0, 4]:.1f}", fontsize=10)

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 5]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 6]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 7]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 8]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y,
                 f"{decoded[i, 0, 9]:.1f}", fontsize=10)

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 10]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 11]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 12]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 13]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 2,
                 f"{decoded[i, 0, 14]:.1f}", fontsize=10)

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 15]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 16]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 17]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 18]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 3,
                 f"{decoded[i, 0, 19]:.1f}", fontsize=10)

        plt.text(-(side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 20]:.1f}", fontsize=10)
        plt.text(-(side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 21]:.1f}", fontsize=10)
        plt.text(0.0, side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 22]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 23]:.1f}", fontsize=10)
        plt.text((side_length_x / 2 + side_length_x + side_length_x_center / 2), side_length_y / 2 + side_length_y * 4,
                 f"{decoded[i, 0, 24]:.1f}", fontsize=10)

        if (i + 1) % 300 == 0:
            fig.savefig("../result/09/1_" + str(i+1) + ".png")
        plt.pause(0.001)
    fig.savefig("../result/09/test.png")
