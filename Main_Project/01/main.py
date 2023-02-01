import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置运行设备的环境为GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'本次程序运行的设备环境为{device}')


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()

    def forward(self, x):
        pass


