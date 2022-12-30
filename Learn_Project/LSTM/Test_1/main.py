# 基于网上LSTM的代码进行改进


# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


# 读取txt文件，由于txt在存储的时候顺序为stamp、acc、position、velocity，所以需要在此处调整
def read_txt(filename):
    file_data = open(filename, 'r')
    file_data_lines = file_data.readlines()
    data = np.zeros((len(file_data_lines), 4))
    temp = 0
    for row in file_data_lines:
        find_data = row.split('\t')
        data[temp, 0] = float(find_data[0])
        data[temp, 1] = float(find_data[1])
        data[temp, 2] = float(find_data[2])
        data[temp, 3] = float(find_data[3])
        temp += 1
    data[:, 0] = data[:, 0] - data[0, 0]
    print("完成读取")
    return data


if __name__ == '__main__':

    filepath = 'out.txt'
    txt_data = read_txt(filepath)
    data_len = len(txt_data)
    t = txt_data[:, 0]
    # 对y值进行预测
    original_t = txt_data[:, 3]
    predict_t = txt_data[:, 2]
    # plt.figure()
    # plt.plot(t[:], original_t[:])
    # plt.plot(t[:], predict_t[:])
    # plt.show()

    dataset = np.zeros((data_len, 2))
    dataset[:, 0] = original_t
    dataset[:, 1] = predict_t
    dataset = dataset.astype('float32')

    # plot part of the original dataset
    plt.figure()
    plt.plot(t[0:data_len], dataset[0:data_len, 0], label='original')
    plt.plot(t[0:data_len], dataset[0:data_len, 1], label='predict')
    plt.xlabel('t')
    plt.ylim(-8, 6)
    plt.ylabel('position&acc')
    plt.legend(loc='upper right')

    # choose dataset for training and testing
    train_data_ratio = 0.5  # Choose 50% of the data for testing
    train_data_len = int(data_len * train_data_ratio)
    train_original = dataset[:train_data_len, 0]
    train_predict = dataset[:train_data_len, 1]
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    t_for_training = t[:train_data_len]

    # test_original + train_original = All
    # test_predict + train_predict = All
    test_original = dataset[train_data_len:, 0]
    test_predict = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    train_original_tensor = train_original.reshape(-1, 10, INPUT_FEATURES_NUM)  # set batch size to 5
    train_predict_tensor = train_predict.reshape(-1, 10, OUTPUT_FEATURES_NUM)  # set batch size to 5

    # transfer data to pytorch tensor
    train_original_tensor = torch.from_numpy(train_original_tensor)
    train_predict_tensor = torch.from_numpy(train_predict_tensor)
    # test_original_tensor = torch.from_numpy(test_original)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=1)  # 16 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    max_epochs = 100000
    for epoch in range(max_epochs):
        output = lstm_model(train_original_tensor)
        loss = loss_function(output, train_predict_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item() < 1e-3:
            print('----------------------------')
            print('Epoch: [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_original_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files

    # ----------------- test.py -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test.py dataset
    test_original_tensor = test_original.reshape(-1, 10,
                                                 INPUT_FEATURES_NUM)  # set batch size to 5
    test_original_tensor = torch.from_numpy(test_original_tensor)

    predictive_y_for_testing = lstm_model(test_original_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_original, 'g', label='original_trn')
    plt.plot(t_for_training, train_predict, 'b', label='ref_predict_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_predict_trn')

    plt.plot(t_for_testing, test_original, 'c', label='original_tst')
    plt.plot(t_for_testing, test_predict, 'k', label='ref_predict_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_predict_tst')

    # plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line')  # separation line

    plt.xlabel('t')
    plt.ylabel('position&acc')
    plt.xlim(t[0], t[-1])
    plt.ylim(-8, 6)
    plt.legend(loc='upper right')

    plt.show()
