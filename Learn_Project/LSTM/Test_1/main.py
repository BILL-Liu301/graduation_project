# 基于网上LSTM的代码进行改进
import sys

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


# 重构数组，简单默认每一行都一样
def reshape_original_data(origin, rows, input_feature_num, output_feature_num):
    reshaped = origin
    if len(origin) % (rows * (input_feature_num + output_feature_num)) != 0.0:
        print("Reshape Fill")
        sys.exit(0)
    len_reshaped = len(origin) - (input_feature_num + output_feature_num) + 1
    reshaped = np.zeros((len_reshaped, rows, input_feature_num))
    for each_block in range(len_reshaped):
        for each_row in range(rows):
            for each_feature_num in range(input_feature_num):
                reshaped[each_block, each_row, each_feature_num] = origin[each_block + each_feature_num]
    return reshaped


if __name__ == '__main__':

    INPUT_FEATURES_NUM = 9
    OUTPUT_FEATURES_NUM = 1
    NUM_LAYER = 2
    ROWS = 1
    LR = 1e-2
    LOSS = 1e-5
    MAX_EPOCHS = 1

    filepath = 'out.txt'
    txt_data = read_txt(filepath)
    data_len = len(txt_data)
    t = txt_data[:, 0]
    # 对y值进行预测
    # original_t = txt_data[:, 3]
    original_t = txt_data[:, 2] * 0.5
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
    # plt.figure()
    # plt.plot(t[0:data_len], dataset[0:data_len, 0], label='original')
    # plt.plot(t[0:data_len], dataset[0:data_len, 1], label='predict')
    # plt.xlabel('t')
    # plt.ylabel('position&acc')
    # plt.legend(loc='upper right')

    # choose dataset for training and testing
    train_data_ratio = 0.5  # Choose 50% of the data for testing
    train_data_len = int(data_len * train_data_ratio)
    train_original = dataset[0:train_data_len, 0]
    train_predict = dataset[INPUT_FEATURES_NUM:train_data_len, 1]
    train_original_tensor = reshape_original_data(train_original, ROWS, INPUT_FEATURES_NUM, OUTPUT_FEATURES_NUM)
    train_predict_tensor = train_predict.reshape(-1, ROWS, OUTPUT_FEATURES_NUM)

    # test_original + train_original = All
    # test_predict + train_predict = All
    test_original = dataset[train_data_len:, 0]
    test_predict = dataset[train_data_len + INPUT_FEATURES_NUM:, 1]
    test_original_tensor = reshape_original_data(test_original, ROWS, INPUT_FEATURES_NUM, OUTPUT_FEATURES_NUM)
    test_predict_tensor = test_predict.reshape(-1, ROWS, OUTPUT_FEATURES_NUM)

    # transfer data to pytorch tensor
    train_original_tensor = torch.from_numpy(train_original_tensor).to(torch.float32)
    train_predict_tensor = torch.from_numpy(train_predict_tensor).to(torch.float32)
    test_original_tensor = torch.from_numpy(test_original_tensor).to(torch.float32)
    test_predict_tensor = torch.from_numpy(test_predict_tensor).to(torch.float32)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=NUM_LAYER)
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR)

    for epoch in range(MAX_EPOCHS):
        output = lstm_model(train_original_tensor)
        loss = loss_function(output, train_predict_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item() < LOSS:
            print('----------------------------')
            print('Epoch: [{}/{}], Loss: {:.5f}'.format(epoch + 1, MAX_EPOCHS, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss: {:.5f}'.format(epoch + 1, MAX_EPOCHS, loss.item()))

    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_original_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files

    # ----------------- test.py -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test.py dataset
    predictive_y_for_testing = lstm_model(test_original_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- plot -------------------
    plt.figure()
    # plt.plot(t_for_training, train_original, 'g', label='original_trn')
    # plt.plot(t_for_training, train_predict, 'b', label='ref_predict_trn')
    # plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_predict_trn')
    #
    # plt.plot(t_for_testing, test_original, 'c', label='original_tst')
    # plt.plot(t_for_testing, test_predict, 'k', label='ref_predict_tst')
    # plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_predict_tst')
    #
    # # plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line')  # separation line
    #
    # plt.xlabel('t')
    # plt.ylabel('position&acc')
    # plt.xlim(t[0], t[-1])
    # plt.legend(loc='upper right')
    all_original = np.append(train_original, test_original)
    all_predict = np.append(train_predict, test_predict)
    all_predict_from_lstm = np.append(predictive_y_for_training, predictive_y_for_testing)
    plt.plot(all_original, 'r', label='all_original')
    plt.plot(all_predict, 'g', label='all_predict')
    plt.plot(all_predict_from_lstm, 'b', label='all_predict_from_lstm')
    plt.legend(loc='upper right')
    start_x = 10
    start_y = 8
    delta_y = -0.5
    txt_size = 10
    plt.text(start_x, start_y + delta_y * 0, "SETTINGS : ", size=txt_size)
    plt.text(start_x, start_y + delta_y * 1, "lr : " + str(LR), size=txt_size)
    plt.text(start_x, start_y + delta_y * 2, "loss_set : " + str(LOSS), size=txt_size)
    plt.text(start_x, start_y + delta_y * 3, "input feature num : " + str(INPUT_FEATURES_NUM), size=txt_size)
    plt.text(start_x, start_y + delta_y * 4, "output feature num : " + str(OUTPUT_FEATURES_NUM), size=txt_size)
    plt.text(start_x, start_y + delta_y * 5, "num layer : " + str(NUM_LAYER), size=txt_size)
    plt.text(start_x, start_y + delta_y * 6, "source : " + "Pos * 0.5", size=txt_size)
    plt.ylim(-start_y - 0.5, start_y + 0.5)
    plt.show()

