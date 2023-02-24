import numpy as np


def csv_trans(csv_data):
    return np.array(csv_data[csv_data.columns[0]][:]), \
        np.array(csv_data[csv_data.columns[1]][:]), \
        np.array(csv_data[csv_data.columns[2]][:])


def csv_reshape(data, batch_size):
    out = np.zeros([data.shape[0] - batch_size + 1, batch_size])
    for i in range(out.shape[0]):
        for j in range(batch_size):
            out[i, j] = data[i + j]
    return out
