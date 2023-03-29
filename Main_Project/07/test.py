import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TrajectoryDataset(Dataset):
    def __init__(self):
        self.training_data_input = np.load("training_data_input.npy")
        self.training_data_output = np.load("training_data_output.npy")
        self.testing_data_input = np.load("testing_data_input.npy")
        self.testing_data_output = np.load("testing_data_output.npy")
        self.sample_training_data_input = self.training_data_input.shape[0]
        self.sample_training_data_output = self.training_data_output.shape[0]
        self.sample_testing_data_input = self.testing_data_input.shape[0]
        self.sample_testing_data_output = self.testing_data_output.shape[0]

    def __getitem__(self, index):
        return self.training_data_input[index], \
            self.training_data_output[index], \
            self.testing_data_input[index], \
            self.testing_data_output[index]

    def __len__(self):
        return self.sample_training_data_input, self.sample_training_data_output, \
            self.sample_testing_data_input, self.sample_testing_data_output


trajectory = TrajectoryDataset()
print(trajectory.__len__())




