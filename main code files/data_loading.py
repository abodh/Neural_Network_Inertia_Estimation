import h5py
import numpy as np
# from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt
import pdb
import torch
from torch.utils.data import Dataset

class freq_data(Dataset):
    # Constructor
    def __init__(self, path):
        file_freq = path + 'freq_norm.mat'
        file_rocof = path + 'rocof_norm.mat'
        freq_data, rocof_data = loading(file_freq, file_rocof)
        self.x, self.y, _, _ = separate_dataset(freq_data, rocof_data)
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # Return the length
    def __len__(self):
        return self.len

def loading(file_freq, file_rocof):
    '''
    loading the data from the mat file

    :param file_freq: mat file that contains the frequency data
    :param file_rocof: mat file that contains the rocof data
    :return: array of frequency and rocof

    '''

    # loading total data
    file_f = h5py.File(file_freq, 'r')
    file_rocof = h5py.File(file_rocof, 'r')
    f_var = file_f.get('f')
    rocof_var = file_rocof.get('rf')
    f_var = np.array(f_var).T
    rocof_var = np.array(rocof_var).T
    return f_var, rocof_var

def separate_dataset(freq_data, rocof_data):
    '''

    :param freq_data: change of frequency data extracted from the matfile
    :param rocof_data: rocof data extracted from the matfile
    :return: separate training dataset for each of the inputs(frequency, rocof, and p) and an output dataset of inertia

    Note: the data have been normalized already in MATLAB

    '''
    # loads = np.genfromtxt('pulses.csv', delimiter=',')
    # loads = loads.transpose()

    total_dataset = np.hstack((freq_data[:,0:201],rocof_data[:,0:201], freq_data[:,-1:])) # here 201 is used just to
    # extract first 201 datapoints

    # total_dataset = np.random.permutation(total_dataset)
    # train_num = int(0.8 * len(total_dataset))  # number of data to be trained
    # pdb.set_trace()
    # train_f_rf = total_dataset[0:train_num,:-1]
    # train_M_D = total_dataset[0:train_num,-1]
    # test_f_rf = total_dataset[train_num:len(total_dataset), :-1]
    # test_M_D = total_dataset[train_num:len(total_dataset), -1]
    # pdb.set_trace()
    # return train_f_rf, train_M_D, test_f_rf, test_M_D

    x = total_dataset[:,:-1] # x contains freq and rocof datapoints
    y = total_dataset[:,-1] # y contains inertia constant

    return x, y, freq_data[:,0:201],rocof_data[:,0:201]

if __name__ == '__main__':
    # testing if the above functions work properly

    path = "..\\..\\matlab files\\0.2Hz\\manipulated\\"
    file_freq = path + 'freq_norm.mat'
    file_rocof = path + 'rocof_norm.mat'
    freq_data, rocof_data = loading(file_freq, file_rocof)
    _, _, f, rf = separate_dataset(freq_data, rocof_data)
    for i in range(f.shape[0]):
        plt.subplot(211)
        plt.plot(f[i,:])
        plt.subplot(212)
        plt.plot(rf[i,:])
    plt.show()