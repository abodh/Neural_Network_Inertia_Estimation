import h5py
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt
import pdb

def loading(file_freq, file_rocof):
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
    total_dataset = np.hstack((freq_data[:,0:-2],rocof_data[:,0:-2],freq_data[:,-1:]))
    # pdb.set_trace()
    total_dataset = np.random.permutation(total_dataset)
    train_num = int(0.8 * len(total_dataset))  # number of data to be trained

    train_f_rf = total_dataset[0:train_num,:-1]
    train_M_D = total_dataset[0:train_num,-1]
    test_f_rf = total_dataset[train_num:len(total_dataset), :-1]
    test_M_D = total_dataset[train_num:len(total_dataset), -1]
    # pdb.set_trace()
    return train_f_rf, train_M_D, test_f_rf, test_M_D

if __name__ == '__main__':

    # testing if the above functions work properly

    path = ".\\data files\\excitation_test\\manipulated\\"
    file_freq = path + 'freq_norm.mat'
    file_rocof = path + 'rocof_norm.mat'
    freq_data, rocof_data = loading(file_freq, file_rocof)
    train_f, train_M_D, test_f, test_M_D = separate_dataset(freq_data, rocof_data)
    plt.subplot(221)
    plt.plot(train_f[55])
    plt.subplot(222)
    plt.plot(test_f[55])
    plt.show()
    pdb.set_trace()