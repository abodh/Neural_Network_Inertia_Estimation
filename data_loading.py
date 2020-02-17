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
    f_var = np.array(f_var[0:77, :]).T
    rocof_var = np.array(rocof_var[0:77, :]).T
    return f_var, rocof_var

def separate_dataset(freq_data, rocof_data):
    '''

    :param freq_data: change of frequency data extracted from the matfile
    :param rocof_data: rocof data extracted from the matfile
    :return: separate training dataset for each of the inputs(frequency, rocof, and p) and an output dataset of inertia

    Note: the data have been normalized already in MATLAB

    '''

    train_num = int(0.8*len(freq_data))       # number of data to be trained

    train_f = freq_data[0:train_num,:-2]
    train_rf = rocof_data[0:train_num,:-2]
    train_p = freq_data[0:train_num,-2]
    train_M = freq_data[0:train_num,-1]

    test_f = freq_data[train_num:len(freq_data), :-2]
    test_rf = rocof_data[train_num:len(freq_data), :-2]
    test_p = freq_data[train_num:len(freq_data), -2]
    test_M = freq_data[train_num:len(freq_data), -1]

    return train_f, train_rf, train_p, train_M, test_f, test_rf, test_p, test_M

if __name__ == '__main__':

    # testing if the above functions work properly

    path = ".\\data files\\new_data\\manipulated\\"
    file_freq = path + 'freq_norm.mat'
    file_rocof = path + 'rocof_norm.mat'
    freq_data, rocof_data = loading(file_freq, file_rocof)
    train_f, train_rf, train_p, train_M, test_f, test_rf, test_p, test_M = separate_dataset(freq_data, rocof_data)
    plt.subplot(221)
    plt.plot(train_f[0])
    plt.subplot(222)
    plt.plot(train_rf[0])
    plt.subplot(223)
    plt.plot(test_f[0])
    plt.subplot(224)
    plt.plot(test_rf[0])
    plt.show()
    pdb.set_trace()