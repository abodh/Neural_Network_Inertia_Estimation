import h5py
import numpy as np
from sklearn.preprocessing import normalize
import pdb

def loading(filename):

    # loading total data
    file = h5py.File(filename, 'r')
    f_var = file.get('x')
    final_data = np.array(f_var)
    return final_data[:,1:np.shape(final_data)[1]]

def normalize_data (total_data):
    '''
        normalize and create dataset to train
        
        :return: 
        train x, train y, test x, test y
    '''

    # permuted_total_data = np.random.permutation(total_data)
    permuted_total_data = total_data[:,:]
    # train_num = int(0.8 * len(permuted_total_data))  # generalizes for all, use any of the inertia values data to get length
    train_num = len(permuted_total_data)

    # normalized_input_data = normalize(permuted_total_data[:, 0:3], axis=0, norm='l2')
    train_5 = total_data[np.where(total_data[:,-1]==5)]
    train_10 = total_data[np.where(total_data[:, -1] == 10)]
    train_15 = total_data[np.where(total_data[:, -1] == 15)]
    train_20 = total_data[np.where(total_data[:, -1] == 20)]
    train_25 = total_data[np.where(total_data[:, -1] == 25)]

    training_data = np.stack((train_5, train_10, train_15, train_20, train_25))
    training_data = training_data.reshape((-1,4))
    # training_data = permuted_total_data[:,:]
    normalized_input_data = training_data[:, 0:3]
    train_x = normalized_input_data[0:train_num, :]
    train_y = np.transpose(training_data[0:train_num, np.shape(training_data)[1]-1])
    # test_x = normalized_input_data[train_num:np.shape(training_data)[0], :]
    # test_y = np.transpose(training_data[train_num:np.shape(training_data)[0], np.shape(training_data)[1]-1])
    # pdb.set_trace()
    return train_x, train_y, \
           # test_x, test_y

if __name__ == '__main__':
    ans = loading('file.mat')
    print (np.shape(ans))
    print (len(ans))
    # print(ans)
    train_x, train_y, test_x, test_y  = normalize_data(ans)
    pdb.set_trace()