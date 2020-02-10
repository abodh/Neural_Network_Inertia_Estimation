import h5py
import numpy as np
from sklearn.preprocessing import normalize
import pdb

def loading(inertia, low, high, f_file, rocof_file):
    # loading frequency data
    file = h5py.File(f_file, 'r')
    f_var = file.get('ans')
    f_var = np.array(f_var)

    #loading rocof data
    rf = h5py.File(rocof_file, 'r')
    rf_var = rf.get('ans')
    rf_var = np.array(rf_var)

    #concatenating data to form a tabular data form
    rocof = rf_var[:,1]
    rocof.shape = (len(rocof),1)
    main_data = np.hstack((f_var, rocof))

    inertia_vector = inertia * np.ones(len(rocof))
    inertia_vector.shape = (len(rocof), 1)
    final_data = np.hstack((main_data, inertia_vector))
    # print(final_data[4000:5000, :])
    # print (np.shape(final_data))
    return final_data[((final_data[:,0] > low) & (final_data[:,0] < high)),1:4]

def normalize_create (iner_data_a, iner_data_b, iner_data_c):
    '''
        normalize and create dataset to train
        
        :return: 
        train x, train y, test x, test y
    '''

    iner_data_5 = np.random.permutation(iner_data_a)
    iner_data_8 = np.random.permutation(iner_data_b)
    iner_data_10 = np.random.permutation(iner_data_c)

    train_num = int(0.8 * len(iner_data_5))  # generalizes for all, use any of the inertia values data to get length
    # pdb.set_trace()
    normalized_iner_data_5 = normalize(iner_data_5[:, 0:2], axis=0, norm='l2')
    train_x_5 = normalized_iner_data_5[0:train_num, :]
    train_y_5 = np.transpose(iner_data_5[0:train_num, 2])
    test_x_5 = normalized_iner_data_5[train_num:len(iner_data_5), :]
    test_y_5 = np.transpose(iner_data_5[train_num:len(iner_data_5), 2])

    normalized_iner_data_8 = normalize(iner_data_8[:, 0:2], axis=0, norm='l2')
    train_x_8 = normalized_iner_data_8[0:train_num, :]
    train_y_8 = np.transpose(iner_data_8[0:train_num, 2])
    test_x_8 = normalized_iner_data_8[train_num:len(iner_data_8), :]
    test_y_8 = np.transpose(iner_data_8[train_num:len(iner_data_8), 2])

    normalized_iner_data_10 = normalize(iner_data_10[:, 0:2], axis=0, norm='l2')
    train_x_10 = normalized_iner_data_10[0:train_num, :]
    train_y_10 = np.transpose(iner_data_10[0:train_num, 2])
    test_x_10 = normalized_iner_data_10[train_num:len(iner_data_10), :]
    test_y_10 = np.transpose(iner_data_10[train_num:len(iner_data_10), 2])

    train_x = np.vstack((train_x_5, train_x_8, train_x_10))
    train_y = np.vstack((train_y_5.reshape(-1,1), train_y_8.reshape(-1,1), train_y_10.reshape(-1,1)))
    test_x = np.vstack((test_x_5, test_x_8, test_x_10))
    test_y = np.vstack((test_y_5.reshape(-1,1), test_y_8.reshape(-1,1), test_y_10.reshape(-1,1)))
    # pdb.set_trace()
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    # inertia = 10
    # low = 10
    # high = 12.5
    # f_file = 'frequency_10.mat'
    # rocof_file = 'rocofrequency_10.mat'
    ans = loading(inertia = 10, low = 10, high = 12.5, f_file = 'frequency_10.mat', rocof_file = 'rocofrequency_10.mat')
    print (np.shape(ans))
    print (len(ans))
    print(ans)