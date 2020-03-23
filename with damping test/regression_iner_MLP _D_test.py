import numpy as np
import h5py
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import matplotlib as mpl
import math as m
import torch as T
from data_loading_D_test import loading, separate_dataset
import pdb

# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
axis_font = {'fontname': 'Arial', 'size': '16'}

# def accuracy(model, train_f, train_rf, train_p, train_M, pct_close):
def accuracy(model, test_f_rf, test_M_D, pct_close):
    with T.no_grad():
        n_items = len(test_M_D)
        # pdb.set_trace()
        X = T.Tensor(test_f_rf)
        # pdb.set_trace()
        # X3 = T.Tensor (train_p)
        # X3 = X3.reshape(-1,1)       # reshaping to 2-d Tensor
        Y = T.Tensor(test_M_D)        # actual as 1-d Tensor
        # oupt = model(X1, X2, X3)    # all predicted as 2-d Tensor
        oupt = model(X)
        # pdb.set_trace()
        pred = oupt.view(n_items)     # all predicted as 1-d

        # loss_val_M = T.sqrt(loss_func(oupt[:,0], Y[:,0]))
        # loss_val_D = T.sqrt(loss_func(oupt[:, 0], Y[:, 0]))
        # loss_val = loss_val_M + loss_val_D

        # RMSE_M = m.sqrt(T.mean((Y[:,0] - oupt[:,0]) ** 2))
        # RMSE_D = m.sqrt(T.mean((Y[:,1] - oupt[:,1]) ** 2))
        # RMSE = RMSE_M + RMSE_D
        RMSE = m.sqrt(T.mean((Y - pred) ** 2))

        # MAPE_M = 100 * (T.mean((T.abs(Y[:,0]-oupt[:,0])/Y[:,0])))
        # MAPE_D = 100 * (T.mean((T.abs(Y[:,1]-oupt[:,1])/Y[:,1])))
        # MAPE = MAPE_M + MAPE_D

        # pdb.set_trace()
        # n_correct_M = T.sum((T.abs(oupt[:,0] - Y[:,0]) < T.abs(pct_close * Y[:,0])))
        # n_correct_D = T.sum((T.abs(oupt[:,1] - Y[:,1]) < T.abs(pct_close * Y[:,1])))
        n_correct = T.sum((T.abs(pred - Y) < T.abs(pct_close * Y)))
        # pdb.set_trace()
        # result_M = (n_correct_M.item() * 100.0 / (n_items))  # scalar
        # result_D = (n_correct_D.item() * 100.0 / (n_items))
        result = (n_correct.item() * 100.0 / (n_items))
        # if result > 100:
        #     pdb.set_trace()

        if b == 69000:
            pdb.set_trace()
        # return result_M, result, RMSE
        return result, RMSE

# def accuracy2(model, train_f, train_rf, train_p, train_M, pct_close):
def accuracy2(model, train_f, train_rf, train_M, pct_close):
  n_items = len(train_M)
  # pdb.set_trace()
  X1 = T.Tensor(train_f)  # 2-d Tensor
  X2 = T.Tensor(train_rf)
  # X3 = T.Tensor (train_p)
  # X3 = X3.reshape(-1,1)       # reshaping to 2-d Tensor
  Y = T.Tensor(train_M)  # actual as 1-d Tensor
  # oupt = model(X1, X2, X3)    # all predicted as 2-d Tensor
  oupt = model(X1, X2)
  pred = oupt.view(n_items)  # all predicted as 1-d
  RMSE_test = m.sqrt(T.mean((Y - pred) ** 2))
  MAPE_test = 100 * (T.mean((T.abs(Y - pred) / Y)))
  n_correct = T.sum((T.abs(pred - Y) < T.abs(pct_close * Y)))
  result = (n_correct.item() * 100.0 / n_items)  # scalar
  # pdb.set_trace()
  return result, RMSE_test, MAPE_test

# MLP based model
class Net(T.nn.Module):
    def __init__(self, n_inp, n_hid1, n_hid2, n_out):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(n_inp, n_hid1)
        self.hid2 = T.nn.Linear(n_hid1, n_hid2)
        self.oupt = T.nn.Linear(n_hid2, n_out)
        self.dropout = T.nn.Dropout()

    # initializing the weights and biases

        # T.nn.init.xavier_uniform_(self.hid.weight, gain = 0.05)
        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        # T.nn.init.xavier_uniform_(self.oupt.weight, gain = 0.05)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    # def forward(self, X1,X2,X3):
    def forward(self, X):
        # x = T.cat((X1, X2, X3), dim=1) # concatenating the input vectors
        # after removing del_P as input, we only use 2 input vectors
        # x = T.cat((X1, X2), dim=1)
        # pdb.set_trace()
        z = T.tanh(self.hid1(X))
        z = T.tanh(self.hid2(z))
        z = self.oupt(z)  # no activation, aka Identity()
        return z

# def SSE(out, target):
#     return (target-out)*(target-out)

if __name__ == '__main__':

    T.manual_seed(1);  np.random.seed(1)

    ###################################################################################################################
        ###################      1. Loading the delw and delw_dot data from the mat file      #######################

    '''
        file_freq = mat file with all the frequency data input stacked in row along with del_p and M
        file_rocof = mat file with all the rocof data input stacked in row along with del_p and M
        
        loading() returns the array of freq_data and rocof_data
        separate_dataset() returns the separate training and testing dataset
    '''
    path = ".\\data files\\excitation_10\\manipulated\\"
    file_freq = path + 'freq_norm.mat'
    file_rocof = path + 'rocof_norm.mat'
    freq_data, rocof_data = loading(file_freq, file_rocof)
    train_f_rf, train_M_D, test_f_rf, test_M_D = separate_dataset(freq_data, rocof_data)
    # pdb.set_trace()
    ###################################################################################################################
                         ###################      2. creating the model      #######################
    '''
        n_inp (number of input nodes): calculated the number of inputs based on number of input features 
        n_hid (number of hidden nodes): followed the general convention for 8-3-8 i.e. 2^n - n - 2^n
        n_out (number of output nodes): calculated the number of output to be provided from the training dataset  
    
    '''
    # pdb.set_trace()
    n_inp = len(train_f_rf[0])
    n_hid1 = 15
    # pdb.set_trace()
    n_hid2 = n_hid1
    # n_out = len(train_M_D[0])
    n_out = 1
    net = Net(n_inp, n_hid1, n_hid2, n_out)
    ###################################################################################################################
                         ###################      3. Training the model      #######################
    net = net.train()
    bat_size = 10
    loss_func1 = T.nn.MSELoss()
    loss_func2 = T.nn.MSELoss()
    optimizer = T.optim.SGD(net.parameters(), lr=0.001)
    n_items = len(train_f_rf)
    # pdb.set_trace()
    batches_per_epoch = n_items // bat_size
    # max_batches = 1000 * batches_per_epoch
    max_batches = 70000
    print("Starting training")
    # weight_ih = []                # storing the weights from input to hidden
    weight_ho = []                  # storing the weights from hidden unit to output unit
    output_avg = []                 # storing the average output of the model
    train_losses = []               # storing the training losses
    val_losses = []                 # storing the validation losses

    min_RMSE = 100
    min_MAPE = 100
    min_batch_loss = 100
    min_R_epoch = 100
    min_M_epoch = 100
    min_B_epoch = 100

    for b in range(max_batches):
        curr_bat = np.random.choice(n_items, bat_size, replace=False)
        X = T.Tensor(train_f_rf[curr_bat])
        # Y = T.Tensor(train_M_D[curr_bat])
        Y = T.Tensor(train_M_D[curr_bat]).view(bat_size, 1)
        # pdb.set_trace()
        optimizer.zero_grad()
        # pdb.set_trace()
        oupt = net(X)
        # loss_obj_M = T.sqrt(loss_func1(oupt[:,0], Y[:,0]))
        # loss_obj_D = T.sqrt(loss_func2(oupt[:,1], Y[:,1]))
        # loss_obj = (loss_obj_M + loss_obj_D)
        loss_obj = loss_func1(oupt,Y)
        loss_obj.backward()
        optimizer.step()
        # pdb.set_trace()
        # weight_ih.append(np.reshape(net.hid1.weight.data.clone().cpu().numpy(), (1, n_inp * n_hid1)))
        weight_ho.append(np.reshape(net.oupt.weight.data.clone().cpu().numpy(), (1, n_hid2 * n_out)))

        net = net.eval()
        # acc_M, acc_D, RMSE = accuracy(net, test_f_rf, test_M_D, 0.1, b)
        acc_D, RMSE = accuracy(net, test_f_rf, test_M_D, 0.1)
        # val_losses.append([loss])
        net = net.train()

        if b % 1000 == 0:
            print("batch = %6d" % b, end="")
            print("  train loss = %7.4f" % loss_obj.item(), end="")
            # net = net.eval()
            # acc_M, acc_D, RMSE = accuracy(net, test_f_rf, test_M_D, 0.1, b)
            # # val_losses.append([loss])
            # net = net.train()
            # print("  val loss = %7.4f" % loss, end="")
            # print("  accuracy_M = %0.2f%%" % acc_M, end="")
            print("  accuracy_D = %0.2f%%" % acc_D, end="")
            # print("  MAPE = %0.3f%%" % MAPE, end="")
            print("  RMSE = %7.4f" % RMSE)

            # if loss_obj.item() < min_batch_loss:
            #     min_batch_loss = loss_obj.item()
            #     min_B_epoch = b

            if RMSE < min_RMSE:
                min_RMSE = RMSE
                min_R_epoch = b

            # if MAPE < min_MAPE:
            #     min_MAPE = MAPE
            #     min_M_epoch = b

            train_losses.append([loss_obj.item()])
            val_losses.append([RMSE])
    # avg_loss = np.concatenate((avg_loss,losses), axis = 1)
    print("Training complete \n")

    # label_graph = ['train_loss', 'val_loss', 'fitted_train_loss', 'fitted_val_loss']
    label_graph = ['train_loss','fitted_train_loss']
    losses = np.squeeze(train_losses)
    # val_losses = np.squeeze(val_losses)
    plt.figure()

    t_x = np.arange(len(losses))
    plt.scatter(t_x, losses, label=label_graph[0], marker = 'x', c = '#1f77b4', alpha = 0.5)
    poly = np.polyfit(t_x, losses, 5)
    losses = np.poly1d(poly)(t_x)
    plt.plot(t_x, losses, label=label_graph[1], c = 'red', linewidth = '5')

    # v_x = np.arange(len(val_losses))
    # plt.scatter(v_x, val_losses, label=label_graph[1], marker = '>', c = '#9467bd', alpha = 0.5)
    # poly = np.polyfit(v_x, val_losses, 5)
    # val_losses = np.poly1d(poly)(v_x)
    # plt.plot(v_x, val_losses, label=label_graph[3], c = 'green', linewidth = '5')

    plt.ylabel("Mean Squared Error", **axis_font)
    plt.xlabel("Number of batches in entire epochs (x1000)", **axis_font)
    # plt.title("Batch training loss vs number of batch", **title_font)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.legend()
    plt.savefig('./output/output_feb19/batch_loss.png', dpi=600, bbox_inches='tight')
    plt.show()

    weight_ho = np.reshape(weight_ho, (np.shape(weight_ho)[0], np.shape(weight_ho)[2]))
    weights_ho_num = int(np.shape(weight_ho)[1])
    for i in range(0, weights_ho_num):
        plt.plot(weight_ho[:, i])
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("weights from hidden to output layer", **axis_font)
    plt.xlabel("Number of batches in entire epochs", **axis_font)
    plt.xlim(0, max_batches)
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig('./output/output_feb19/h2o_weight.png', dpi=600, bbox_inches='tight')
    plt.show()



    # print(" min batch loss = {} at {} batch \n".format(min_batch_loss, min_B_epoch))
    print(" min RMSE = {} at {} batch \n".format(min_RMSE, min_R_epoch))
    # print(" min MAPE = {} at {} batch \n".format(min_MAPE, min_M_epoch))
    # pdb.set_trace()
    # avg_loss = np.mean(avg_loss[:,1:], axis = 1)
    #
    # fig, axx = plt.subplots()
    # axx.plot(avg_loss)
    # axx.set_xlim(0, max_batches) # apply the x-limits
    # # axx.set_ylim(0, 100) # apply the y-limits
    # plt.ylabel("Mean Squared Error", **axis_font)
    # plt.xlabel("Number of batches", **axis_font)
    # # plt.title("Batch training loss vs number of batch", **title_font)
    # plt.grid(linestyle='-', linewidth=0.5)
    # plt.xticks(fontsize = 12)
    # plt.yticks(fontsize = 12)
    # plt.rcParams['agg.path.chunksize'] = 1000
    # # plt.savefig('./output/for_1500_samples/batch_loss.png', dpi = 600, bbox_inches='tight')
    # plt.show()
    # pdb.set_trace()

    ###################################################################################################################
                    ###################      4. Evaluating the model      #######################

    # net = net.eval()  # set eval mode
    # # acc1, RMSE_test, MAPE_test = accuracy2(net, test_f, test_rf, test_p, test_M, 0.05)
    # # acc2, _, _ = accuracy2(net, test_f, test_rf, test_p, test_M, 0.1)
    # # acc3, _, _ = accuracy2(net, test_f, test_rf, test_p, test_M, 0.15)
    # acc1, RMSE_test, MAPE_test = accuracy2(net, test_f, test_rf, test_M, 0.05)
    # acc2, _, _ = accuracy2(net, test_f, test_rf, test_M, 0.1)
    # acc3, _, _ = accuracy2(net, test_f, test_rf, test_M, 0.15)
    # print("Accuracy on test data with 0.05 tolerance = %0.2f%%" % acc1)
    # print("Accuracy on test data with 0.1 tolerance = %0.2f%%" % acc2)
    # print("Accuracy on test data with 0.15 tolerance = %0.2f%%" % acc3)
    # print("MAPE on test data  = %0.3f%%" % MAPE_test)
    # print("RMSE on test data  = %7.4f" % RMSE_test)
    # pdb.set_trace()
    ###################################################################################################################
                    ###################      5. Using the model      #######################

    eval_file = h5py.File(path + 'eval_data.mat', 'r')
    eval_var = eval_file.get('eval_data')
    # f_var = np.array(eval_var[0:201, :]).T
    # rocof_var = np.array(eval_var[201:402, :]).T
    # power_var = np.array(eval_var[150, :])
    # pdb.set_trace()
    # X1 = T.Tensor(f_var)
    # X2 = T.Tensor(rocof_var)
    # X3 = T.Tensor(power_var).view(-1, 1)
    # y = net(X1, X2, X3)
    # X_12 = T.cat((X1, X2), dim=1)
    X_eval = np.array(eval_var[0:402, :]).T
    X_e = T.Tensor(X_eval)
    y = net(X_e)
    # print(y)
    pdb.set_trace()

    ###################################################################################################################
                        ###################      6. Plotting the results      #######################

    # weight_ih = np.reshape(weight_ih, (np.shape(weight_ih)[0], np.shape(weight_ih)[2]))
    # weights_ih_num = int(np.shape(weight_ih)[1])
    # for i in range(0, weights_ih_num):
    #     plt.plot(weight_ih[:, i])
    # plt.grid(linestyle='-', linewidth=0.5)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylabel("weights from input to hidden layer", **axis_font)
    # plt.xlabel("Number of batches in entire epochs", **axis_font)
    # plt.xlim(0, max_batches)
    # plt.rcParams['agg.path.chunksize'] = 10000
    # plt.savefig('./output/output_feb19/i2h_weight.png', dpi = 600, bbox_inches='tight')
    # plt.show()

    weight_ho = np.reshape(weight_ho, (np.shape(weight_ho)[0], np.shape(weight_ho)[2]))
    weights_ho_num = int(np.shape(weight_ho)[1])
    for i in range(0, weights_ho_num):
        plt.plot(weight_ho[:, i])
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("weights from hidden to output layer", **axis_font)
    plt.xlabel("Number of batches in entire epochs", **axis_font)
    plt.xlim(0, max_batches)
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig('./output/output_feb19/h2o_weight.png', dpi=600, bbox_inches='tight')
    plt.show()
    # pdb.set_trace()
    #
    # # ############  full output plot  #############
    # #
    # output_full_array = output_full.data.cpu().numpy()
    # fig, ax = plt.subplots()
    # ax.plot(output_full_array)
    # # ax.plot(output)
    # ax.set_xlim(0, max_batches) # apply the x-limits
    # # ax.set_ylim(0, 10) # apply the y-limits
    # # plt.hlines(10, 0, 6000, colors='r', linestyles='dashed', linewidth=3)
    # plt.grid(linestyle='-', linewidth=0.5)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylabel("Estimated Inertia", **axis_font)
    # plt.xlabel("Number of batches in entire epochs", **axis_font)
    # # plt.title("Trained output in entire batch", **title_font)
    # # plt.show()
    # # axins = zoomed_inset_axes(ax, 2.5, loc=4) # zoom-factor: 2.5, location: upper-left
    # # axins.plot(output_full_array)
    # # x1, x2, y1, y2 = 25000, 30000, 6, 8 # specify the limits
    # # axins.set_xlim(x1, x2) # apply the x-limits
    # # axins.set_ylim(y1, y2) # apply the y-limits
    # # plt.yticks(visible=False)
    # # plt.xticks(visible=False)
    # # axins.xaxis.set_visible('False')
    # # axins.yaxis.set_visible('False')
    # # mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    # # plt.grid()
    # plt.savefig('./output/output_feb19/estimated_output.png', dpi = 600)
    # plt.show()
    #
    # '''
    #     ############  averaged output plot  #############
    #     fig, ax = plt.subplots()
    #     ax.plot(output_avg)
    #     # ax.set_xlim(0, 600) # apply the x-limits
    #     # ax.set_ylim(0, 12) # apply the y-limits
    #     # plt.hlines(10, 0, 600, colors='r', linestyles='dashed', linewidth=3)
    #     plt.grid(linestyle='-', linewidth=0.5)
    #     plt.xticks(fontsize=14)
    #     plt.yticks(fontsize=14)
    #     plt.ylabel("Estimated Inertia", **axis_font)
    #     plt.xlabel("Number of batches", **axis_font)
    #     plt.title("estimated system inertia with increase batch number", **title_font)
    #     # plt.show()
    #     axins = zoomed_inset_axes(ax, 2.5, loc=4) # zoom-factor: 2.5, location: upper-left
    #     axins.plot(output_avg)
    #     x1, x2, y1, y2 = 0, 2500, 4, 6 # specify the limits
    #     axins.set_xlim(x1, x2) # apply the x-limits
    #     axins.set_ylim(y1, y2) # apply the y-limits
    #     plt.yticks(visible=False)
    #     plt.xticks(visible=False)
    #     # axins.xaxis.set_visible('False')
    #     # axins.yaxis.set_visible('False')
    #     mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    #     plt.grid()
    #     # plt.savefig('output_avg.png', dpi = 600)
    #     plt.show()
    # '''
    #
    ############ plotting loss #############
    fig, axx = plt.subplots()
    axx.plot(losses)
    axx.set_xlim(0, max_batches) # apply the x-limits
    # axx.set_ylim(0, 100) # apply the y-limits
    plt.ylabel("Mean Squared Error", **axis_font)
    plt.xlabel("Number of batches in entire epochs", **axis_font)
    # plt.title("Batch training loss vs number of batch", **title_font)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.savefig('./output/output_feb19/batch_loss.png', dpi = 600, bbox_inches='tight')
    plt.show()

    #
    #
    #
