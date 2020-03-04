import numpy as np
import h5py
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import matplotlib as mpl
import math as m
import torch as T
from data_loading import loading, separate_dataset
import pdb

# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
axis_font = {'fontname': 'Arial', 'size': '16'}

# def accuracy(model, train_f, train_rf, train_p, train_M, pct_close):
def accuracy(model, train_f, train_rf, train_M, pct_close):
    n_items = len(train_M)
    X1 = T.Tensor(train_f)        # 2-d Tensor
    X2 = T.Tensor(train_rf)       # 2-d Tensor
    # X3 = T.Tensor (train_p)
    # X3 = X3.reshape(-1,1)       # reshaping to 2-d Tensor
    Y = T.Tensor(train_M)         # actual as 1-d Tensor
    # oupt = model(X1, X2, X3)    # all predicted as 2-d Tensor
    oupt = model(X1, X2)
    pred = oupt.view(n_items)     # all predicted as 1-d
    loss_val = loss_func(oupt, Y)
    RMSE = m.sqrt(T.mean((Y-pred)**2))
    MAPE = 100 * (T.mean((T.abs(Y-pred)/Y)))
    # pdb.set_trace()
    n_correct = T.sum((T.abs(pred - Y) < T.abs(pct_close * Y)))
    result = (n_correct.item() * 100.0 / n_items)  # scalar
    # pdb.set_trace()
    return result, RMSE, MAPE, loss_val.item()

# def accuracy2(model, train_f, train_rf, train_p, train_M, pct_close):
def accuracy2(model, train_f, train_rf, train_M, pct_close):
  n_items = len(train_M)
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
  pdb.set_trace()
  return result, RMSE_test, MAPE_test

# MLP based model
class Net(T.nn.Module):
    def __init__(self, n_inp, n_hid1, n_hid2, n_out):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(n_inp, n_hid1)
        self.hid2 = T.nn.Linear(n_hid1, n_hid2)
        self.oupt = T.nn.Linear(n_hid2, n_out)


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
    def forward(self, X1, X2):
        # x = T.cat((X1, X2, X3), dim=1) # concatenating the input vectors
        # after removing del_P as input, we only use 2 input vectors
        x = T.cat((X1, X2), dim=1)
        z = T.tanh(self.hid1(x))
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
    path = ".\\data files\\noisy_data_1000\\manipulated\\"
    file_freq = path + 'freq_norm.mat'
    file_rocof = path + 'rocof_norm.mat'
    freq_data, rocof_data = loading(file_freq, file_rocof)
    train_f, train_rf, train_p, train_M, test_f, test_rf, test_p, test_M = separate_dataset(freq_data, rocof_data)

    ###################################################################################################################
                         ###################      2. creating the model      #######################
    '''
        n_inp (number of input nodes): calculated the number of inputs based on number of input features 
        n_hid (number of hidden nodes): followed the general convention for 8-3-8 i.e. 2^n - n - 2^n
        n_out (number of output nodes): calculated the number of output to be provided from the training dataset  
    
    '''

    # n_inp = len(train_f[0]) + len(train_rf[0]) + len([train_p[0]])
    n_inp = len(train_f[0]) + len(train_rf[0])
    n_hid1 = int(m.log(n_inp)/m.log(2)) + 3
    # pdb.set_trace()
    n_hid2 = n_hid1
    n_out = len([train_M[0]])
    net = Net(n_inp, n_hid1, n_hid2, n_out)

    ###################################################################################################################
                         ###################      3. Training the model      #######################
    net = net.train()
    bat_size = 10
    loss_func = T.nn.MSELoss()
    optimizer = T.optim.SGD(net.parameters(), lr=5e-3)
    n_items = len(train_f)
    batches_per_epoch = n_items // bat_size
    # max_batches = 1000 * batches_per_epoch
    max_batches = 57000
    print("Starting training")
    output_full = T.tensor([])      # capturing the full output of the model in total batches
    # weight_ih = []                # storing the weights from input to hidden
    weight_ho = []                  # storing the weights from hidden unit to output unit
    output_avg = []                 # storing the average output of the model
    losses = []                     # storing the batch losses
    val_losses = []
    min_RMSE = 100
    min_MAPE = 100
    min_batch_loss = 100
    min_R_epoch = 100
    min_M_epoch = 100
    min_B_epoch = 100
    for b in range(max_batches):
        curr_bat = np.random.choice(n_items, bat_size, replace=False)
        X1 = T.Tensor(train_f[curr_bat])
        X2 = T.Tensor(train_rf[curr_bat])
        # X3 = T.Tensor(train_p[curr_bat]).view(bat_size, 1)
        Y = T.Tensor(train_M[curr_bat]).view(bat_size, 1)
        # pdb.set_trace()
        optimizer.zero_grad()
        # oupt = net(X1,X2,X3)
        oupt = net(X1, X2)
        # oupt_numpy = oupt.data.cpu().numpy()
        output_full = T.cat((output_full, oupt), 0)
        # output_avg.append(np.mean(oupt_numpy))
        loss_obj = T.sqrt(loss_func(oupt, Y))
        loss_obj.backward()
        optimizer.step()
        # weight_ih.append(np.reshape(net.hid1.weight.data.clone().cpu().numpy(), (1, n_inp * n_hid1)))
        weight_ho.append(np.reshape(net.oupt.weight.data.clone().cpu().numpy(), (1, n_hid2 * n_out)))
        # if b % (max_batches // 20) == 0:
        if b % 1000 == 0:
            # print(output.size(), end="")
            print("batch = %6d" % b, end="")
            print("  train loss = %7.4f" % loss_obj.item(), end="")
            net = net.eval()
            acc, RMSE, MAPE, loss = accuracy(net, train_f, train_rf, train_M, 0.1)
            # val_losses.append([loss])
            net = net.train()
            print("  val loss = %7.4f" % loss, end="")
            print("  accuracy = %0.2f%%" % acc, end="")
            print("  MAPE = %0.3f%%" % MAPE, end="")
            print("  RMSE = %7.4f" % RMSE)

            if loss_obj.item() < min_batch_loss:
                min_batch_loss = loss_obj.item()
                min_B_epoch = b

            if RMSE < min_RMSE:
                min_RMSE = RMSE
                min_R_epoch = b

            if MAPE < min_MAPE:
                min_MAPE = MAPE
                min_M_epoch = b

            losses.append([loss_obj.item()])
            val_losses.append([RMSE])
    # avg_loss = np.concatenate((avg_loss,losses), axis = 1)
    print("Training complete \n")

    losses = np.squeeze(losses)
    val_losses = np.squeeze(val_losses)
    plt.figure()

    plt.plot(losses)
    t_x = np.arange(len(losses))
    poly = np.polyfit(t_x, losses, 6)
    losses = np.poly1d(poly)(t_x)
    plt.plot(t_x, losses)

    plt.plot(val_losses)
    v_x = np.arange(len(val_losses))
    poly = np.polyfit(v_x, val_losses, 6)
    val_losses = np.poly1d(poly)(v_x)
    plt.plot(v_x, val_losses)

    plt.ylabel("Mean Squared Error", **axis_font)
    plt.xlabel("Number of batches in entire epochs", **axis_font)
    # plt.title("Batch training loss vs number of batch", **title_font)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.rcParams['agg.path.chunksize'] = 1000
    # plt.savefig('./output/output_feb19/batch_loss.png', dpi=600, bbox_inches='tight')
    plt.show()
    pdb.set_trace()

    print(" min batch loss = {} at {} batch \n".format(min_batch_loss, min_B_epoch))
    print(" min RMSE = {} at {} batch \n".format(min_RMSE, min_R_epoch))
    print(" min MAPE = {} at {} batch \n".format(min_MAPE, min_M_epoch))

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

    net = net.eval()  # set eval mode
    # acc1, RMSE_test, MAPE_test = accuracy2(net, test_f, test_rf, test_p, test_M, 0.05)
    # acc2, _, _ = accuracy2(net, test_f, test_rf, test_p, test_M, 0.1)
    # acc3, _, _ = accuracy2(net, test_f, test_rf, test_p, test_M, 0.15)
    acc1, RMSE_test, MAPE_test = accuracy2(net, test_f, test_rf, test_M, 0.05)
    acc2, _, _ = accuracy2(net, test_f, test_rf, test_M, 0.1)
    acc3, _, _ = accuracy2(net, test_f, test_rf, test_M, 0.15)
    print("Accuracy on test data with 0.05 tolerance = %0.2f%%" % acc1)
    print("Accuracy on test data with 0.1 tolerance = %0.2f%%" % acc2)
    print("Accuracy on test data with 0.15 tolerance = %0.2f%%" % acc3)
    print("MAPE on test data  = %0.3f%%" % MAPE_test)
    print("RMSE on test data  = %7.4f" % RMSE_test)
    pdb.set_trace()
    ###################################################################################################################
                    ###################      5. Using the model      #######################

    eval_file = h5py.File(path + 'eval_data_with_noise.mat', 'r')
    eval_var = eval_file.get('eval_data')
    f_var = np.array(eval_var[0:75, :]).T
    rocof_var = np.array(eval_var[75:150, :]).T
    # power_var = np.array(eval_var[150, :])
    pdb.set_trace()
    X1 = T.Tensor(f_var)
    X2 = T.Tensor(rocof_var)
    # X3 = T.Tensor(power_var).view(-1, 1)
    # y = net(X1, X2, X3)
    y = net(X1, X2)
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


    #
    #
    #
