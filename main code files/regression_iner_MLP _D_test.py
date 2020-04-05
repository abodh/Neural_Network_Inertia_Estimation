'''
Author: Abodh Poudyal
MSEE, South Dakota State University
Last updated: April 2, 2020
'''

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import torch as T
import pdb

from data_loading_D_test import loading, separate_dataset
from torch.utils.data import Dataset, DataLoader

class freq_data(Dataset):
    # Constructor
    def __init__(self, path):
        file_freq = path + 'freq_norm.mat'
        file_rocof = path + 'rocof_norm.mat'
        freq_data, rocof_data = loading(file_freq, file_rocof)
        self.x, self.y = separate_dataset(freq_data, rocof_data)
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # Return the length
    def __len__(self):
        return self.len

def accuracy(model, validation_loader, pct_close, eval = False):

    with T.no_grad():
        val_correct = []
        val_loss_func = []
        n_items = 0
        for idx, (test_f_rf,test_M_D) in enumerate(validation_loader):
            n_items += len(test_M_D)
            X = test_f_rf.float()
            Y = test_M_D.float().view(-1,1)    # reshaping to 2-d Tensor
            oupt = model(X)                    # all predicted as 2-d Tensor
            loss = criterion(oupt,Y)
            n_correct = T.sum((T.abs(oupt - Y) < T.abs(pct_close * Y)))
            val_correct.append(n_correct)
            val_loss_func.append(loss)

        loss_func = sum(val_loss_func)/ len(val_loss_func)
        result = (sum(val_correct) * 100.0 / n_items)
        RMSE_loss = T.sqrt(loss_func)

        # observing the result when set to eval mode
        if (eval):
            print('Predicted test output for random batch = {}, actual output = {} with accuracy of {:.2f}% '
                  'and RMSE = {:.6f}'.format(oupt, Y, result, RMSE_loss))
        return result, RMSE_loss, loss_func

# MLP based model
class Net(T.nn.Module):
    def __init__(self, n_inp, n_hid1, n_hid2, n_out, dropout_rate, weight_ini, dropout_decision=False):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(n_inp, n_hid1)
        self.hid2 = T.nn.Linear(n_hid1, n_hid2)
        self.oupt = T.nn.Linear(n_hid2, n_out)
        self.dropout_decision = dropout_decision
        self.dropout = T.nn.Dropout(dropout_rate)

    # initializing the weights and biases
        T.nn.init.xavier_uniform_(self.hid1.weight, gain = weight_ini)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight, gain = weight_ini)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight, gain = weight_ini)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, X):
        z = T.tanh(self.hid1(X))
        if (dropout_decision):
            z = self.dropout(z)
        z = T.tanh(self.hid2(z))
        z = self.oupt(z)  # no activation, aka Identity()
        return z

def testing(model_path,
            data_path,
            counter,
            net,
            load_model = True):

    net = net.eval()
    with T.no_grad():
        eval_file = h5py.File(data_path + 'eval_data.mat', 'r')
        eval_var = eval_file.get('eval_data')
        X_eval = T.Tensor(np.array(eval_var[0:-2, :]).T)            # a 2-d tensor
        Y_eval = T.Tensor(np.array(eval_var[-1, :]).T).view(-1, 1)  # converting to a 2-d tensor

        if (load_model):
            test_loss = []  # accumulating the losses on different models
            i = 0
            for filename in os.listdir(model_path):
                if filename.endswith(".pth"):
                    net.load_state_dict(T.load(model_path + '/' + filename))
                    net.eval()
                    y = net(X_eval)
                    loss_test = T.sqrt(criterion(y, Y_eval))
                    test_loss.append(loss_test.item())
                    print ('test result at epoch {} is y = {}'.format(counter[i], y.view(len(y))))
                    i =  i + 1
                    continue
                else:
                    continue
            min_val_epoch = counter[np.argmin(test_loss)]
            test_RMSE = min(test_loss)
            print('the best model is at epoch {} which gives a loss of {:.6f} \n'
                  .format(min_val_epoch, test_RMSE))
        else:
            y = net(X_eval)
            test_loss = criterion(y, Y_eval)
            test_RMSE = T.sqrt(test_loss)
            print ('Actual output = {} and Predicted test output = {} with RMSE = {:.6f}'.format(Y_eval, y, test_RMSE))
    return test_loss, test_RMSE

# resets weights for different learning rates
def weight_init(m):
    if isinstance(m, T.nn.Linear):
        m.reset_parameters()

if __name__ == '__main__':

    # manual seed to reproduce same results every time
    T.manual_seed(1);  np.random.seed(1)

    # setting the parameters
    epoch = 489                # number of epochs -> in 1 epoch all of the training data are used
    mini_batch = 10             # number of mini-batches -> subset of the training data
    learning_rate = 1e-3        # SGD learning rate -> considers SGD as optimizer
    momentum = 0              # SGD momentum term -> considers SGD as optimizer
    n_hidden1 = 10              # number of hidden units in first hidden layer
    n_hidden2 = 10              # number of hidden units in second hidden layer
    n_output = 1                # number of output units
    frac_train = 0.8            # fraction of data to be used as training set
    dropout_rate = 0.5          # dropout rate -> remember to set dropout_decision as True
    weight_initializer = 0.05   # weight initializer -> initializes between [-x,x)
    dropout_decision = False     # do you want to dropout or not?

    tolerance = 0.1             # tolerance for the estimated value
                                # -> 0.1 means the output around 10% is considers to be correct

    save_model = False          # set True when you want to save models for specific conditions
    load_model = False          # set True when you want to load the saved models for specific conditions
    model_path = './models'     # path to the saved model

    data_path = "C:\\Users\\abodh\\Box Sync\\Box Sync\\Spring 2020\\inertia project\\" \
                "Neural-Network-Regression\\data files\\varying both_M_P_posneg_pulse\\manipulated\\"

    ###################################################################################################################
                    ###################      2. creating the model      #######################

    dataset = freq_data(data_path)           # loads data from the freq_data class (dataset class -> awesome in pytorch)
    print('the length of the dataset = ', len(dataset))

    train_num = int(frac_train * len(dataset))     # number of data for training
    test_num = len(dataset) - train_num            # number of data for validating
    max_batches = epoch * int(train_num / mini_batch)

    # splitting into training and validation dataset
    training, validation = T.utils.data.random_split(dataset,(train_num,test_num))

    # load separate training and validating dataset -> repeat !!! dataset and dataloader are awesome in pytorch)
    train_loader = DataLoader(training, batch_size = mini_batch, shuffle = True)
    validation_loader = DataLoader(validation, batch_size = 100, shuffle = False)

    n_inp = len(training[0][0])
    n_hid1 = n_hidden1
    n_hid2 = n_hidden2
    n_out = n_output

    # call your neural network model right here
    net = Net(n_inp, n_hid1, n_hid2, n_out, dropout_rate, weight_initializer, dropout_decision)
    print (net)

    ##################################################################################################################
                        #############      3. Training the model      #######################

    net = net.train()                   # set the network to training mode
    criterion = T.nn.MSELoss()          # set the loss criterion
    optimizer = T.optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)

    print("Starting training")

    weight_ih = []                  # storing the weights from input to hidden
    weight_ho = []                  # storing the weights from hidden unit to output unit
    train_losses = []               # storing the training losses
    val_losses = []                 # storing the validation losses
    test_losses = []                # storing the validation losses
    min_val_RMSE = 1e5              # initializing to find min validation RMSE
    min_R_epoch = 1e5               # initializing to find the epoch with min validation RMSE
    counter = []                    # to store the different epochs that gives validation accuracy > 90%

    # uncomment below to test on different learning rates
    ###### important: comment out the criterion, optimizer, and net initialized above #####
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # lr_train_loss = []
    # lr_val_loss = []
    # for iter_learn, l_rate in enumerate(learning_rates):
    #     T.manual_seed(1); np.random.seed(1)
    #     # net = net.train()
    #     net.apply(weight_init)
    #     optimizer = T.optim.SGD(net.parameters(), lr=l_rate, momentum=0.5)
    #     weight_ho = []

    for ep in range(epoch):
        train_loss = []     # saves the batch training loss for each epoch

        for batch_idx, (data, target) in enumerate(train_loader):
            X = data.float()                        # pytorch input should be float -> awkward right?
            Y = target.float().view(-1, 1)          # converting into 2-d tensor
            optimizer.zero_grad()                   # making the gradient zero before optimizing
            oupt = net(X)                           # neural network output
            loss_obj = criterion(oupt,Y)            # loss calculation
            loss_obj.backward()                     # backpropagation
            optimizer.step()                        # remember w(t) = w(t-1) - alpha*cost ??
                                                    # if not, we are just updating weights here

            # saving the weights from input to hidden and hidden to output
            # weight_ih.append(np.reshape(net.hid1.weight.data.clone().cpu().numpy(), (1, n_inp * n_hid1)))
            weight_ho.append(np.reshape(net.oupt.weight.data.clone().cpu().numpy(), (1, n_hid2 * n_out)))
            train_loss.append(loss_obj.item())      # batch losses -> length = number of batches in an epoch

        # getting the training loss for each epoch
        train_loss_avg = sum(train_loss) / len(train_loss)      # batch averaging
        train_losses.append([train_loss_avg])                   # saving average batch loss for each epoch

        # testing validation set after training all the batches
        net = net.eval()                            # set the network to evaluation mode
        val_acc, val_RMSE, vali_loss = accuracy(net, validation_loader, tolerance, eval=False)
        val_losses.append([vali_loss.item()])       # validation loss on entire samples for each epoch

        # find the epoch that gives minimum validation loss
        if val_RMSE < min_val_RMSE:
            min_val_RMSE = val_RMSE
            min_R_epoch = ep

        # set the network to training mode after validation
        net = net.train()

        # if we are willing to test the models on testing data that gives validation accuracy > 90%
        if (save_model) and val_RMSE <= 0.5:
                counter.append((ep))
                T.save(net.state_dict(),'./models/model{}.pth'.format(ep))

        print("epoch = %d" % ep, end="")
        print("  train loss = %7.4f" % train_loss_avg, end="")
        print("  val_accuracy = %0.2f%%" % val_acc, end="")
        print("  val_RMSE = %7.4f" % val_RMSE, end="")
        print("  val_loss = %7.4f" % vali_loss.item())  # similar to RMSE, can comment out if unnecessary


        # uncomment below if you are testing for different learning rates
        # plotted loss after each learning rate test
        # print(" min RMSE = {} at {} batch \n".format(min_RMSE, min_R_epoch))
        # weight_ho = np.reshape(weight_ho, (np.shape(weight_ho)[0], np.shape(weight_ho)[2]))
        # weights_ho_num = int(np.shape(weight_ho)[1])
        # for i in range(0, weights_ho_num):
        #     plt.plot(weight_ho[:, i])
        # plt.grid(linestyle='-', linewidth=0.5)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.ylabel("weights from hidden to output layer", **axis_font)
        # plt.xlabel("Number of batches in entire epochs", **axis_font)
        # plt.xlim(0, epoch)
        # plt.rcParams['agg.path.chunksize'] = 10000
        # plt.savefig('C:/Users/abodh/Box Sync/Box Sync/Spring 2020/inertia project/Neural-Network-Regression/output/'
        #             'output_feb19/iteration{}'.format(iter_learn), dpi=600, bbox_inches='tight')
        # plt.close()

        # averaged the loss to test on learning rates
        # lr_train_loss.append([sum(train_losses) / len(train_losses)])
        # lr_val_loss.append([sum(val_losses) / len(val_losses)])

        # train_losses.append([loss_obj.item()])
        # val_losses.append([vali_loss.item()])

    np.savetxt('train_5.csv', train_losses, delimiter=',')
    np.savetxt('val_5.csv', val_losses, delimiter=',')
    print("Training complete \n")
    print(" min RMSE = {} at {} epoch \n".format(min_val_RMSE, min_R_epoch))
    ###################################################################################################################
                    ###################      4. Evaluating the model (validation) #######################

    net = net.eval()  # set eval mode
    acc_val, val_RMSE, _ = accuracy(net, validation_loader, tolerance, eval=True)
    print('validation accuracy with {} tolerance = {:.2f} and RMSE = {:.6f}\n'
          .format(tolerance, acc_val, val_RMSE))

    ###################################################################################################################
                    ###################      5. Using the model (testing)     #######################

    test_loss, test_RMSE = testing(model_path, data_path, counter, net, load_model)

    ###################################################################################################################
                        ###################      6. Plotting the results      #######################

    # Set the font dictionaries (for plot title and axis titles)
    title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '16'}

    # uncomment below to plot the losses along with different learning rates

    # losses = np.squeeze(lr_train_loss)
    # val_losses = np.squeeze(lr_val_loss)

    # Plot the training loss and validation loss for different learning rates
    # pdb.set_trace()
    # plt.semilogx(np.array(learning_rates), losses, label='training loss/total Loss')
    # plt.semilogx(np.array(learning_rates), val_losses, label='validation cost/total Loss')
    # plt.ylabel('Cost\ Total Loss')
    # plt.xlabel('learning rate')
    # plt.legend()
    # plt.savefig('./losses{}'.format(iter_learn), dpi=600, bbox_inches='tight')
    # pdb.set_trace()

    label_graph = ['train_loss', 'val_loss', 'fitted_train_loss', 'fitted_val_loss', 'test_loss']
    losses = np.squeeze(train_losses)
    val_losses = np.squeeze(val_losses)
    plt.figure()

    t_x = np.arange(len(losses))
    v_x = np.arange(len(val_losses))

    plt.plot(t_x, losses, label=label_graph[0], c='blue', linewidth='5')
    plt.plot(v_x, val_losses, label=label_graph[1], c='green', linewidth='2')
    # uncomment below if you want to have a vertical line at your best epoch
    # plt.axvline(x=min_R_epoch, color='r', linestyle='--', linewidth=3)
    plt.ylabel("Mean Squared Error", **axis_font)
    plt.xlabel("Number of epochs", **axis_font)
    plt.xlim(0, len(t_x))
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.legend()
    plt.savefig('./batch_loss.png', dpi=600, bbox_inches='tight')
    plt.show()
    pdb.set_trace()

    # uncomment below to plot polyfit on losses
    # # t_x = np.arange(len(losses))
    # plt.scatter(t_x, losses, label=label_graph[0], marker = 'x', c = '#1f77b4', alpha = 0.5)
    # poly = np.polyfit(t_x, losses, 4)
    # losses = np.poly1d(poly)(t_x)
    # plt.plot(t_x, losses, label=label_graph[2], c = 'red', linewidth = '5')
    #
    # v_x = np.arange(len(val_losses))
    # plt.scatter(v_x, val_losses, label=label_graph[1], marker = '>', c = '#9467bd', alpha = 0.5)
    # poly = np.polyfit(v_x, val_losses, 4)
    # val_losses = np.poly1d(poly)(v_x)
    # plt.plot(v_x, val_losses, label=label_graph[3], c = 'green', linewidth = '5')
    #
    # plt.ylabel("Mean Squared Error", **axis_font)
    # plt.xlabel("Number of epochs", **axis_font)
    # # plt.title("Batch training loss vs number of batch", **title_font)
    # plt.grid(linestyle='-', linewidth=0.5)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.rcParams['agg.path.chunksize'] = 1000
    # plt.legend()
    # plt.savefig('./batch_loss.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close()

    # uncomment below to plot input to hidden weights
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
    # plt.savefig('./wih.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close()

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
    plt.savefig('./who', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()