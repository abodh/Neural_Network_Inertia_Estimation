'''
Author: Abodh Poudyal (@abodh_ltd)
MSEE, South Dakota State University
Last updated: July 1, 2020
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pdb
from datetime import date, datetime
import os
import time

from data_loading import loading, separate_dataset, freq_data
from model import Net, Simple1DCNN
from utils import accuracy, testing
from torch.utils.data import DataLoader

# resets weights for different learning rates
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()

if __name__ == '__main__':
    # manual seed to reproduce same results every time
    torch.manual_seed(0);  np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting the parameters
    network = 'CNN'         # you can choose to train on MLP or CNN
    epoch = 200             # number of epochs -> in 1 epoch all of the training data are used
    mini_batch = 30         # number of mini-batches -> subset of the training data
    learning_rate = 1e-3    # SGD learning rate -> considers SGD as optimizer
    momentum = 0.5          # SGD momentum term -> considers SGD as optimizer
    n_hidden1 = 25          # number of hidden units in first hidden layer (for MLP)
    n_hidden2 = 25          # number of hidden units in second hidden layer (for MLP)
    n_output = 1            # number of output units
    frac_train = 0.80       # fraction of data to be used as training set
    dropout_rate = 0.2      # dropout rate -> remember to set dropout_decision as True
    weight_initializer = 0.05   # weight initializer -> initializes between [-x,x)
    dropout_decision = False    # do you want to dropout or not?
    w_lambda = 0.0005           # weight decay parameter

    tolerance = 0.1             # tolerance for the estimated value
    # -> 0.1 means the output around 10% is considers to be correct

    save_figs = False       # set True if you want to save figs and data
    save_model = False      # set True when you want to save models for specific conditions
    load_model = False      # set True when you want to load the saved models for specific conditions

    # data_path = "..\\..\\Neural-Network-Regression\\data files\\other data\\varying both_M_P_posneg_pulse" \
    #             "\\manipulated\\"
    data_path = "..\\..\\matlab files\\0.2Hz\\manipulated\\"  # set the path of the data file

    # loading the data
    dataset = freq_data(data_path)  # loads data from the freq_data class (dataset class -> awesome in pytorch)
    print('the length of the dataset = ', len(dataset))

    train_num = int(frac_train * len(dataset))  # number of data for training
    test_num = len(dataset) - train_num  # number of data for validating
    max_batches = epoch * int(train_num / mini_batch)

    '''' brute force search '''
    # hidden = [10, 25, 50, 60]
    # lr = [1e-4, 1e-3, 1e-2, 1e-1]
    # decay = [1e-4, 5e-4, 1e-3, 1e-2]
    # batch = [10, 20, 30, 50]
    # results = np.zeros((len(hidden) * len(lr) * len(decay) * len(batch), 6))
    # cnt = 0
    # for n_hidden1 in hidden:
    #     n_hidden2 = n_hidden1
    #     for learning_rate in lr:
    #         for w_lambda in decay:
    #             for mini_batch in batch:

    # creating a unique folder to save the output files
    str(date.today().strftime("%d/%m/%Y"))
    output_path = "../../Neural-Network-Regression/log/testing_models/" + str(date.today().strftime("%b-%d-%Y")) + \
                  str(datetime.now().strftime("-%H.%M.%S-")) \
                  + "h{}_lr{}_lam{}_bat_{}".format(n_hidden1, learning_rate, w_lambda, mini_batch)
    try:
        os.mkdir(output_path) # creates a directory based on current date and time
    except OSError:
        print("Creation of the directory %s failed" % output_path)

    # creating models folder if save_model is set to true
    if (save_model):
        os.mkdir(output_path + '/models')

    if (load_model):
        # path to the saved model from where it needs to be loaded
        model_path = output_path + '/models'
    else:
        model_path = ' '

    ###################################################################################################################
    ###################      2. creating the model      #######################

    # splitting into training and validation dataset
    training, validation = torch.utils.data.random_split(dataset, (train_num, test_num))

    # load separate training and validating dataset -> repeat !!! dataset and dataloader are awesome in pytorch)
    train_loader = DataLoader(training, batch_size=mini_batch, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=mini_batch, shuffle=False)

    # these initializations are for MLP network
    n_inp = len(training[0][0])
    n_hid1 = n_hidden1
    n_hid2 = n_hidden2
    n_out = n_output

    # call your neural network model right here

    if (network == 'CNN'):
        net = Simple1DCNN().double().to(device)
    else:
        net = Net(n_inp, n_hid1, n_hid2, n_out, dropout_rate, weight_initializer, dropout_decision).to(device)

    print(net) # prints the architecture of your current NN model

    ##################################################################################################################
    #############      3. Training the model      #######################

    net = net.train()  # set the network to training mode
    # net.apply(weight_init)
    criterion = torch.nn.MSELoss()  # set the loss criterion
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=w_lambda)
    print("Starting training \n")
    # print("h{}_lr{}_lam{}_bat_{}".format(n_hidden1, learning_rate, w_lambda, mini_batch))
    print("####################################################################### \n")

    weight_ih = []  # storing the weights from input to hidden
    weight_ho = []  # storing the weights from hidden unit to output unit
    train_losses = []  # storing the training losses
    val_losses = []  # storing the validation losses
    test_losses = []  # storing the validation losses
    min_val_RMSE = 1e5  # initializing to find min validation RMSE
    min_R_epoch = 1e5  # initializing to find the epoch with min validation RMSE
    counter = []  # to store the different epochs that gives validation accuracy > 90%
    t_correct = []
    t_acc = []
    v_acc = []

    # uncomment below to test on different learning rates
    ###### important: comment out the criterion, optimizer, and net initialized above #####
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # lr_train_loss = []
    # lr_val_loss = []
    # for iter_learn, l_rate in enumerate(learning_rates):
    #     torch.manual_seed(1); np.random.seed(1)
    #     # net = net.train()
    #     net.apply(weight_init)
    #     optimizer = torch.optim.SGD(net.parameters(), lr=l_rate, momentum=0.5)
    #     weight_ho = []

    t0 = time.time()
    for ep in range(epoch):
        train_loss = []  # saves the batch training loss for each epoch
        t_item = 0
        t_correct = []
        for batch_idx, (data, target) in enumerate(train_loader):
            t_item += len(target)
            data, target = data.to(device), target.to(device)  # passing the variables to gpu
            if (network == 'CNN'):
                X = data.double().unsqueeze(1)  # pytorch input should be float -> awkward right?
                Y = target.double().view(-1, 1)  # converting into 2-d tensor
            else:
                X = data.float()  # pytorch input should be float -> awkward right?
                Y = target.float().view(-1, 1)  # converting into 2-d tensor
            optimizer.zero_grad()  # making the gradient zero before optimizing
            oupt = net(X)  # neural network output
            loss_obj = criterion(oupt, Y)  # loss calculation
            loss_obj.backward()  # back propagation
            optimizer.step()  # remember w(t) = w(t-1) - alpha*cost ??
            # if not, we are just updating weights here

            # saving the weights from input to hidden and hidden to output
            # weight_ih.append(np.reshape(net.hid1.weight.data.clone().cpu().numpy(), (1, n_inp * n_hid1)))
            # weight_ho.append(np.reshape(net.oupt.weight.data.clone().cpu().numpy(), (1, n_hid2 * n_out)))

            train_loss.append(loss_obj.item())  # batch losses -> length = number of batches in an epoch
            correct = torch.sum((torch.abs(oupt - Y) < torch.abs(0.1 * Y)))
            t_correct.append(correct)

        if (network == 'CNN'):
            weight_ho.append((net.fc3.weight.data.clone().cpu().numpy()))
        else:
            weight_ho.append(np.reshape(net.oupt.weight.data.clone().cpu().numpy(), (1, n_hid2 * n_out)))

        t_result = ((sum(t_correct)).item() / t_item)
        t_acc.append(t_result)

        # getting the training loss for each epoch
        train_loss_avg = sum(train_loss) / len(train_loss)  # batch averaging
        train_losses.append([train_loss_avg])  # saving average batch loss for each epoch

        # testing validation set after training all the batches
        net = net.eval()  # set the network to evaluation mode
        val_acc, val_RMSE, vali_loss = accuracy(net, validation_loader, tolerance, criterion, device, network, eval=False)
        val_losses.append([vali_loss.item()])  # validation loss on entire samples for each epoch
        v_acc.append(val_acc.item() / 100)

        # find the epoch that gives minimum validation loss
        if val_RMSE < min_val_RMSE:
            min_val_RMSE = val_RMSE
            min_R_epoch = ep

        # set the network to training mode after validation
        net = net.train()

        # if we are willing to test the models on testing data that gives validation accuracy > 90%
        # if (save_model) and val_RMSE <= 0.65:
        if (save_model) and val_RMSE<=0.25:
            counter.append((ep))
            torch.save(net.state_dict(), output_path + '/models/model{}.pth'.format(ep))

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

    print("####################################################################### \n")
    print("Training complete \n")
    print("Time taken = {}".format(time.time() - t0))
    print(" min RMSE = {} at {} epoch \n".format(min_val_RMSE, min_R_epoch))
    print("####################################################################### \n")

    # results[cnt, 0] = min_val_RMSE
    # results[cnt, 1] = min_R_epoch
    # results[cnt, 2] = n_hidden1
    # results[cnt, 3] = learning_rate
    # results[cnt, 4] = w_lambda
    # results[cnt, 5] = mini_batch
    # cnt += 1

    if (save_figs):
        np.savetxt(output_path + '/train_losses.csv', train_losses, delimiter=',')
        np.savetxt(output_path + '/val_losses.csv', val_losses, delimiter=',')

    ###################################################################################################################
    ###################      4. Evaluating the model (validation) #######################

    net = net.eval()  # set eval mode
    acc_val, val_RMSE, _ = accuracy(net, validation_loader, tolerance, criterion, device, network, eval=True)
    print('validation accuracy with {} tolerance = {:.2f} and RMSE = {:.6f}\n'
          .format(tolerance, acc_val, val_RMSE))

    ###################################################################################################################
    ###################      5. Using the model (testing)     #######################

    test_loss, test_RMSE = testing(model_path, data_path, counter, net, criterion, device, load_model)

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
    # plt.savefig(output_path + '/losses{}'.format(iter_learn), dpi=600, bbox_inches='tight')
    # pdb.set_trace()

    label_graph = ['train_loss', 'val_loss', 'fitted_train_loss', 'fitted_val_loss', 'test_loss']
    losses = np.squeeze(train_losses)
    val_losses = np.squeeze(val_losses)

    t_x = np.arange(len(losses))
    v_x = np.arange(len(val_losses))

    plt.figure()
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
    if (save_figs):
        plt.savefig(output_path + '/batch_loss.png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()

    plt.figure()
    plt.plot(t_x, t_acc, label='training accuracy', c='blue', linewidth='5')
    plt.plot(v_x, v_acc, label='validation accuracy', c='green', linewidth='2')
    # uncomment below if you want to have a vertical line at your best epoch
    # plt.axvline(x=min_R_epoch, color='r', linestyle='--', linewidth=3)
    plt.ylabel("Accuracy", **axis_font)
    plt.xlabel("Number of epochs", **axis_font)
    plt.xlim(0, len(t_x))
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.legend()
    if (save_figs):
        plt.savefig(output_path + '/accuracy.png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()

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
    # plt.savefig(output_path + '/wih.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close()

    plt.figure()
    weight_ho = np.reshape(weight_ho, (np.shape(weight_ho)[0], np.shape(weight_ho)[2]))
    weights_ho_num = int(np.shape(weight_ho)[1])
    for i in range(0, weights_ho_num):
        plt.plot(weight_ho[:, i])
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("weights from hidden to output layer", **axis_font)
    plt.xlabel("Number of epochs", **axis_font)
    plt.xlim(0, epoch)
    plt.rcParams['agg.path.chunksize'] = 10000
    if (save_figs):
        plt.savefig(output_path + '/who', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()

    # # finally saving the results
    # np.savetxt('../../Neural-Network-Regression/log/testing_models/results.csv', results, delimiter=',')
    # best_idx = np.argmax(results[:, 0])
    # print ("the best result is given with hid: {} "
    #        "lr: {} lambda: {} batch_size: {} with an min RMSE of: {}"
    #        " at epoch: {}".format(results[best_idx,2], results[best_idx,3],
    #                               results[best_idx,4], results[best_idx, 5],
    #                               results[best_idx,0], results[best_idx,1]))