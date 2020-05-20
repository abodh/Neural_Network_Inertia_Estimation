import torch
import os
import h5py
import numpy as np
from model import Net
import pdb

def accuracy(model, validation_loader, pct_close, criterion, device, eval = False):
    with torch.no_grad():
        val_correct = []
        val_loss_func = []
        n_items = 0
        for idx, (test_f_rf,test_M_D) in enumerate(validation_loader):
            n_items += len(test_M_D)
            test_f_rf, test_M_D = test_f_rf.to(device), test_M_D.to(device)
            X = test_f_rf.float()
            Y = test_M_D.float().view(-1,1)    # reshaping to 2-d Tensor
            oupt = model(X)                    # all predicted as 2-d Tensor
            loss = criterion(oupt,Y)
            n_correct = torch.sum((torch.abs(oupt - Y) < torch.abs(pct_close * Y)))
            val_correct.append(n_correct)
            val_loss_func.append(loss)

        loss_func = sum(val_loss_func)/ len(val_loss_func)
        result = (sum(val_correct) * 100.0 / n_items)
        RMSE_loss = torch.sqrt(loss_func)

        # observing the result when set to eval mode
        if (eval):
            print('Predicted test output for random batch = {}, actual output = {} with accuracy of {:.2f}% '
                  'and RMSE = {:.6f}'.format(oupt, Y, result, RMSE_loss))
        return result, RMSE_loss, loss_func

def testing(model_path,
            data_path,
            counter,
            net,
            criterion,
            device,
            load_model = True,
            trained_test = False):

    net = net.eval()
    with torch.no_grad():
        eval_file = h5py.File(data_path + 'eval_data.mat', 'r')
        eval_var = eval_file.get('eval_data')
        X_eval = torch.Tensor(np.array(eval_var[0:-2, :]).T)            # a 2-d tensor
        Y_eval = torch.Tensor(np.array(eval_var[-1, :]).T).view(-1, 1)  # converting to a 2-d tensor

        X_eval, Y_eval = X_eval.to(device), Y_eval.to(device)
        if (load_model):
            test_loss = []  # accumulating the losses on different models
            i = 0
            epis = []
            for filename in os.listdir(model_path):
                if filename.endswith(".pth"):
                    net.load_state_dict(torch.load(model_path + '/' + filename))
                    net.eval()
                    # pdb.set_trace()
                    y = net(X_eval)
                    loss_test = torch.sqrt(criterion(y, Y_eval))
                    test_loss.append(loss_test.item())
                    if not trained_test:
                        print ('test result at epoch {} is y = {}'.format(counter[i], y.view(len(y))))
                    else:
                        if len(filename) == 12:
                            epo = int(filename[5:8])
                            epis.append(epo)
                        else:
                            epo = int(filename[5:7])
                            epis.append(epo)
                        print('Actual output = {}'.format(Y_eval))
                        print('test result at epoch {} is y = {}'.format(epo, y))
                    i =  i + 1
                    continue
                else:
                    continue
            test_RMSE = min(test_loss)
            if not trained_test:
                min_val_epoch = counter[np.argmin(test_loss)]
                print('the best model is at epoch {} which gives a loss of {:.6f} \n'
                  .format(min_val_epoch, test_RMSE))
            else:
                print('the best model is at epoch {} gives a loss of {:.6f} \n'
                      .format(epis[np.argmin(test_loss)], test_RMSE))
        else:
            y = net(X_eval)
            test_loss = criterion(y, Y_eval)
            test_RMSE = torch.sqrt(test_loss)
            print ('Actual output = {} and Predicted test output = {} with RMSE = {:.6f}'.format(Y_eval, y, test_RMSE))
    return test_loss, test_RMSE

if __name__ == '__main__':
    net = Net(n_inp = 402, n_hid1 = 25, n_hid2 = 25, n_out = 1,
              dropout_rate = 0.5, weight_ini = 0.05, dropout_decision = False)
    testing(model_path = "../../Neural-Network-Regression/log/testing_models/"
                         "Apr-23-2020-10.20.43-h25_lr0.001_lam0.0005_bat_30/models"

    ,data_path =  "..\\..\\matlab files\\area2_non_IID\\manipulated\\"
    ,counter = 0
    ,net = net
    ,criterion = torch.nn.MSELoss()
    ,device = torch.device("cpu")
    ,load_model = True
    ,trained_test = True        # setting this true means we are using a pre-trained model to test
    )
