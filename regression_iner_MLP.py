import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import torch as T
from loading_data import loading, normalize_data
import pdb

# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
axis_font = {'fontname': 'Arial', 'size': '16'}


def accuracy(model, data_x, data_y, pct_close):
  n_items = len(data_y)
  X = T.Tensor(data_x)  # 2-d Tensor
  Y = T.Tensor(data_y)  # actual as 1-d Tensor
  Y = Y.view(n_items)
  pdb.set_trace()
  oupt = model(X)       # all predicted as 2-d Tensor
  pdb.set_trace()
  pred = oupt.view(n_items)  # all predicted as 1-d
  n_correct = T.sum((T.abs(pred - Y) < T.abs(pct_close * Y)))
  result = (n_correct.item() * 100.0 / n_items)  # scalar
  return result

def accuracy_2(model, data_x, data_y, pct_close):
  n_items = len(data_y)
  X = T.Tensor(data_x)  # 2-d Tensor
  Y = T.Tensor(data_y)  # actual as 1-d Tensor
  Y = Y.view(n_items)
  oupt = model(X)       # all predicted as 2-d Tensor
  pred = oupt.view(n_items)  # all predicted as 1-d
  n_correct = T.sum((T.abs(pred - Y) < T.abs(pct_close * Y)))
  pdb.set_trace()
  result = (n_correct.item() * 100.0 / n_items)  # scalar
  return result

# MLP based model
class Net(T.nn.Module):
  def __init__(self, n_inp, n_hid1, n_out):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(n_inp, n_hid1)  # 3-5-1
    self.oupt = T.nn.Linear(n_hid1, n_out)
    T.nn.init.xavier_uniform_(self.hid1.weight, gain = 0.01)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight, gain = 0.01)
    T.nn.init.zeros_(self.oupt.bias)
  def forward(self, x):
    z = T.tanh(self.hid1(x))
    z = self.oupt(z)  # no activation, aka Identity()
    return z

T.manual_seed(1);  np.random.seed(1)

#1. Loading the data from the text(mat) file and normalizing the data using l2 norm

filename = 'file.mat'
total_data = loading(filename)  # load the mat_file including input and output data
# pdb.set_trace()
# train_x, train_y, test_x, test_y = normalize_data (total_data)
train_x, train_y = normalize_data (total_data)

# pdb.set_trace()

#2. creating the model
n_inp = 20
n_hid1 = 10
# n_hid2 = 10
n_out = 1
net = Net(n_inp, n_hid1, n_out)

#3. Training the model
net = net.train()
bat_size = 1
loss_func = T.nn.MSELoss()
optimizer = T.optim.SGD(net.parameters(), lr=0.01)
n_items = len(train_x)
batches_per_epoch = n_items // bat_size
max_batches = 10 * batches_per_epoch
print("Starting training")
# pdb.set_trace()
output_full = T.tensor([])
# weights = T.tensor([])
weights = []
output_avg = []
losses = []
# pdb.set_trace()
choice_a = 0
choice_b = 1
for b in range(max_batches):
    # curr_bat = np.random.choice(n_items, bat_size, replace=False)
    # pdb.set_trace()
    curr_bat = np.arange(choice_a , (10 * choice_b))
    choice_a = choice_a + 10
    choice_b = choice_b + 1
    if (np.where(curr_bat >= len(train_x))):
        choice_a = 0
        choice_b = 1
        curr_bat = np.arange(choice_a, (10 * choice_b))
        choice_a = choice_a + 10
        choice_b = choice_b + 1
    current_train = train_x[curr_bat]
    X = T.Tensor(np.transpose(np.stack((current_train[:,0],current_train[:,1])).reshape((-1,1))))
    # Y = T.Tensor(train_y[curr_bat]).view(bat_size,1)
    Y = T.Tensor(train_y[curr_bat])
    # pdb.set_trace()
    optimizer.zero_grad()
    # pdb.set_trace()
    oupt = net(X)
    oupt_numpy = oupt.data.cpu().numpy()
    output_full = T.cat((output_full, oupt), 0)
    output_avg.append(np.mean(oupt_numpy))
    loss_obj = loss_func(oupt, Y)
    loss_obj.backward()
    optimizer.step()
    weights.append(np.reshape(net.hid1.weight.data.clone().cpu().numpy(), (1, n_inp * n_hid1)))
    # pdb.set_trace()
    if b % (max_batches // 398) == 0:
        # print(output.size(), end="")
        print("batch = %6d" % b, end="")
        # print("  batch loss = %7.4f" % loss_obj.item(), end="")
        print("  batch loss = %7.4f" % loss_obj.item())
        # net = net.eval()
        # # pdb.set_trace()
        # acc = accuracy(net, train_x, train_y, 0.10)
        # net = net.train()
        # print("  accuracy = %0.2f%%" % acc)
    losses.append(loss_obj.item())
print("Training complete \n")
# pdb.set_trace()
weights = np.reshape(weights, (np.shape(weights)[0],np.shape(weights)[2]))
weights_num = int(np.shape(weights)[1])
# label_graph = ['Out1', 'Out2', 'Out3', 'Out4']
for i in range(0, weights_num):
    plt.plot(weights[:,i])
plt.grid(linestyle='-', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("weights from input to hidden layer", **axis_font)
plt.xlabel("Number of batches in entire epoch", **axis_font)
plt.xlim(0, 500)
plt.savefig('iner_caseB_weight.png', dpi = 600, bbox_inches='tight')
plt.show()


# pdb.set_trace()

# for param in net.parameters():
#   print(param.data)
# pdb.set_trace()

# 4. Evaluate model
# net = net.eval()  # set eval mode
# acc = accuracy_2(net, test_x, test_y, 0.2)
# print("Accuracy on test data = %0.2f%%" % acc)

# print (output_full.shape)
output_full_array = output_full.data.cpu().numpy()
# print (np.shape(output_avg))
# np.savetxt('inertiaout.csv', arr)

# print (weight.shape)
# print(losses)
# arr1 = weight.data.cpu().numpy()
# np.savetxt('weights.csv', arr1)

# weights_array = weights.data.cpu().numpy()
# pdb.set_trace()

'''
    # 5. Use model
    eval_data = loading(inertia = 5,                       # value of inertia to be tested
                      low = 10,                            # lower limit of data to be captured
                      high = 12.5,                         # upper limit of the data to be captured
                      f_file = 'frequency_5.mat',          # frequency data mat file
                      rocof_file = 'rocofrequency_5.mat')  # rocof data mat file
    
    # number of data to evaluate
    eval_num = int(len(eval_data))
    # pdb.set_trace()
    # data normalization and separating training and testing data
    normalized_eval_data = normalize(eval_data[:,0:2], axis = 0, norm = 'l2')
    eval_x = normalized_eval_data [0:eval_num,:]
    X = T.Tensor(eval_x)
    y = net(X)
    pred_array = y.view(eval_num).data.cpu().numpy()
    
    ############ predicted value #############
    fig, axx = plt.subplots()
    axx.plot(pred_array)
    axx.set_xlim(0, 1250) # apply the x-limits
    # axx.set_ylim(0, 12) # apply the y-limits
    plt.ylabel("Predicted Inertia", **axis_font)
    plt.xlabel("ID of evaluating samples", **axis_font)
    plt.title("Inertia vs frequency", **title_font)
    # plt.hlines(5, 0, 1250, colors='r', linestyles='dashed', linewidth=3)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('prediction_5.png', dpi = 600)
    plt.show()

'''


# ############  full output plot  #############
#
# fig, ax = plt.subplots()
# ax.plot(output_full_array)
# # ax.plot(output)
# ax.set_xlim(0, 54000) # apply the x-limits
# # ax.set_ylim(0, 10) # apply the y-limits
# # plt.hlines(10, 0, 6000, colors='r', linestyles='dashed', linewidth=3)
# plt.grid(linestyle='-', linewidth=0.5)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.ylabel("Estimated Inertia", **axis_font)
# plt.xlabel("Number of batches in entire epoch", **axis_font)
# plt.title("Trained output in entire batch", **title_font)
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
# # plt.savefig('output_full_array.png', dpi = 600)
# plt.show()

'''
    ############  averaged output plot  #############
    fig, ax = plt.subplots()
    ax.plot(output_avg)
    # ax.set_xlim(0, 600) # apply the x-limits
    # ax.set_ylim(0, 12) # apply the y-limits
    # plt.hlines(10, 0, 600, colors='r', linestyles='dashed', linewidth=3)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Estimated Inertia", **axis_font)
    plt.xlabel("Number of batches", **axis_font)
    plt.title("estimated system inertia with increase batch number", **title_font)
    # plt.show()
    axins = zoomed_inset_axes(ax, 2.5, loc=4) # zoom-factor: 2.5, location: upper-left
    axins.plot(output_avg)
    x1, x2, y1, y2 = 0, 2500, 4, 6 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    # axins.xaxis.set_visible('False')
    # axins.yaxis.set_visible('False')
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    plt.grid()
    # plt.savefig('output_avg.png', dpi = 600)
    plt.show()
'''

############ loss #############
fig, axx = plt.subplots()
axx.plot(losses)
axx.set_xlim(0, 500) # apply the x-limits
# axx.set_ylim(0, 100) # apply the y-limits
plt.ylabel("Mean Squared Error", **axis_font)
plt.xlabel("Number of batches", **axis_font)
plt.title("Batch training loss vs number of batch", **title_font)
plt.grid(linestyle='-', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('batch_loss_caseB.png', dpi = 600, bbox_inches='tight')
plt.show()



