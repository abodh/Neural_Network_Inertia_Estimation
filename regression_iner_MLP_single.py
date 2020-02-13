# author: Abodh Poudyal
# referenced from : James McCaffrey (Microsoft)
# Date: February, 2020

import numpy as np
from sklearn.preprocessing import normalize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import pdb
import torch as T

with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

def accuracy(model, data_x, data_y, pct_close):
  n_items = len(data_y)
  X = T.Tensor(data_x)  # 2-d Tensor
  Y = T.Tensor(data_y)  # actual as 1-d Tensor
  oupt = model(X)       # all predicted as 2-d Tensor
  pred = oupt.view(n_items)  # all predicted as 1-d
  n_correct = T.sum((T.abs(pred - Y) < T.abs(pct_close * Y)))
  result = (n_correct.item() * 100.0 / n_items)  # scalar
  return result

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(2, 5)  # 2-(5)-1
    self.oupt = T.nn.Linear(5, 1)
    T.nn.init.xavier_uniform_(self.hid1.weight)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)
  def forward(self, x):
    z = T.tanh(self.hid1(x))
    z = self.oupt(z)  # no activation, aka Identity()
    return z

T.manual_seed(1);  np.random.seed(1)

#1. Loading the data from the text file and normalizing the data using l2 norm

data_main = np.genfromtxt('datafrequ.csv', delimiter = ",") # define the file name ('datafrequ.csv') is default here

data = data_main[:,1:3]
normalized_input_data = normalize(data, axis = 0, norm = 'l2')
train_x = normalized_input_data [0:600,:]
train_y = data_main[0:600, 3]
test_x = normalized_input_data [600:800,:]
test_y = data_main[600:800, 3]

#2. creating the model
net = Net()

#3. Training the model
net = net.train()
bat_size = 10
loss_func = T.nn.MSELoss()
optimizer = T.optim.SGD(net.parameters(), lr=0.00093)
n_items = len(train_x)
batches_per_epoch = n_items // bat_size
max_batches = 10 * batches_per_epoch
print("Starting training")
output = T.tensor([])
weight = T.tensor([])
losses = []

for b in range(max_batches):
    curr_bat = np.random.choice(n_items, bat_size, replace=False)
    X = T.Tensor(train_x[curr_bat])
    Y = T.Tensor(train_y[curr_bat]).view(bat_size,1)
    # pdb.set_trace()
    optimizer.zero_grad()
    oupt = net(X)
    # pdb.set_trace()
    output = T.cat((output, oupt), 0)
    loss_obj = loss_func(oupt, Y)
    loss_obj.backward()
    optimizer.step()
    weight =T.cat((weight, net.hid1.weight.data), 0)
    # pdb.set_trace()
    if b % (max_batches // 60) == 0:
      print("batch = %6d" % b, end="")
      print("  batch loss = %7.4f" % loss_obj.item(), end="")
      net = net.eval()
      acc = accuracy(net, train_x, train_y, 0.00835)
      net = net.train()
      print("  accuracy = %0.2f%%" % acc)
    losses.append(loss_obj.item())
print("Training complete \n")

# 4. Evaluate model
net = net.eval()  # set eval mode
acc = accuracy(net, test_x, test_y, 0.99)
print("Accuracy on test data = %0.2f%%" % acc)

# 5. Use model
# raw_inpt = np.array([[50,60]], dtype=np.float32)
# norm_inpt = np.array([[0.833, 1]], dtype=np.float32)
# X = T.Tensor(norm_inpt)
# y = net(X)
# print("For f and rocof: ")
# for (idx,val) in enumerate(raw_inpt[0]):
#    if idx % 1 == 0: print("")
#    print("%11.6f " % val, end="")
# print("\n\nPredicted Inertia = %0.2fs" %
#       (y.item()))

print (output.shape)
output_array = output.data.cpu().numpy()
# np.savetxt('inertiaout.csv', arr)

print (weight.shape)
# print(losses)
# arr1 = weight.data.cpu().numpy()
# np.savetxt('weights.csv', arr1)


# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
axis_font = {'fontname': 'Arial', 'size': '14'}

fig, ax = plt.subplots()
ax.plot(output_array)
ax.set_xlim(0, 6000) # apply the x-limits
ax.set_ylim(0, 12) # apply the y-limits
plt.hlines(10, 0, 6000, colors='r', linestyles='dashed', linewidth=3)
plt.grid(linestyle='-', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Estimated Inertia", **axis_font)
plt.xlabel("Number of training epochs", **axis_font)
plt.title("estimated system inertia with increase in training epochs", **title_font)
# plt.show()
axins = zoomed_inset_axes(ax, 2.5, loc=4) # zoom-factor: 2.5, location: upper-left
axins.plot(output_array)
x1, x2, y1, y2 = 1000, 1700, 6, 8 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
# axins.xaxis.set_visible('False')
# axins.yaxis.set_visible('False')
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.grid()
plt.savefig('inertia.png', dpi = 600)
plt.show()

fig, axx = plt.subplots()
axx.plot(losses)
axx.set_xlim(0, 600) # apply the x-limits
axx.set_ylim(0, 100) # apply the y-limits
plt.ylabel("Mean Squared Error", **axis_font)
plt.xlabel("Batch Training Number", **axis_font)
plt.title("Batch training loss vs number of batch", **title_font)
plt.grid(linestyle='-', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('batch_loss.png', dpi = 600)
plt.show()

# print ("size = ", T.size(weight))
