"""
Notes to self for http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

Looking at the network diagram, we take an input that is 32x32x3 and we convolve, downsample
and then spit it out into some fully connected layers.

Let's unpack.

To go from input >> C1, we see a convolution. The input is 32x32 and we need our convolution
to give us a 6-filter output with dimensions 28x28. To find the kernel size that gives us
such an output, we simply use the formula I stole from CS231n: 
output = (input - kernel / stride) + 1. We can rearrange to solve for kernel.. 

We get kernel = (output - input / stride) + 1

In this instance we can see that: k = (4/1) + 1 = 5. So we need a 5x5 kernel for our first 
convolution.

Repeating this process for the second convolution (which we see mapping S2 >> C3), gives us
the same result.

We define these in init because that's where we lay out the learnable layers. We'll define
the functions that relate those layers in the forward method, which must be implemented.

The rest of our init is comprised of linear layers. Params are input size, output size.

"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # gives us c1 where we go from 32x32 >> 28x28
        self.conv2 = nn.Conv2d(6, 16, 5) # gives us c3 14x14 >> 10x10
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # this takes max pool output, which will be flattened, after C3
        self.fc2 = nn.Linear(120, 84) # c5
        self.fc3 = nn.Linear(84, 10) # f6

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) # input to s2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # equivalent to (2,2). s2 to s4
        x = x.view(-1, self.num_flat_features(x)) 
        # This ^ reshapes the tensor. We don't know how many rows we want but we know 
        # how many columns we have. The result is a [<whatever python feels is right>, num_flat_features(x)] tensor. 
        # We're flattening a tensor with a depth of 16 so we can feed it to the FC layer.
        x = F.relu(self.fc1(x)) # pool > 120
        x = F.relu(self.fc2(x)) # 120 > 84
        x = self.fc3(x) #84 > 10
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:] # ignore the batch dimensions, grab everything else
        num_features = 1
        for s in size:
            num_features *= s 
        return num_features

net = Net()
print(net)

"""
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""

params = list(net.parameters())
print(len(params))
print(params[0].size())

"""
10
torch.Size([6, 1, 5, 5])
"""

# Note: This is for 32x32 images, so everything else needs to be resized (i.e. MNIST)
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

"""
Variable containing:
 0.0208 -0.0077 -0.0682 -0.0419 -0.0150 -0.1079 -0.0107  0.0379  0.0167 -0.0348
[torch.FloatTensor of size 1x10]
"""

# The Variable allows us to auto-grad for backprop! Gotta zero the network gradients 
# before we use backward. PyTorch gradients are cumulative to facilitate some architectures
# (i.e. GANs). For non-cumulative gradients, we reset the gradient ourselves. 
net.zero_grad()
out.backward(torch.randn(1,10)) # TODO why rando? initialisation?

# Loss
output = net(input)
target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()
loss = criterion(output, target)

print(loss)

"""
Variable containing:
 38.6910
[torch.FloatTensor of size 1]
"""

"""
From the tutorial:
If we were to follow loss backward using .grad_fn, computational graph looks like this:

input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss

When we call loss.backward(), the graph is differentiated wrt the loss, and all Variable's 
.grad will be accumulated with the gradient.

Backward steps:
"""
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

"""
<MseLossBackward object at 0x7fc8bc7c6160>
<AddmmBackward object at 0x7fc8bc7c6240>
<ExpandBackward object at 0x7fc8bc7c6240>
"""

# Remember to zero the grad
net.zero_grad()

print('Conv1.bias.grad before backprop')
print(net.conv1.bias.grad)

loss.backward()

print('After')
print(net.conv1.bias.grad)

"""
Conv1.bias.grad before backprop
Variable containing:
 0
 0
 0
 0
 0
 0
[torch.FloatTensor of size 6]

After
Variable containing:
1.00000e-02 *
  2.1366
 -1.0169
  4.8340
 -2.1016
  0.6366
 -2.8716
[torch.FloatTensor of size 6]

"""

# Weights
# Update them using SGD

# learning_rate = 0.01 # hyper param
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# create this bad boy
optimizer = optim.SGD(net.parameters(), lr=0.01)
# TRAINING loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward
optimizer.step() # does the update