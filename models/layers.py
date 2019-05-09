from torch import nn
from torch.autograd import Function
import sys 
import time
import numpy as np
import cv2
import torch

# three layers defined in this py file, and all three layer only forward propagate
# the setup parameter 'inp_dim,out_dim' are come from config in task/pose.py

Pool = nn.MaxPool2d

def batchnorm(x):
    # normalize the variable,
    # be putted into layers has improvement in accuracy
    return nn.BatchNorm2d(x.size()[1])(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, (1./n)**0.5) 


# three layers Convolution, Hourglass, Fully-connected layer

class Full(nn.Module):
    # pass the function via parameter
    # set up Fully connected layer
    # inp_dim,out_dim from config in task/pose.py
    # bn is batchnorm, relu is activation function that keep input in (0,x)
    def __init__(self, inp_dim, out_dim, bn = False, relu = False):
        super(Full, self).__init__()
        # the bias is additional parameter, like hyperparameter
        # f = ax + b, b here is the bias, linear function
        self.fc = nn.Linear(inp_dim, out_dim, bias = True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        # make x through relu and batchnorm and then forward
        x = self.fc(x.view(x.size()[0], -1))
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class Conv(nn.Module):
    # initialize Convolution layer for input, output dimension
    # stride is number of pixel skipped in each X-axis and Y-axis
    # relu is to prevent backward gradient vanish
    # kernel size is filter in Convolution layer, here is 3x3 filter
    # stride is the pixel jumped in each X-axis and Y-axis round
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # make convolution layer x through relu and batchnorm and then forward
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class Hourglass(nn.Module):
    # n,f are inp_dim, oup_dim respectively defined in config
    def __init__(self, n, f, bn=None, increase=128):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Conv(f, f, 3, bn=bn)
        # Lower branch
        # pooling layer 2x2
        self.pool1 = Pool(2, 2)
        self.low1 = Conv(f, nf, 3, bn=bn)
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Conv(nf, nf, 3, bn=bn)
        self.low3 = Conv(nf, f, 3)
        self.up2  = nn.UpsamplingNearest2d(scale_factor=2)
        # this is to upsampling input channel composed with multiple input channel
        # scale_factor=2 is multiplier for the output image height / weight
    def forward(self, x):
        # forward method
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2
