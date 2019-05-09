import torch
import os
import time
from torch.autograd import Function
from torch import nn
from extensions.AE._ext import my_lib

# using C build extension to calculate the loss

class AElossFunction(Function):
    # two direction in NN, and to use C build extension loss function
    def forward(self, tags, keypoints):
        # declare arrays for loss calculation
        output = torch.zeros(torch.Size((tags.size()[0], 2)))
        mean_tags = torch.zeros(torch.Size((tags.size()[0], keypoints.size()[1], tags.size()[2]+1)))
        self.mean_tags = mean_tags
        # the parameter declared in my_lib.c, parameter here is for calculation in c
        # loss_forward(THCudaTensor *Tag,THLongTensor *keypoints,THFloatTensor *output,THFloatTensor *mean_tags)
        my_lib.my_lib_loss_forward(tags, keypoints, output, mean_tags)
        self.save_for_backward(tags, keypoints)
        return output

    def backward(self, grad_output):
        # this is get the data from 'self.save_for_backward(tags, keypoints)'
        tags, keypoints = self.saved_tensors
        # declare array for loss calculation
        grad_input = torch.zeros(tags.size()).cuda(tags.get_device())
        #grad_input = tags.new(tags.size()).zero_()
        # the parameter declared in my_lib.c
        # loss_backward( *Tag, *keypoints, *mean_tags, *grad_output, *grad_input)
        my_lib.my_lib_loss_backward(tags, keypoints, self.mean_tags, grad_output, grad_input)
        self.mean_tags = None
        return grad_input, torch.zeros(keypoints.size())

class AEloss(nn.Module):

    def forward(self, input, input1):
        if not input.is_cuda:
            input = input.cuda()
        output = AElossFunction()(input, input1)
        return output
