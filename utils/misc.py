import torch
import numpy as np
import importlib

# Helpers when setting up training

def importNet(net):
    # this is actually function as string value in way of splitting and regrouping
    # to make same string values are separated
    # this function be used at pose.py make_network() and make_network() as init() in train.py
    # also has code for decreasing the learning rate after a iteration number
    t = net.split('.')
    path, name = '.'.join(t[:-1]), t[-1]
    module = importlib.import_module(path)
    return eval('module.{}'.format(name))

def make_input(t, requires_grad=False, need_cuda = True):
    # used in pose.py
    # calculate the gradient and convert the data type to the cuda.LongTensor
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    if need_cuda:
        inp = inp.cuda()
    return inp

def make_output(x):
    # make sure data is matrix and cpu data type
    if not (type(x) is list):
        return x.cpu().data.numpy()
    else:
        return [make_output(i) for i in x]

# Image processing functions

def inv_mat(mat):
    # compute Moore-Penrose inverse matrix
    ans = np.linalg.pinv(np.array(mat).tolist() + [[0,0,1]])
    return ans[:2]

def get_transform(center, scale, res, rot=0):
    # used in dp.py & test.py
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        # defining rotation matrix
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def kpt_affine(kpt, mat):
    # used in dp.py
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    # concatenate along Y-axis
    return np.dot(np.concatenate((kpt, kpt[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

def resize(im, res):
    # used in test.py
    import cv2
    return np.array([cv2.resize(im[i],res) for i in range(im.shape[0])])
