import sys
import time
import os
import tqdm
from os.path import dirname

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

import pickle
import torch
import importlib
import argparse

def parse_command_line():
    # function to setup how parse commend line
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='pose', help='task to be trained')
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--mode', type=str, default='single', help='scale mode')
    args = parser.parse_args()
    return args

def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    # the config["opt"] added in init(), it is parsed command line
    opt = config['opt']

    if opt.continue_exp:
        # checkpoint file is saved as pth.tar extension, these two line is to load this file
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pth.tar')
        if os.path.isfile(resume_file):
            # load the trained checkpoint file
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)
            # write info from checkpoint to the config
            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    from pytorch/examples
    """
    # this func called in save()
    # will save checkpoint after each epoch done
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save(config):
    # will save checkpoint each epoch, and this func is used in train() function
    resume = os.path.join('exp', config['opt'].exp)
    resume_file = os.path.join(resume, 'checkpoint.pth.tar')

    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print('=> save checkpoint')

def train(train_func, data_func, config, post_epoch=None):
    # training basic pipeline, train and valid section in each epoch
    while True:
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break
        # after train will be a valid section
        for phase in ['train', 'valid']:
            num_step = config['train']['{}_iters'.format(phase)]
            generator = data_func(phase)
            print('start', phase, config['opt'].exp)

            show_range = range(num_step)
            # tqdm is toolbar shows percentage of the progress
            show_range = tqdm.tqdm(show_range, total = num_step, ascii=True)
            batch_id = num_step * config['train']['epoch']
            for i in show_range:
                datas = next(generator)
                outs = train_func(batch_id + i, config, phase, **datas)
        config['train']['epoch'] += 1
        save(config)

def init():
    """
    import configurations specified by opt.task

    task.__config__ contains the variables that control the training and testing
    make_network builds a function which can do forward and backward propagation

    please check task/base.py
    """
    # get parsed command line and then make a directory for the train with user's customized experiment name
    # ex: python train.py -e trainRun, then will create a trainRun directory for training log and checkpoint
    opt = parse_command_line()
    task = importlib.import_module('task.' + opt.task)
    exp_path = os.path.join('exp', opt.exp)
    config = task.__config__
    try: os.makedirs(exp_path)
    except FileExistsError: pass
    # add parsed command line to the config and the data provider(here is coco dataset)
    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])
    # the config is imported from task directory that contain learning rate, batch size,etc
    # func contains loss calculation method and train details(training or evaluation,
    # decreasing learning rate after 200000 iterations)
    func = task.make_network(config)
    reload(config)
    return func, config

def main():
    # get initial data from this py file
    func, config = init()
    # this is call init()function from coco_pose directory dp.py to initialize the config by using pytorch.utils
    data_func = config['data_provider'].init(config)
    train(func, data_func, config)

if __name__ == '__main__':
    main()
