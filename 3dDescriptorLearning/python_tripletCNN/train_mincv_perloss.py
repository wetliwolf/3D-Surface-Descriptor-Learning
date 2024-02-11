

# coding: utf-8

# In[1]:


import argparse
import os
import sys
import time
from os.path import join

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


# In[2]:


parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpuid', '-g', default='0', type=str, metavar='N',
                    help='GPU id to run') 

parser.add_argument('--learning_rate', '--lr', default=0.0003, type=float, 
                    help='the learning rate')
parser.add_argument('--l2_regularizer_scale', default=0.005, type=float,
                    help='scale parameter used in the l2 regularization')
parser.add_argument('--n_iteration', '-n', default=100000, type=int,metavar='N',
                    help='number of training iterations')

parser.add_argument('--batch_size', '--bs', default=128, type=int, 
                    help='size of training batch, it is equal to batch_keypoint_num*batch_gi_num')
parser.add_argument('--batch_keypoint_num', '--bkn', default=16, type=int, 
                    help='number of different keypoints in a training batch')
parser.add_argument('--batch_gi_num', '--bgn', default=8, type=int,
                    help='number of geometry images of one keypoint in a training batch')

parser.add_argument('--val_freq', '--vf', default=5, type=int,
                    help='frequency of validation.')
parser.add_argument('--print_freq', '--pf', default=100, type=int,
                    help=r'print info every {print_freq} iterations')
parser.add_argument('--save_freq', '--sf', default=500, type=int,
                    help=r'save the current trained model every {save_freq} iterations')

parser.add_argument('--summary_saving_dir', '--ssd', default='./summary', type=str, 
                    help='directory to save summaries')
parser.add_argument('--model_saving_dir', '--msd', default='./saved_models', type=str,
                    help='directory to save trained models')

parser.add_argument('--restore', '-r', dest='restore',default=True, action='store_true',
                    help='bool value, restore variables from saved model of not')
parser.add_argument('--restore_path',default='/data/yqwang/Project/3dDescriptor/train_softmax_adam/saved_models/training_model_multiuse-9999', 
                    type=str, 
                    help='path to the saved model(if restore)')

parser.add_argument('--use_kpi_set', dest='use_kpi_set', default=True, action='store_true', 
                    help='bool value, use keypoint set from keypoint file or not')
parser.add_argument('--keypoints_path', default='/data/yqwang/Dataset/faust_256p/keypoints_faust256.kpi', 
                    type=str, 
                    help='path to the keypoint file(if use_kpi_set)')
parser.add_argument('--n_all_points', default=6890, type=int, 
                    help='number of all points in the model')


parser.add_argument('--shuffle_batch_capacity', default=400, type=int,
                    help='capacity of shuffle bacth buffer')
parser.add_argument('--gi_size', default=32, type=int,
                    help='length and width of geometry image, assuming it\'s square')
parser.add_argument('--gi_channel', default=31, type=int,
                    help='number of geometry image channels')
parser.add_argument('--triplet_loss_gap', default=1, type=float,
                    help='the gap value used in the triplet loss')
parser.add_argument('--n_loss_compute_iter', default=17, type=int,
                    help='number of iterations to compute the training loss')