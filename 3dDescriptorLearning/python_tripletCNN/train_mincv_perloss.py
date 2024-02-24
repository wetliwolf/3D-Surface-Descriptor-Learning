

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
# parser.add_argument('--n_test_iter', default=100, type=int,
#                     help='number of iterations to compute the test results')
parser.add_argument('--tfr_dir', default='/data/yqwang/Dataset/faust_256p/gi_TFRecords', 
                    type=str, 
                    help='directory of training TFRecords, containing' 
                         '3 subdirectories: \"train\", \"val\", and \"test\"')
parser.add_argument('--tfr_name_template', default=r'pidx_%04d.tfrecords', type=str, 
                    help='name template of TFRecords filenames')

global args


# In[3]:


def read_index_file(path, delimiter=' '):
    """
    Read indices from a text file and return a list of indices.
    :param path: path of the text file.
    :return: a list of indices.
    """

    index_list = []
    with open(path, 'r') as text:

        for line in text:
            ls = line.strip(' {}[]\t')

            if not ls or ls[0] == '#':  # Comment content
                continue
            ll = ls.split(delimiter)

            for id_str in ll:
                idst = id_str.strip()
                if idst == '':
                    continue
                index_list.append(int(idst))

    return index_list


# In[4]:


def append_log(path, string_stream):
    """
    Write string_stream in a log file.
    :param path: path of the log file.
    :param string_stream: string that will be write.
    """

    with open(path, 'a') as log:
        log.write(string_stream)
    return


# In[5]:


class TripletNet:
    def __init__(self, args=None, is_training=True):
        self.args = args
        self.is_training = is_training
        # self.predict_net =None
        self.anchor_net = None  # anchor_net is also the predict_net
        self.positive_net = None
        self.negative_net = None
        self.descriptors = None  # descriptors of anchors
        self.cost = None
        self.cost_same = None
        self.cost_diff = None
        self.all_multiuse_params = None
        self.predictions = None
        self.acc = None

    def inference(self, gi_placeholder, reuse=None):  # reuse=None is equal to reuse=False(i.e. don't reuse)
        with tf.variable_scope('model', reuse=reuse):
            tl.layers.set_name_reuse(reuse)  # reuse!

            network = tl.layers.InputLayer(gi_placeholder, name='input')

            """ conv2 """
            network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv2_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn2_1')

            network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                padding='SAME', name='pool2')

            """ conv3 """
            network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv3_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn3_1')

            network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                padding='SAME', name='pool3')

            """ conv4 """
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.identity,
                             padding='SAME', W_init=args.conv_initializer, name='conv4_1')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn4_1')

            network = MeanPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                 padding='SAME', name='pool4')


            network = FlattenLayer(network, name='flatten')
            network = DenseLayer(network, n_units=512, act=tf.identity, name='fc1_relu')

            network = BatchNormLayer(network, decay=0.9, epsilon=1e-4, act=args.activation,
                                     is_train=self.is_training, name='bn_fc')
            # network = DenseLayer(network, n_units=4096, act=args.activation, name='fc2_relu')
            # network = DenseLayer(network, n_units=10, act=tf.identity, name='fc3_relu')
            network = DenseLayer(network, n_units=128, act=tf.identity, name='128d_embedding')

        return network


    def build_nets(self, anchor_placeholder, positive_placeholder, negative_placeholder, anchor_label_placeholder, keypoint_num):
        self.anchor_net = self.inference(anchor_placeholder, reuse=None)
        self.positive_net = self.inference(positive_placeholder, reuse=True)
        self.negative_net = self.inference(negative_placeholder, reuse=True)

#         gap = tf.constant(np.float32(args.triplet_loss_gap))
# #         zero = tf.constant(np.float32(0))
#         zeros = tf.zeros(args.batch_size)


        self.all_multiuse_params = self.anchor_net.all_params.copy()

#         self.anchor_net.outputs = args.activation(self.anchor_net.outputs)

#         self.anchor_net = DenseLayer(self.anchor_net, n_units=keypoint_num, act=tf.identity, name='feature')

#         logits = self.anchor_net.outputs
#         self.predictions = tf.nn.softmax(logits)
#         self.cost = tl.cost.cross_entropy(output=logits, target=anchor_label_placeholder, name='cost')

#         correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), anchor_label_placeholder)
#         self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')


        self.cost_same = tf.norm(self.anchor_net.outputs - self.positive_net.outputs, axis=1)
        print(self.cost_same.shape)
#         assert self.cost_same.shape[0] == args.batch_size
        self.cost_diff = tf.norm(self.anchor_net.outputs - self.negative_net.outputs, axis=1)

        delta_mean = tf.reduce_mean(self.cost_same - self.cost_diff)
        cost_same_mean, cost_same_variance = tf.nn.moments(self.cost_same, axes=[0])
        cost_diff_mean, _ = tf.nn.moments(self.cost_diff, axes=[0])
        
        cv = tf.sqrt(cost_same_variance)/cost_same_mean

        # self.cost = tf.maximum(zero, gap + delta)