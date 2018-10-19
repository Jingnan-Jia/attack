#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:30:24 2018

@author: jiajingnan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


slim = tf.contrib.slim


def lenet(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
  end_points = {}
  with tf.variable_scope(scope, 'LeNet', [images],reuse = tf.AUTO_REUSE):
    net = end_points['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = end_points['conv2'] = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = end_points['fc3'] = slim.fully_connected(net, 1024, scope='fc3')
    if not num_classes:
      return net, end_points
    net = end_points['dropout3'] = slim.dropout(
        net, dropout_keep_prob, is_training=is_training, scope='dropout3')
    logits = end_points['Logits'] = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='fc4')

  end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  
  print(logits.shape)
  jcobin = tf.gradients(logits[0], images)
  return logits, end_points, jcobin
lenet.default_image_size = 28


def my_load_dataset(dataset = 'mnist'):
    '''
    
    
    '''

    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
        img_rows, img_cols, img_chns = 32, 32, 3
        
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, img_chns = 28, 28, 1
        
    # unite different shape formates to the same one
    x_train = np.reshape(x_train, (-1 , img_rows, img_cols, img_chns)).astype(np.float32)
    x_test = np.reshape(x_test, (-1, img_rows, img_cols, img_chns)).astype(np.float32)
    
     # change labels shape to (-1, )
    y_train = np.reshape(y_train, (-1 ,)).astype(np.int32)
    y_test = np.reshape(y_test, (-1 ,)).astype(np.int32)
        
# =============================================================================
#     x_train = (x_train - img_depth/2) / img_depth
#     x_train = (x_train - img_depth/2) / img_depth
# =============================================================================
    print('load dataset ' + str(dataset) + ' finished')
    print('train_size:', x_train.shape)
    print('test_size:', x_test.shape)
    print('train_labels_shape:', y_train.shape)
    print('test_labels_shape:', y_test.shape)
    
    return x_train, y_train, x_test, y_test

def lenet_arg_scope(weight_decay=0.0):
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      activation_fn=tf.nn.relu) as sc:
    return sc


mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
x_train, y_train, x_test, y_test = my_load_dataset()

x_input = tf.placeholder(tf.float32, [None, 28, 28, 1])
with slim.arg_scope(lenet_arg_scope()):
    _, end_points, grad = lenet(
        x_input, num_classes=10, is_training=False) 
saver = tf.train.Saver(slim.get_model_variables())
# =============================================================================
# 
# 
# with tf.Session() as sess:
#     saver.restore(sess, './lenet-model/model.ckpt-20000')
# 
#     image_grad = sess.run(grad, feed_dict={x_input:x_train[12].reshape(-1, 28, 28, 1)})
#     image_grad = np.array(image_grad)
#     image_grad = image_grad.reshape(28, 28)
#     image_grad = np.abs(image_grad)
#     print(image_grad.shape, np.max(image_grad), np.min(image_grad))
# =============================================================================
