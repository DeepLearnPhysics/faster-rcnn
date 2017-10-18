# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# Updated by Kazuhiro Terao and Ji Won Park
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from faster_rcnn import faster_rcnn
from config import cfg

class vgg(faster_rcnn):
  def __init__(self):
    faster_rcnn.__init__(self)
    self._total_strides = [1,16,16,1]
    #self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'vgg_16'

  def _image_to_head(self, is_training=True, reuse=False):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                          trainable=False, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

    #self._act_summaries.append(net)
    #self._layers['head'] = net
    
    return net

  def _head_to_tail(self, net, is_training=True, reuse=False):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      net_flat = slim.flatten(net, scope='flatten')
      fc6 = slim.fully_connected(net_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, 
                            scope='dropout6')
      rcnn_input = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        rcnn_input = slim.dropout(rcnn_input, keep_prob=0.5, is_training=True, 
                            scope='dropout7')

    return rcnn_input

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == (self._scope + '/fc6/weights:0') or \
         v.name == (self._scope + '/fc7/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if var_keep_dic is not None and v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv, 
                                      self._scope + "/fc7/weights": fc7_conv,
                                      self._scope + "/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                            self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                            self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))

        
