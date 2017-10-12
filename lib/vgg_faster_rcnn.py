# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
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
    self._layers['head'] = net
    
    return net

  def _head_to_tail(self, net, is_training=True, reuse=False):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, 
                            scope='dropout6')
      rcnn_input = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        rcnn_input = slim.dropout(rcnn_input, keep_prob=0.5, is_training=True, 
                            scope='dropout7')

    return rcnn_input

