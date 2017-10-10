# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Original: Ross Girshick and Sean Bell (https://github.com/rbgirshick/py-faster-rcnn)
# Updated: Xinlei Chen (https://github.com/endernewton/tf-faster-rcnn)
# Updated: Kazuhiro Terao (https://github.com/DeepLearnPhysics/faster-rcnn)
# in chronological order
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def regroup_rpn_channels_2d(bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
        # change the channel to the caffe format
        to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
        # then force it to have channel 2, putting extra into "H" channel (default tf "NHWC")
        reshaped = tf.reshape(to_caffe,
                              tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
                              #tf.concat(axis=0, values=[[1, num_dim, -1], [32]]))
        # then swap the channel back
        to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf

def regroup_rpn_channels_3d(bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
        # change the channel to the caffe format
        to_caffe = tf.transpose(bottom, [0, 4, 1, 2, 3])
        # then force it to have channel 2, putting extra into "D" channel (default tf "NDHWC")
        reshaped = tf.reshape(to_caffe,
                              tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[3]]]))
                              #tf.concat(axis=0, values=[[1, num_dim, -1], [32,48]]))
        # then swap the channel back
        to_tf = tf.transpose(reshaped, [0, 2, 3, 4, 1])
        return to_tf

if __name__ == '__main__':

    print('\033[95mTo get useful output comment-out line 19 and 32, and un-comment line 20 and 33\033[00m')

    bottom_2d = tf.placeholder(tf.float32,[1,16,32,3])
    a2d = regroup_rpn_channels_2d(bottom_2d, 2, "aho")
    print('regroup_rpn_channels_2d changed shape {:s} => {:s}'.format(bottom_2d.shape,a2d.shape))

    bottom_3d = tf.placeholder(tf.float32,[1,16,32,48,3])
    a3d = regroup_rpn_channels_3d(bottom_3d, 2, "aho")
    print('regroup_rpn_channels_3d changed shape {:s} => {:s}'.format(bottom_3d.shape,a3d.shape))

