# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Original by Xinlei Chen (https://github.com/endernewton/tf-faster-rcnn)
# Updated by Kazuhiro Terao (https://github.com/DeepLearnPhysics/faster-rcnn)
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np
from rcnn_reshape import regroup_rpn_channels_2d

def build(net, trainable):
    NUM_ANCHORS=9
    RPN_CHANNELS=512
    RPN_KERNEL=[3,3]
    RPN_STRIDE=[1,1]
    TEST_MODE='nms'
    INITIALIZER=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

    #
    # Step of operations:
    # 0) Convolution layer ... shared by RPN and detection
    # 1) Two parallel convolution layers ... 4k region proposals (bbox) and 2k object-ness classification (cls) 
    # 2) Reshaping + softmax + re-reshaping to get candidate ROIs
    
    # Step 0) Convolution for RPN/Detection shared layer
    rpn = slim.conv2d(net, RPN_CHANNELS, [3, 3], 
                      trainable=trainable, 
                      weights_initializer=INITIALIZER,
                      scope="rpn_conv/3x3")
    # Step 1-a) RPN 4k bbox prediction
    rpn_bbox_pred = slim.conv2d(rpn, NUM_ANCHORS * 4, [1, 1], 
                                trainable=trainable,
                                weights_initializer=INITIALIZER,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    # Step 1-b) Generate 2k class scores
    rpn_cls_score = slim.conv2d(rpn, NUM_ANCHORS * 2, [1, 1], 
                                trainable=trainable,
                                weights_initializer=INITIALIZER,
                                padding='VALID', 
                                activation_fn=None, 
                                scope='rpn_cls_score')
    # Step 2-a) Reshape such that num. channel=2
    rpn_cls_score_reshape = regroup_rpn_channels_2d(bottom=rpn_cls_score, 
                                                    num_dim=2, 
                                                    name='rpn_cls_score_reshape')
    # Step 2-b) Compute softmax
    rpn_cls_prob_reshape = tf.reshape(tf.nn.softmax(tf.reshape(rpn_cls_score_reshape, [-1, tf.shape(rpn_cls_score_reshape)[-1]]),
                                                    name='rpn_cls_prob_reshape'),
                                      tf.shape(rpn_cls_score_reshape))
    # Step 2-c) Now put back into 2k shape
    rpn_cls_prob = regroup_rpn_channels_2d(bottom=rpn_cls_prob_reshape, 
                                           num_dim=NUM_ANCHORS*2,
                                           name='rpn_cls_prob')

    # Step 2-d) Get a (meaningful) subset of rois and associated scores
    if trainable:
        
        pass
    elif TEST_MODE == 'nms':
        #rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        pass
    elif TEST_MODE == 'top':
        #rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        pass
    else:
        raise NotImplementedError

if __name__ == '__main__':

    bottom = tf.placeholder(tf.float32,[1,16,32,64])
    build(net=bottom, trainable=True)
