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

# faster-rcnn imports
from config import cfg as DEFAULT_CFG
from rcnn_reshape import regroup_rpn_channels_2d
from rcnn_anchors import generate_anchors_2d
from proposal_layer import proposal_layer_2d
from anchor_target_layer import anchor_target_layer_2d
from proposal_target_layer import proposal_target_layer_2d

class rpn(object):

    def __init__(self):
        # variable holders (to be generated)
        self._anchors = None
        self._num_base_anchors = -1
        self._num_classes = 21
        # variable holders (to be specified via user/data)
        #self._input_data = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        #self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._input_shape  = []
        self._total_stride = [-1,16,16,-1]
        # configuration parameter holders
        self._cfg = DEFAULT_CFG
        self._configure()

    def configure(self,fname):
        import config
        self._cfg = config.cfg_from_file(fname)
        self._configure()

    def _configure(self):
        # post-action after changing configuration
        self._num_base_anchors = len(self._cfg.ANCHOR_SCALES) * len(self._cfg.ANCHOR_RATIOS)

    def set_input_shape(self,tensor):
        self._input_shape = tf.shape(tensor)

    def build(self, net, trainable):
        TEST_MODE='nms'
        INITIALIZER=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

        #
        # Step of operations:
        # 0) Convolution layer ... shared by RPN and detection
        # 1) Two parallel convolution layers ... 4k region proposals (bbox) and 2k object-ness classification (cls) 
        # 2) Reshaping + softmax + re-reshaping to get candidate ROIs
        # 3) Select a sub-set of ROIs and scores from proposal_layer

        # Step 0) Convolution for RPN/Detection shared layer
        rpn = slim.conv2d(net, 
                          self._cfg.TRAIN.RPN_BATCHSIZE, 
                          self._cfg.TRAIN.RPN_KERNELS,
                          trainable=trainable, 
                          weights_initializer=INITIALIZER,
                          scope="rpn_conv/3x3")
        # Step 1-a) RPN 4k bbox prediction parameters
        rpn_bbox_pred = slim.conv2d(rpn, self._num_base_anchors * 4, [1, 1], 
                                    trainable=trainable,
                                    weights_initializer=INITIALIZER,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        # Step 1-b) Generate 2k class scores
        rpn_cls_score = slim.conv2d(rpn, self._num_base_anchors * 2, [1, 1], 
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
        rpn_cls_prob_reshape = tf.reshape(tf.nn.softmax(tf.reshape(rpn_cls_score_reshape, 
                                                                   [-1, tf.shape(rpn_cls_score_reshape)[-1]]),
                                                        name='rpn_cls_prob_reshape'),
                                          tf.shape(rpn_cls_score_reshape))
        # Step 2-c) Now put back into 2k shape
        rpn_cls_prob = regroup_rpn_channels_2d(bottom=rpn_cls_prob_reshape, 
                                               num_dim=self._num_base_anchors*2,
                                               name='rpn_cls_prob')
        rois,roi_scores = (None,None)
        # Step 3) Get a (meaningful) subset of rois and associated scores
        if trainable:
            # Step 3-a) Derive predicted bbox (rois) with scores (roi_scores) from prediction parameters (rpn_bbox_pred)
            #           and anchors. Some boxes are filtered out based on NMS of proposed regions and objectness
            #           probability (rpn_cls_prob)
            rois, roi_scores = self._proposal_layer_2d(rpn_cls_prob, rpn_bbox_pred, trainable, "proposal_layer_2d")

            # Step 3-b) Map RPN labels to ground-truth boxes
            rpn_labels = self._anchor_target_layer_2d("anchor_target_layer_2d")

            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer_2d(rois, roi_scores, "proposal_target_layer_2d")

        elif TEST_MODE == 'nms':
            #rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            pass
        elif TEST_MODE == 'top':
            #rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            pass
        else:
            raise NotImplementedError
        return rois, roi_scores

    #
    # tf.py_func wrappers
    #
    def _generate_anchors_2d(self):
        with tf.variable_scope('generate_anchors_2d') as scope:
            height = tf.to_int32(tf.ceil(self._input_shape[1] / np.float32(self._total_stride[1])))
            width  = tf.to_int32(tf.ceil(self._input_shape[2] / np.float32(self._total_stride[2])))
            anchors, num_base_anchors = tf.py_func(generate_anchors_2d,
                                                   [height, width, self._total_stride,
                                                    np.array(self._cfg.ANCHOR_SCALES),
                                                    np.array(self._cfg.ANCHOR_RATIOS)],
                                                   [tf.float32, tf.int32],
                                                   name='generate_anchors_2d')
            # assert dimension
            anchors.set_shape([None,4])
            num_base_anchors.set_shape([])
            self._anchors = anchors



    def _proposal_layer_2d(self,rpn_cls_prob, rpn_bbox_pred, trainable, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_layer_2d,
                                          [rpn_cls_prob, rpn_bbox_pred, self._input_shape,
                                           self._total_stride, self._anchors, self._num_base_anchors,
                                           np.int32(self._cfg.TRAIN.RPN_PRE_NMS_TOP_N),
                                           np.int32(self._cfg.TRAIN.RPN_POST_NMS_TOP_N),
                                           np.float32(self._cfg.TRAIN.RPN_NMS_THRESH)],
                                          [tf.float32, tf.float32], name="proposal_layer_2d")
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])
            
        return rois, rpn_scores

    def _anchor_target_layer_2d(self, name):
        with tf.variable_scope(name) as scope:
            height = tf.to_int32(tf.ceil(self._input_shape[1] / np.float32(self._total_stride[1])))
            width  = tf.to_int32(tf.ceil(self._input_shape[2] / np.float32(self._total_stride[2])))
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights \
            = tf.py_func(anchor_target_layer_2d,
                         [height, width, self._gt_boxes, self._input_shape, 
                          self._total_stride, self._anchors, self._num_base_anchors],
                         [tf.float32, tf.float32, tf.float32, tf.float32],
                         name="anchor_target")

        rpn_labels.set_shape([1, 1, None, None])
        rpn_bbox_targets.set_shape([1, None, None, self._num_base_anchors * 4])
        rpn_bbox_inside_weights.set_shape([1, None, None, self._num_base_anchors * 4])
        rpn_bbox_outside_weights.set_shape([1, None, None, self._num_base_anchors * 4])

        rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
        #self._anchor_targets['rpn_labels'] = rpn_labels
        #self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        #self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        #self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        #self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer_2d(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer_2d,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([self._cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([self._cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([self._cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([self._cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([self._cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([self._cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            #self._proposal_targets['rois'] = rois
            #self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            #self._proposal_targets['bbox_targets'] = bbox_targets
            #self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            #self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
            
            #self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

if __name__ == '__main__':

    import sys
    if len(sys.argv) == 1:
        rpn()
        sys.exit(0)
    else:
        net = rpn()
        if sys.argv[1] == 'generate_anchors_2d':
            image = tf.placeholder(tf.float32,[1,256,512,3])
            net.set_input_shape(image)
            net._generate_anchors_2d()
            # Create a session
            sess = tf.InteractiveSession()
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            ret = sess.run(net._anchors,feed_dict={})
            print('{:s}'.format(ret))
        if sys.argv[1] == 'build':
            # create a 0-filled image tensor
            image = tf.placeholder(tf.float32,[1,256,512,3])
            net.set_input_shape(image)
            net._generate_anchors_2d()
            # create a 0-filled feature map for rpn
            bottom = tf.placeholder(tf.float32,[1,16,32,64])
            roi,roi_scores = net.build(net=bottom, trainable=True)
            # Create a session
            sess = tf.InteractiveSession()
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            sess.run([roi,roi_scores], 
                     feed_dict = { bottom:np.zeros((1,16,32,64),np.float32),
                                   net._gt_boxes:np.array(((1,10,10,20,20),)) } )
