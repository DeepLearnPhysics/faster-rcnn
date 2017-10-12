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
from rcnn_utils.reshape import regroup_rpn_channels_2d
from rcnn_utils.anchors import generate_anchors_2d
from rcnn_layers.proposal_layer import proposal_layer_2d
from rcnn_layers.anchor_target_layer import anchor_target_layer_2d
from rcnn_layers.proposal_target_layer import proposal_target_layer_2d

class faster_rcnn(object):

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
        # variable holders
        self._predictions={}
        self._anchor_targets={}
        self._proposal_targets={}
        self._losses = {}
        # summary holders
        self._event_summaries = {}
        self._score_summaries = {}
        self._train_summaries = []
        #self._act_summaries = []

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def configure(self,fname):
        import config
        self._cfg = config.cfg_from_file(fname)
        self._configure()

    def set_input_shape(self,tensor):
        self._input_shape = tf.shape(tensor)

    def create_architecture(self,net,  mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        #self._tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        #assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(self._cfg.TRAIN.WEIGHT_DECAY)
        if self._cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
            rois, cls_prob, bbox_pred = self._build_network(net=net, trainable=training)

        layers_to_output = {'rois': rois}

        # add train summary
        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        #for var in tf.trainable_variables():
        #    self._train_summaries.append(var)

        if testing:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

            val_summaries = []
            with tf.device("/cpu:0"):
                #val_summaries.append(self._add_gt_image_summary())
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
            #    for var in self._act_summaries:
            #        self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def _configure(self):
        # post-action after changing configuration
        self._num_base_anchors = len(self._cfg.ANCHOR_SCALES) * len(self._cfg.ANCHOR_RATIOS)

    def _region_proposal(self, net, trainable, rcnn_initializer=None):
        TEST_MODE='nms'
        if rcnn_initializer is None:
            rcnn_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

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
                          weights_initializer=rcnn_initializer,
                          scope="rpn_conv/3x3")
        # Step 1-a) RPN 4k bbox prediction parameters
        rpn_bbox_pred = slim.conv2d(rpn, self._num_base_anchors * 4, [1, 1], 
                                    trainable=trainable,
                                    weights_initializer=rcnn_initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        # Step 1-b) Generate 2k class scores
        rpn_cls_score = slim.conv2d(rpn, self._num_base_anchors * 2, [1, 1], 
                                    trainable=trainable,
                                    weights_initializer=rcnn_initializer,
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

            # Step 3-b) Map RPN labels to ground-truth boxes. rpn_labels.size == total # of anchors
            rpn_labels = self._anchor_target_layer_2d("anchor_target_layer_2d")

            # Step 3-c) Anchor rois and roi_scores with ground truth
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer_2d(rois, roi_scores, "proposal_target_layer_2d")

        elif TEST_MODE == 'nms':
            rois, _ = self._proposal_layer_2d(rpn_cls_prob, rpn_bbox_pred, "proposal_layer_2d")

        elif TEST_MODE == 'top':
            #rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "proposal_layer_2d")
            raise NotImplementedError
        else:
            raise NotImplementedError

        #self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        #self._predictions["rpn_cls_prob"] = rpn_cls_prob
        #self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        #self._predictions["rois"] = rois

        return rois

    def _head_to_tail(self,net,trainable=True):
        net = tf.reshape(net,(self._cfg.TRAIN.BATCH_SIZE,-1),name='fake_head_to_tail')
        return net

    def _build_network(self, net, trainable=True):
        # select initializers
        if self._cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
            
        #net = self._image_to_head(trainable)
        with tf.variable_scope('faster_rcnn','faster_rcnn'):
            # build the anchors for the image
            self._generate_anchors_2d()
            # region proposal network
            rois = self._region_proposal(net, trainable, initializer)
            # region of interest pooling
            if self._cfg.POOLING_MODE == 'crop':
                rpn_pooling = self._crop_pool_layer_2d(net, rois, "rpn_pooling")
            else:
                raise NotImplementedError

        fc7 = self._head_to_tail(rpn_pooling, trainable)
        with tf.variable_scope('faster_rcnn','faster_rcnn'):
            # region classification
            cls_prob, bbox_pred = self._region_classification_2d(fc7, trainable, 
                                                                 initializer, initializer_bbox)

        #self._score_summaries.update(self._predictions)

        return rois, cls_prob, bbox_pred

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS') as scope:
            # RPN, class loss
            # Object present/absent classification for _every_ anchor (# channel =2)
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            # Object present/absent label for _every_ anchor
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            # Get rpn_bbox_pred from region_proposal
            # ... which is a prediction of objecxt location based on the anchor location
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            # Get rpn_bbox_targets from anchor_target_layer
            # (which is "true distance" of an anchor to a corresponding truth box location)
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
            
            # RCNN, class loss
            # Get cls_score from region_classification, dim = (#box, #class)
            cls_score = self._predictions["cls_score"]
            # Get label from  proposal_target_layer, dim = (#box,1) 
            # ... the value in the axis=1 of "label" is class type integer (0=>22 in case of faster rcnn)
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            # Because label is just an integer (i.e. not a "hot label" array), use sparse
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

            # RCNN, bbox loss
            # Get bbox_pred from region_classification, dim = (#box, 4)
            # (which is a prediction from cropped ROI region to a better truth box location)
            bbox_pred = self._predictions['bbox_pred']
            # Get bbox_targets from region_classification 
            # (which is "true distance" of a cropped ROI region to a corresponding truth box location)
            bbox_targets = self._proposal_targets['bbox_targets']
            # ???
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            # ???
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            # Now make L1 loss
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
            
            #self._losses['cross_entropy'] = cross_entropy
            #self._losses['loss_box'] = loss_box
            #self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            #self._losses['rpn_loss_box'] = rpn_loss_box
            
            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

            if len(tf.losses.get_regularization_losses())>0:
                regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
                self._losses['total_loss'] = loss + regularization_loss
            else:
                print('\033[95mWARNING\033[00m no weights regularizer found ... you sure????')
                
            #self._event_summaries.update(self._losses)
            
            return loss

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

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
        self._anchor_targets['rpn_labels'] = rpn_labels
        self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

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

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
            
            #self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    def _crop_pool_layer_2d(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            print('{:s}'.format(rois))
            print('rois shape @ crop_pool_layer_2d {:s}'.format(rois.shape))
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._total_stride[1])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._total_stride[2])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = self._cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, 
                                             tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _region_classification_2d(self, fc7, trainable, initializer, initializer_bbox):
        cls_score = slim.fully_connected(fc7, self._num_classes, 
                                         weights_initializer=initializer,
                                         trainable=trainable,
                                         activation_fn=None, scope='cls_score')
        cls_prob = tf.nn.softmax(cls_score,name="cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                         weights_initializer=initializer_bbox,
                                         trainable=trainable,
                                         activation_fn=None, scope='bbox_pred')

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

if __name__ == '__main__':

    import sys
    if len(sys.argv) == 1:
        rpn()
        sys.exit(0)
    else:
        net = faster_rcnn()
        if sys.argv[1] == 'generate_anchors_2d':
            image = tf.placeholder(tf.float32,[1,256,512,3])
            net.set_input_shape(image)
            # Create a session
            sess = tf.InteractiveSession()
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            ret = sess.run(net._anchors,feed_dict={})
            print('{:s}'.format(ret))
        if sys.argv[1] == 'rpn':
            # create a 0-filled image tensor
            image = tf.placeholder(tf.float32,[1,256,512,3])
            net.set_input_shape(image)
            net._generate_anchors_2d()
            # create a 0-filled feature map for rpn
            bottom = tf.placeholder(tf.float32,[1,16,32,64])
            roi = net._region_proposal(net=bottom, trainable=True)
            # Create a session
            sess = tf.InteractiveSession()
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            sess.run([roi],
                     feed_dict = { bottom:np.zeros((1,16,32,64),np.float32),
                                   net._gt_boxes:np.array(((1,10,10,20,20),)) } )
        if sys.argv[1] == 'build':
            # create a 0-filled image tensor
            image = tf.placeholder(tf.float32,[1,256,512,3])
            net.set_input_shape(image)
            #net._generate_anchors_2d()
            # create a 0-filled feature map for rpn
            bottom = tf.placeholder(tf.float32,[1,16,32,64])
            roi,cls_prob, bbox_pred = net._build_network(net=bottom, trainable=True)
            loss = net._add_losses()
            # Create a session
            sess = tf.InteractiveSession()
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            sess.run([roi,cls_prob,bbox_pred,loss], 
                     feed_dict = { bottom:np.zeros((1,16,32,64),np.float32),
                                   net._gt_boxes:np.array(((1,10,10,20,20),)) } )
        if sys.argv[1] == 'architecture':
            image = tf.placeholder(tf.float32, [1,256,512,3])
            net.set_input_shape(image)
            bottom = tf.placeholder(tf.float32,[1,16,32,64])
            layers_to_output = net.create_architecture(net = bottom, mode = 'TRAIN', num_classes=5)
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            sess.run([layers_to_output], feed_dict={ bottom: np.zeros((1,16,32,64), np.float32),
                                                     net._gt_boxes:np.array(((1,10,10,20,20),))})
