# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Original: Ross Girshick and Sean Bell (https://github.com/rbgirshick/py-faster-rcnn)
# Updated: Xinlei Chen (https://github.com/endernewton/tf-faster-rcnn)
# Updated: Kazuhiro Terao (https://github.com/DeepLearnPhysics/faster-rcnn)
# in chronological order
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from rcnn_bbox_transform import bbox_transform_inv, clip_boxes
from nms_wrapper import nms
#import tensorflow as tf

def proposal_layer_2d(rpn_cls_prob, rpn_bbox_pred, 
                      input_shape, total_stride, 
                      anchors, num_base_anchors,
                      pre_nms_topN=-1, post_nms_topN=-1, nms_thresh=0.0):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  # Get the scores and bounding boxes
  #print('\033[93mrpn_cls_prob \033[00m{:s}'.format(rpn_cls_prob.shape))
  #print('num_anchors {:d}'.format(num_anchors))
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  #print('scores {:s}'.format(scores.shape))
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  #print('scores {:s}'.format(scores.shape))
  #print('anchors {:s}'.format(anchors.shape))
  #print('rpn_bbox_pred {:s}'.format(rpn_bbox_pred.shape))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  #print('proposals {:s}'.format(proposals.shape))
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)
  #keep_max = len(proposals)
  #if post_nms_topN > 0:
  #  keep_max = post_nms_topN
  #keep = tf.image.non_max_suppression(proposals, scores, keep_max, nms_threshold)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores

