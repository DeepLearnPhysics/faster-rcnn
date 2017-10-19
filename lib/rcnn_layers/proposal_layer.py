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
from rcnn_utils.bbox_transform import bbox_transform_inv, clip_boxes
from rcnn_utils.nms_wrapper import nms
#import tensorflow as tf

def proposal_layer_2d(rpn_cls_prob, rpn_bbox_pred, 
                      input_shape, total_stride, 
                      anchors, num_base_anchors,
                      pre_nms_topN=-1, post_nms_topN=-1, nms_thresh=float(0.0)):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  #
  # Step 0) Convert region proposals based on anchors back to proposed boxes
  # Step 1) Select proposed boxes & scores to be kept
  #
  print('pre_nms_topN: {:g}'.format(pre_nms_topN))
  print('post_nms_topN: {:g}'.format(post_nms_topN))
  print('nms_thresh: {:g}'.format(nms_thresh))
  print('im_info: {:s}'.format(input_shape))

  # Get the scores and bounding boxes by keeping only "object exist" entries
  scores = rpn_cls_prob[:, :, :, num_base_anchors:]
  print('\033[93ma1\033[00m {:s}'.format(scores.reshape([scores.size]).mean()))
  # Reshape to a list in the order of anchors
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  # Transform proposals using rpn_bbox_pred (prediction parameters based on anchors)
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  print('\033[93ma2\033[00m {:s}'.format(proposals.reshape([proposals.size]).mean()))
  # Clip boxes that are out-of-image loc
  proposals = clip_boxes(proposals, input_shape[1:3])
  print('\033[93ma3\033[00m {:s}'.format(proposals.reshape([proposals.size]).mean()))
  # Create an index-array sorted by the score values
  order = scores.ravel().argsort()[::-1]

  # Pick the topN region proposals prior to NMS
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]
  print('\033[93ma4\033[00m {:s}'.format(scores.reshape([scores.size]).mean()))
  # Non-maximal suppression w/ IoUs threshold across proposed regions
  keep = nms(np.hstack((proposals, scores)), float(nms_thresh))
  #keep_max = len(proposals)
  #if post_nms_topN > 0:
  #  keep_max = post_nms_topN
  #keep = tf.image.non_max_suppression(proposals, scores, keep_max, nms_threshold)

  # Pick th topN region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]
  print('\033[93ma5\033[00m {:s}'.format(scores.reshape([scores.size]).mean()))
  # Expand the array via np.hstack and zero-filled array to later store an object class prediction.
  # here we assume only-1-image-batch-size to initialize the array size
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
  #print('@end proposal_layer {:s}'.format(blob.shape))
  return blob, scores

