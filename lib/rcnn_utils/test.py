# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# Updated by Kazuhiro Terao
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
import math

from config import cfg, get_output_dir
from rcnn_utils.bbox_transform import clip_boxes, bbox_transform_inv
from rcnn_utils.nms_wrapper import nms

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def im_detect(sess, net, im):

  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  im_scale = float(cfg.TEST.SCALE) / float(im_size_min)
  if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
    im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
  im_resized = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                          interpolation=cv2.INTER_LINEAR)
  input_shape = np.array([1,im_resized.shape[0],im_resized.shape[1],im_resized.shape[2]],dtype=np.float32)
  im_resized  = im_resized.reshape(input_shape.astype(np.int32))

  _, scores, bbox_pred, rois = net.test_image(sess, im_resized, input_shape)
  
  boxes = rois[:, 1:5] / im_scale
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

