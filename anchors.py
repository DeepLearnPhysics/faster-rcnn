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

import numpy as np

def generate_image_anchors(width, height, total_strides, 
                           anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  anchors = generate_base_anchors(base_size=total_strides,
                                  ratios=np.array(anchor_ratios), 
                                  scales=np.array(anchor_scales))
  print('anchors {:s}'.format(anchors.shape))
  print('width {:f}'.format(np.float32(width)))
  print('height {:f}'.format(np.float32(height)))
  print('total_stride {:d}'.format(total_strides))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * total_strides
  shift_y = np.arange(0, height) * total_strides
  print('shift_x {:s}'.format(shift_x))
  print('shift_y {:s}'.format(shift_y))
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  print('shift_x {:s}'.format(shift_x))
  print('shift_y {:s}'.format(shift_y))
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  print('shifts {:s}'.format(shifts.shape))
  K = shifts.shape[0]
  print('K={:d}, A={:d}'.format(K,A))
  # width changes faster, so here it is H, W, C                                                                              
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  print('anchors {:s}'.format(anchors.shape))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  print('anchors {:s}'.format(anchors.shape))
  length = np.int32(anchors.shape[0])
  print('# anchors {:d}'.format(length))
  return anchors, length

def generate_base_anchors(base_size=16, ratios=[0.5, 1, 2],
                          scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_base_anchors()
    dt = time.time() - t
    print "generate_base_anchors():", dt
    print a
    print
    t = time.time()
    a,l = generate_image_anchors(32,32,16)
    dt = time.time() - t
    print "generate_image_anchors():", dt
    print np.array(a).shape
    print
    #from IPython import embed; embed()
