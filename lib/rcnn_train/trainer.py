# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg
from rcnn_utils.timer import Timer
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  def __init__(self, sess, network, output_dir, tbdir, train_io, val_io=None, pretrained_model=None):
    self.net = network
    self.output_dir = output_dir
    self.tbdir = tbdir
    self.train_io = train_io
    self.val_io = val_io
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = ''
    if self.val_io:
      self.tbvaldir = self.tbdir + '_val'
      if not os.path.exists(self.tbvaldir):
        os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, sess, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(self.output_dir, filename)
    self.saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    return filename

  def from_snapshot(self, sess, sfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    num_iteration = int(sfile.split('_')[-1].replace('.ckpt',''))
    self.saver.restore(sess, sfile)
    print('Restored.')
    return num_iteration

  def get_variables_in_checkpoint_file(self, file_name):
    print('\033[93m HEY HEY HEY \033[00m')
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      print('var_to_shape_map {:s}'.format(var_to_shape_map))
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")

  def construct_graph(self, sess):
    with sess.graph.as_default():
      # Set the random seed for tensorflow
      tf.set_random_seed(cfg.RNG_SEED)
      # Build the main computation graph
      layers = self.net.create_architecture(self.train_io.num_classes(), 'TRAIN', tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
      # Define the loss
      loss = layers['total_loss']
      # Set learning rate and momentum
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
      self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

      # Compute the gradients with regard to the loss
      gvs = self.optimizer.compute_gradients(loss)
      # Double the gradient of the bias if set
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.multiply(grad, scale)
            final_gvs.append((grad, var))
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        train_op = self.optimizer.apply_gradients(gvs)

      # We will handle the snapshots ourselves
      self.saver = tf.train.Saver(max_to_keep=100000)
      # Write the train and validation information to tensorboard
      self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
      self.valwriter = tf.summary.FileWriter(self.tbvaldir)

    return lr, train_op

  def find_previous(self):
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize+1)))
    sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

    lsf = len(sfiles)
    return lsf, sfiles

  def initialize(self, sess):
    # Initial file lists are empty
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    variables = tf.global_variables()
    # Initialize all variables first
    sess.run(tf.variables_initializer(variables, name='init'))

    # Get the variables to restore, ignoring the variables to fix
    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
    variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, self.pretrained_model)
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    self.net.fix_variables(sess, self.pretrained_model)
    print('Fixed.')
    last_snapshot_iter = 0
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return rate, last_snapshot_iter, stepsizes, ss_paths

  def restore(self, sess, sfile):
    # Get the most recent snapshot and restore
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sess, sfile)
    # Set the learning rate
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        rate *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)

    return rate, last_snapshot_iter, stepsizes, ss_paths

  def remove_snapshot(self, ss_paths):
    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      if os.path.exists(str(sfile)):
        os.remove(str(sfile))
      else:
        os.remove(str(sfile + '.data-00000-of-00001'))
        os.remove(str(sfile + '.index'))
      sfile_meta = sfile + '.meta'
      os.remove(str(sfile_meta))
      ss_paths.remove(sfile)

  def train_model(self, sess, max_iters):
    # Construct the computation graph
    lr, train_op = self.construct_graph(sess)

    # Find previous snapshots if there is any to restore from
    lsf, sfiles = self.find_previous()

    # Initialize the variables or restore them from the last snapshot
    if lsf == 0:
      rate, last_snapshot_iter, stepsizes, ss_paths = self.initialize(sess)
    else:
      rate, last_snapshot_iter, stepsizes, ss_paths = self.restore(sess, str(sfiles[-1]))

    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()
    while iter < max_iters + 1:
      # Learning rate
      if iter == next_stepsize + 1:
        # Add snapshot here before reducing the learning rate
        self.snapshot(sess, iter)
        rate *= cfg.TRAIN.GAMMA
        sess.run(tf.assign(lr, rate))
        next_stepsize = stepsizes.pop()

      timer.tic()
      # Get training data, one batch at a time
      blobs = self.train_io.forward()

      now = time.time()
      if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
        # Compute the graph with summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
          self.net.train_step_with_summary(sess, blobs, train_op)
        self.writer.add_summary(summary, float(iter))
        # Also check the summary on the validation set
        blobs_val = self.val_io.forward()
        summary_val = self.net.get_summary(sess, blobs_val)
        self.valwriter.add_summary(summary_val, float(iter))
        last_summary_time = now
      else:
        # Compute the graph without summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
          self.net.train_step(sess, blobs, train_op)
      timer.toc()

      # Display training information
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
              '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
              (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
        print('speed: {:.3f}s / iter'.format(timer.average_time))

      # Snapshotting
      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        ss_path = self.snapshot(sess, iter)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        if len(ss_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(ss_paths)

      iter += 1

    if last_snapshot_iter != iter - 1:
      self.snapshot(sess, iter - 1)

    self.writer.close()
    self.valwriter.close()

def train_net(network, output_dir, tb_dir,
              train_io, val_io=None,
              pretrained_model=None,
              max_iters=40000):
  """Train a Faster R-CNN network."""
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    sw = SolverWrapper(sess, network, output_dir, tb_dir,
                       train_io, val_io,
                       pretrained_model=pretrained_model)
    print('Solving...')
    sw.train_model(sess, max_iters)
    print('done solving')
