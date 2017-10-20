import os.path as osp
import sys

USE_FLIPPED=0

import datasets
from datasets.factory import get_imdb
import datasets.roidb as rdl_roidb

def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    #imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    #print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    imdb.set_proposal_method('gt')
    print('Set proposal method: {:s}'.format('gt'))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb
