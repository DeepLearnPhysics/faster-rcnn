import sys,os

lib_path = os.path.join(os.environ['RCNNDIR'],'lib')
if not lib_path in sys.path:
    sys.path.insert(0,lib_path)

#from trainval_net import combined_roidb
from datasets.api import combined_roidb
import numpy.random as npr
import numpy as np
import cv2

class cocodata_gen(object):
    
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self,cfg=None):
        
        _,self.roidb = combined_roidb('voc_2007_trainval')

        self.cfg = cfg
        if cfg is not None:
            self.roidb = self._filter_roidb(self.roidb,cfg)

    def num_classes(self):

        return len(self.__class__.CLASSES)

    def _filter_roidb(self,roidb,cfg):
        """Remove roidb entries that have no usable RoIs."""

        def is_valid(entry):
            # Valid images have:
            #   (1) At least one foreground RoI OR
            #   (2) At least one background RoI
            overlaps = entry['max_overlaps']
            # find boxes with sufficient overlap
            fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0

            if not valid: return valid

            gt_inds = []
            if cfg.TRAIN.USE_ALL_GT:
                gt_inds = np.where(entry['gt_classes'] != 0)
            else:
                gt_inds = np.where(entry['gt_classes'] != 0 & np.all(entry['gt_overlaps'].toarray() > -1.0, axis=1))[0]
            valid = len(gt_inds) > 0
            return valid

        num = len(roidb)
        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                           num, num_after))
        return filtered_roidb

    def prep_im_for_blob(self, entry, pixel_means, target_size, max_size):
        """Mean subtract and scale an image for use in a blob."""
        im = cv2.imread(entry['image']).astype(np.float32)
        #im = im.astype(np.float32, copy=False)
        im -= (pixel_means).astype(np.float32)
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        gt_inds = []
        if self.cfg.TRAIN.USE_ALL_GT:
            gt_inds = np.where(entry['gt_classes'] != 0)[0]
        else:
            gt_inds = np.where(entry['gt_classes'] != 0 & np.all(entry['gt_overlaps'].toarray() > -1.0, axis=1))[0]

        if len(gt_inds)<1:
            print('Error: no ground-truth boxes found (unexpected!)')
            raise IndexError

        labels = np.empty((len(gt_inds),5),dtype=np.float32)
        labels[:,0:4] = entry['boxes'][gt_inds,:] * im_scale
        labels[:,4] = entry['gt_classes'][gt_inds]

        return im, labels

    def make_data(self,index=-1,debug=False):
        
        if index < 0:
            index = int(npr.uniform() * len(self.roidb))

        entry = self.roidb[index]
        img,labels = (None,None)
        if self.cfg is None:
            img = cv2.imread(entry['image']).astype(np.float32)
            print('{:g}, {:g}'.format(img.mean(),img.std()))
            bboxes  = entry['boxes']
            classes = entry['gt_classes']
            classes = classes.reshape(len(classes),1)
            labels  = np.hstack([bboxes,classes])
        else:
            # if config is given, do operation
            img,labels = self.prep_im_for_blob(entry, self.cfg.PIXEL_MEANS, self.cfg.TRAIN.SCALES[0], self.cfg.TRAIN.MAX_SIZE)
            print('{:g}, {:g}'.format(img.mean(),img.std()))

        img = img[:,:,(2,1,0)]

        if debug:
            print('\033[93mroi index:\033[00m {:d}'.format(index))
            print('\033[93mroi image:\033[00m {:s}'.format(entry['image']))
            print('\033[93mroi bbox :\033[00m {:s}'.format(labels))

        return img, labels

    def forward(self,index=-1,debug=False):

        img,labels = self.make_data(index,debug)
        img = img.reshape([1,img.shape[0],img.shape[1],3])
        blob = {}
        blob['data'] = img
        blob['im_info'] = np.array([1,img.shape[1],img.shape[2],3])
        blob['gt_boxes'] = labels

        return blob

if __name__ == '__main__':
    from config import cfg
    g=cocodata_gen(cfg)
    blob = g.forward()
    print blob['data'].shape
    print blob['im_info'].shape
    print blob['gt_boxes'].shape
    print blob['gt_boxes']

