import sys,os

lib_path = os.path.join(os.environ['RCNNDIR'],'lib')
if not lib_path in sys.path:
    sys.path.insert(0,lib_path)

#from trainval_net import combined_roidb
from datasets.api import combined_roidb
import numpy.random as npr
import numpy as np
import cv2

def _filter_roidb(roidb,cfg):
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
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb

class cocodata_gen(object):
    
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self,cfg=None):
        
        _,self.roidb = combined_roidb('voc_2007_trainval')
        
        if cfg is not None:
            self.roidb = _filter_roidb(self.roidb,cfg)

        self.TRAIN_SCALES = cfg.TRAIN.SCALES
        self.PIXEL_MEANS = cfg.PIXEL_MEANS
        self.MAX_SIZE = cfg.TRAIN.MAX_SIZE

    def num_classes(self):

        return len(self.__class__.CLASSES)

    def get_minibatch(roidb, num_classes):
        """Given a roidb, construct a minibatch sampled from it."""
        num_images = len(roidb)
        # Sample random scales to use for each image in this batch
        random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
        assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

        # Get the input image blob, formatted for caffe
        im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

        blobs = {'data': im_blob}
        
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
  
        # gt boxes: (x1, y1, x2, y2, cls)
        if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
            gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
            gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]],
                                    dtype=np.float32)
        
        return blobs

    def _get_image_blob(roidb, scale_inds):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = len(roidb)
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
            im_scales.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, im_scales

    def im_list_to_blob(ims):
        """Convert a list of images into a network input.
        Assumes images are already prepared (means subtracted, BGR order, ...). """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        return blob


    def prep_im_for_blob(self, im, pixel_means, target_size, max_size):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                  interpolation=cv2.INTER_LINEAR)
        return im


    def make_data(self,index=-1,debug=False):
        
        if index < 0:
            index = int(npr.uniform() * len(self.roidb))

        img = cv2.imread(self.roidb[index]['image'])
        target_size = self.TRAIN_SCALES[0]
        print target_size
        img = self.prep_im_for_blob(img, self.PIXEL_MEANS, target_size, self.MAX_SIZE)
        img = img[:,:,(2,1,0)]

        classes = self.roidb[index]['gt_classes']
        bboxes  = self.roidb[index]['boxes']
        classes = classes.reshape(len(classes),1)

        print('\033[93mroi index:\033[00m {:d}'.format(index))
        print('\033[93mroi image:\033[00m {:s}'.format(self.roidb[index]['image']))
        print('\033[93mroi bbox :\033[00m {:s}'.format(bboxes))

        return img, np.hstack([bboxes,classes])

    def forward(self,index=-1,debug=False):

        img,bboxes = self.make_data(index,debug)
        img = img.reshape([1,img.shape[0],img.shape[1],3])
        blob = {}
        blob['data'] = img
        blob['im_info'] = np.array([1,img.shape[1],img.shape[2],3])
        blob['gt_boxes'] = np.array(bboxes)

        return blob

if __name__ == '__main__':
    g=cocodata_gen()
    blob = g.forward()
    print blob['data'].shape
    print blob['im_info'].shape
    print blob['gt_boxes'].shape
    print blob['gt_boxes']

