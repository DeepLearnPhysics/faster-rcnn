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

    def __init__(self):
        
        _,self.roidb = combined_roidb('voc_2007_trainval')

    def num_classes(self):

        return len(self.__class__.CLASSES)

    def make_data(self,index=-1,debug=False):
        
        if index < 0:
            index = int(npr.uniform() * len(self.roidb))

        img = cv2.imread(self.roidb[index]['image'])
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

