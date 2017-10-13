import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys,os,cv2
lib_path = os.path.join(os.environ['RCNNDIR'],'lib')
if not lib_path in sys.path:
    sys.path.insert(0,lib_path)

#from trainval_net import combined_roidb
from datasets.api import combined_roidb

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

num_images = 1
if len(sys.argv)>1:
    num_images = int(sys.argv[1])

imdb,roidb = combined_roidb('voc_2007_trainval')

for x in xrange(num_images):
    
    image_path = roidb[x]['image']
    classes    = roidb[x]['gt_classes']
    boxes      = roidb[x]['boxes']
    
    im = cv2.imread(image_path)
    im = im[:,:,(2,1,0)]
    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(im,aspect='equal')
    for i in xrange(len(classes)):
        c = classes[i]
        box = boxes[i]
        print('Class {:s} @ bbox {:s}'.format(CLASSES[c],box))
        ax.add_patch(plt.Rectangle( (box[0],box[1]), 
                                    box[2]-box[0], 
                                    box[3]-box[1], 
                                    fill=False, edgecolor='red', linewidth=3.5)
                 )
        ax.text(box[0], box[1] - 2,
                '{:s}'.format(CLASSES[c]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig('out_%04d.png' % x)
