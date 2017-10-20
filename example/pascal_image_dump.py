import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys,os,cv2
lib_path = os.path.join(os.environ['RCNNDIR'],'lib')
if not lib_path in sys.path:
    sys.path.insert(0,lib_path)

pascal_keyword='voc_2007_trainval'
for argv in sys.argv:
    if argv.startswith('pascal='): pascal_keyword=argv.replace('pascal=','')

import numpy as np
from rcnn_train.pascaldata import pascaldata_gen
from config import cfg
#cfg=None

datagen = pascaldata_gen(keyword=pascal_keyword,cfg=cfg)

num_images = 1
for argv in sys.argv:
    if argv.isdigit(): num_images = int(argv)

for x in xrange(num_images):

    blob = datagen.forward()

    labels = blob['gt_boxes']
    im = blob['data'][0]
    if datagen.cfg is not None:
        im += datagen.cfg.PIXEL_MEANS[:,:,(2,1,0)]
    im = im.astype(np.uint8)

    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(im,aspect='equal')
    for i in xrange(len(labels)):
        c = int(labels[i][4])
        box = labels[i][0:4]
        print('Class {:s} @ bbox {:s}'.format(datagen.CLASSES[c],box))
        ax.add_patch(plt.Rectangle( (box[0],box[1]), 
                                    box[2]-box[0], 
                                    box[3]-box[1], 
                                    fill=False, edgecolor='red', linewidth=3.5)
                 )
        ax.text(box[0], box[1] - 2,
                '{:s}'.format(datagen.CLASSES[c]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig('out_%04d.png' % x)
