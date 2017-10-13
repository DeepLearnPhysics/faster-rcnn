import sys,os

lib_path = os.path.join(os.environ['RCNNDIR'],'lib')
if not lib_path in sys.path:
    sys.path.insert(0,lib_path)

from rcnn_train.trainer import train_net
from rcnn_train.cocodata import cocodata_gen
from vgg_faster_rcnn import vgg

net = vgg()
train_io = cocodata_gen()
val_io   = cocodata_gen()

train_net(net, 'out','log', train_io, val_io, '%s/data/vgg16.ckpt' % os.environ['RCNNDIR'])


