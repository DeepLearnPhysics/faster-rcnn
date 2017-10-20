import sys,os

lib_path = os.path.join(os.environ['RCNNDIR'],'lib')
if not lib_path in sys.path:
    sys.path.insert(0,lib_path)

from config import cfg, cfg_from_file
for argv in sys.argv:
    if argv.endswith('.yml'): cfg_from_file(argv)

pascal_keyword='voc_2007_trainval'
for argv in sys.argv:
    if argv.startswith('pascal='): pascal_keyword=argv.replace('pascal=','')

from rcnn_train.trainer import train_net
from rcnn_train.pascaldata import pascaldata_gen
from vgg_faster_rcnn import vgg

net = vgg()
train_io = pascaldata_gen(keyword=pascal_keyword,cfg=net._cfg)
val_io   = pascaldata_gen(keyword=pascal_keyword,cfg=net._cfg)

train_net(net, 'output/vgg','tensorboard/vgg', train_io, val_io, '%s/data/vgg16.ckpt' % os.environ['RCNNDIR'])


