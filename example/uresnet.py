import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import uresnet_layers as L
from resnet_module import double_resnet

def build(input_tensor, num_class=4):

    net = input_tensor

    # assume zero padding in each layer (set as default in uresnet_layers.py)

    # downsampling path

    # initial 7x7 convolution
    net = L.conv2d(input_tensor=net, name='conv7x7_begin', kernel=(7,7), stride=(1,1), num_filter=64, activation_fn=tf.nn.relu)
    downsampling0 = net # save for concatenation with upsampling0 later
    # max pool
    net = L.max_pool(input_tensor=net, name="max_pool", kernel=(3,3), stride=(2,2))

    # 2x resnet: input is 256 x 256 x 64
    net = double_resnet(input_tensor=net, filter_factor=1, dim_factor=1, input_filters=64, num_classes=num_class, step=0)
    downsampling1 = net

    # 2x resnet: input is 256 x 256 x 64
    net = double_resnet(input_tensor=net, filter_factor=2, dim_factor=2, input_filters=64, num_classes=num_class, step=1)
    downsampling2 = net

    # 2x resnet: input is 128 x 128 x 128
    net = double_resnet(input_tensor=net, filter_factor=2, dim_factor=2, input_filters=128, num_classes=num_class, step=2)
    downsampling3 = net

    # 2x resnet: input is 64 x 64 x 256
    net = double_resnet(input_tensor=net, filter_factor=2, dim_factor=2, input_filters=256, num_classes=num_class, step=3)
    downsampling4 = net

    # 2x resnet: input is 32 x 32 x 512
    print(net.shape, "before step 4")
    net = double_resnet(input_tensor=net, filter_factor=2, dim_factor=2, input_filters=512, num_classes=num_class, step=4)
    print(net.shape, "after step 4")

    # upsampling path

    # deconvolution 0
    net = L.deconv2d(input_tensor=net, name='deconv_0', kernel=(3,3), stride=(2,2), output_num_filter=512) 
    print(net.shape, "afterdeconv0")

    # concatenate downsampling4 and upsampling4
    upsampling4 = net # could go with "net" but just for readability, define new variable
    net = tf.concat([downsampling4, upsampling4], axis=3)
    
    # 2x resnet indexed 5: input is 32 x 32 x (512+512)
    net = double_resnet(input_tensor=net, filter_factor=0.5, dim_factor=1, input_filters=1024, num_classes=num_class, step=5)
    print(net.shape, "after step 5, should be 32 32 512")
    
    # deconvolution 1
    net = L.deconv2d(input_tensor=net, name='deconv_1', kernel=(4,4), stride=(2,2), output_num_filter=256)
    print(net.shape, "afterdeconv1, should be 64 64 256")

    # concatenate downsampling3 and upsampling3    
    upsampling3 = net
    net = tf.concat([downsampling3, upsampling3], axis=3)

    # 2x resnet indexed 6: input is 64 x 64 x (256+256)
    net = double_resnet(input_tensor=net, filter_factor=0.5, dim_factor=1, input_filters=512, num_classes=num_class, step=6)
    print(net.shape, "after step 6, should be 64 64 256")
    
    # deconvolution 2
    net = L.deconv2d(input_tensor=net, name='deconv_2', kernel=(4,4), stride=(2,2), output_num_filter=128)
    print(net.shape, "afterdeconv2, should be 128 128 128")

    # concatenate downsampling2 and upsampling2
    upsampling2 = net
    net = tf.concat([downsampling2, upsampling2], axis=3)

    # 2x resnet indexed 7: input is 128 x 128 x (128+128)
    net = double_resnet(input_tensor=net, filter_factor=0.5, dim_factor=1, input_filters=256, num_classes=num_class, step=7)
    print(net.shape, "after step 7, should be 128 128 128")

    # deconvolution 3
    net = L.deconv2d(input_tensor=net, name='deconv_3', kernel=(4,4), stride=(2,2), output_num_filter=64)
    print(net.shape, "afterdeconv3, should be 256 256 64")

    # concatenate downsampling1 and upsampling1
    upsampling1 = net
    net = tf.concat([downsampling1, upsampling1], axis=3)

    # 2x resnet indexed 8: input is 256 x 256 x (64+64), note kernel size in this module is (5,5) not (3,3)
    net = double_resnet(input_tensor=net, filter_factor=0.5, dim_factor=1, kernel=(5,5), input_filters=128, num_classes=num_class, step=8)
    print(net.shape, "after step 8, should be 256 256 64")

    # deconvolution 4
    net = L.deconv2d(input_tensor=net, name='deconv_4', kernel=(4,4), stride=(2,2), output_num_filter=64)
    print(net.shape, "afterdeconv3, should be 512 512 64")

    # concatenate downsampling0 and upsampling0
    upsampling0 = net
    net = tf.concat([downsampling0, upsampling0], axis=3)

    # first of two  7x7 conv at the end
    net = L.conv2d(input_tensor=net, name='conv7x7_end1', kernel=(7,7), stride=(1,1), num_filter=64, activation_fn=tf.nn.relu)
    print(net.shape, "should be 512 512 64")

    # second of two 7x7 conv at the end
    net = L.conv2d(input_tensor=net, name='conv7x7_end2', kernel=(7,7), stride=(1,1), num_filter=3, activation_fn=tf.nn.relu)
    print(net.shape, "should be 512 512 3")

    return net

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,512,512,1])
    net = build(x)
    
