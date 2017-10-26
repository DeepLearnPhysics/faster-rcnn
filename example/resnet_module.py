import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import uresnet_layers as L

def double_resnet(input_tensor, dim_factor=2, filter_factor=2, input_filters=64, num_classes=4, kernel=(3,3), step=1):

    # need to batch normalize 

    net = input_tensor
    
    # Module 1

    # 1st conv layer
    net = L.conv2d(input_tensor=net, name='conv1_1_%d' %step, kernel=kernel, stride=(dim_factor,dim_factor), 
                   num_filter=filter_factor*input_filters, activation_fn=tf.nn.relu)
    # 2nd conv layer
    net = L.conv2d(input_tensor=net, name='conv1_2_%d' %step, kernel=kernel, stride=(1,1), 
                   num_filter=filter_factor*input_filters, activation_fn=tf.nn.relu)

    # decrease spatial dimensions and increase channel number of mod1 output                                                                         
    mod1 = net
   # mod1 = L.avg_pool(input_tensor=mod1, name='decreaseSpatialD', kernel=(1,1), stride=(1,1))
    mod1 = L.conv2d(input_tensor=mod1, name='adjustDimensions_%d' %step,kernel=(1,1), stride=(1,1), 
                    num_filter=filter_factor*input_filters, activation_fn=tf.nn.relu)

    # Module 2

    # 3rd conv layer                                                                                                                                
    net = L.conv2d(input_tensor=net, name='conv2_1_%d' %step, kernel=kernel, stride=(1,1), 
                   num_filter=filter_factor*input_filters, activation_fn=tf.nn.relu)
    # 4th conv layer                                                                                                                                
    net = L.conv2d(input_tensor=net, name='conv2_2_%d' %step, kernel=kernel, stride=(1,1),
                   num_filter=filter_factor*input_filters, activation_fn=tf.nn.relu)

    # add the modified mod1 output to current tensor
    net += mod1

    return net

   # return L.final_inner_product(input_tensor=net, name='fc_final_%d' %step, num_output=num_classes)

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,28,28,1])
    net = double_resnet(x)

    
