import numpy as np
import tensorflow.python.platform
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import batch_norm
#import tensorflow.layers as layers

def conv2d(input_tensor, name, num_filter, kernel, stride, activation_fn):

    num_input = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name, reuse=None):

        weights = tf.get_variable(name='%s_weights' % name,
                                  shape=[kernel[0],kernel[1],num_input,num_filter], 
                                  dtype=tf.float32, 
                                  initializer=xavier_initializer())

        biases  = tf.get_variable(name='%s_biases' % name,
                                  shape=[num_filter],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(input=input_tensor, 
                            filter=weights, 
                            strides=(1, stride[0], stride[1], 1),
                            padding='SAME')

        batchnorm = batch_norm(inputs=input_tensor, epsilon=0.00001)

        activation = activation_fn(tf.nn.bias_add(conv, biases))

        tf.summary.histogram('%s_weights' % name, weights)
        tf.summary.histogram('%s_biases' % name, biases)
        tf.summary.histogram('%s_activation' % name, activation)

    return activation

def deconv2d(input_tensor, name, output_num_filter, kernel, stride):
    
    num_input = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name, reuse=None):

        weights = tf.get_variable(name='%s_deconv_weights' %name,
                                  shape=[kernel[0], kernel[1], num_input, output_num_filter],
                                  dtype=tf.float32,
                                  initializer=xavier_initializer())

        biases = tf.get_variable(name='%s_deconv_biases' % name,
                                  shape=[output_num_filter],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        deconv = tf.layers.conv2d_transpose(inputs=input_tensor,
                                  filters=output_num_filter,
                                  kernel_size=kernel,
                                  strides=stride, 
                                  padding='same')

        tf.summary.histogram('%s_deconv_weights' % name, weights)
        tf.summary.histogram('%s_deconv_biases' % name, biases)

    return deconv

def max_pool(input_tensor,name,kernel,stride):

    with tf.variable_scope(name, reuse=None):

        pool = tf.nn.max_pool(value=input_tensor, 
                              ksize=[1, kernel[0], kernel[1], 1],
                              strides=[1, stride[0], stride[1], 1], 
                              padding='SAME')
    return pool

def avg_pool(input_tensor, name, kernel, stride):
    with tf.variable_scope(name):
        pool = tf.nn.avg_pool(value=input_tensor, 
                              ksize=[1,kernel[0],kernel[1],1],
                              strides=[1, stride[0], stride[1], 1],
                              padding='SAME')
    return pool
                              

def final_inner_product(input_tensor, name, num_output):

    shape=input_tensor.get_shape()
    input_tensor = tf.reshape(input_tensor, [-1, shape[-1].value * shape[-2].value * shape[-3].value])

    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable(name='%s_weights' % name,
                                  shape=[input_tensor.get_shape()[-1].value, num_output],
                                  dtype=tf.float32,
                                  initializer=xavier_initializer())
        biases  = tf.get_variable(name='%s_biases' % name,
                                  shape=[num_output],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))

        tf.summary.histogram('%s_weights' % name, weights)
        tf.summary.histogram('%s_biases' % name, biases)

    return tf.matmul(input_tensor,weights) + biases
    
