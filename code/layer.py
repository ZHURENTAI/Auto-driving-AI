import tensorflow as tf
import numpy as np
import cv2 as cv
from scipy.misc import imread, imresize
from collections import namedtuple
sess = tf.Session()
is_training=False
true=False
conv2d=namedtuple("conv2d",["kernel","padding","stride","bias","activation","batch_normalization"])
separable_conv2d=namedtuple("separable_conv2d",["deep_kernel","point_kernel","padding","stride","activation","batch_normalization"])
pool=namedtuple("pool",["kernel","stride","Type","padding"])
dropout=namedtuple("dropout",["keep_prob"])
flatten=namedtuple("flatten",["nonen"])
full_connect=namedtuple("full_connect",["shape","ini_type"])
layers_def=[
  conv2d(kernel=[5,5,3,24],padding="VALID",stride=[1,2,2,1],bias=[24],activation=tf.nn.relu,batch_normalization=true),
  conv2d(kernel=[5,5,24,36],padding="VALID",stride=[1,2,2,1],bias=[36],activation=tf.nn.relu,batch_normalization=true),
  conv2d(kernel=[5,5,36,48],padding="VALID",stride=[1,2,2,1],bias=[48],activation=tf.nn.relu,batch_normalization=true),
  conv2d(kernel=[3,3,48,64],padding="VALID",stride=[1,1,1,1],bias=[64],activation=tf.nn.elu,batch_normalization=true),
  conv2d(kernel=[3,3,64,64],padding="VALID",stride=[1,1,1,1],bias=[64],activation=tf.nn.elu,batch_normalization=true),
  conv2d(kernel=[1,18,64,1164],padding="VALID",stride=[1,1,1,1],bias=[1164],activation=tf.nn.elu,batch_normalization=true),
  flatten(nonen=None),
  #dropout(keep_prob=1),
  full_connect(shape=100,ini_type=tf.random_normal),
  #dropout(keep_prob=0.5),
  full_connect(shape=50,ini_type=tf.random_normal),
  #dropout(keep_prob=0.5),
  full_connect(shape=10,ini_type=tf.random_normal),
  full_connect(shape=1,ini_type=tf.random_normal)
]
def nvidia_layer(x):
  with tf.name_scope("preprocess"):
    #net =tf.layers.batch_normalization(x,training=is_training) 
     net=x/127.5-1
  with tf.name_scope("layers"):
    for i,layer in enumerate(layers_def):
      if isinstance(layer,separable_conv2d):
        with tf.name_scope("separable_conv"+str(i)):
          weights = tf.Variable(tf.truncated_normal(layer.deep_kernel, stddev=0.1), name="deepwise_weights")
          bias = tf.Variable(tf.constant(0.0,shape=[net.shape[-1]]),name="deepwise_bias")
          net=tf.nn.depthwise_conv2d(net,weights,[1,1,1,1],layer.padding)  
          net=tf.nn.bias_add(net,bias)
          if(layer.batch_normalization):
             net=tf.layers.batch_normalization(net,training=is_training) 
          net=layer.activation(net)  
          weights = tf.Variable(tf.truncated_normal(layer.point_kernel, stddev=0.1), name="pointwise_weights")
          bias = tf.Variable(tf.constant(0.0,shape=[layer.point_kernel[-1]]),name="pointwise_bias")
          net=tf.nn.conv2d(net,weights,layer.stride,layer.padding)
          net=tf.nn.bias_add(net,bias)
          if(layer.batch_normalization):
             net=tf.layers.batch_normalization(net,training=is_training) 
          net=layer.activation(net)
          print net.get_shape()  
      if isinstance(layer,conv2d):
        with tf.name_scope("conv"+str(i)):
          bias = tf.Variable(tf.constant(0.0,shape=[layer.kernel[-1]]),name="bias")
          weights = tf.Variable(tf.truncated_normal(layer.kernel,stddev=0.1), name="weights")
          net=tf.nn.conv2d(net,weights,layer.stride,layer.padding)
          net=tf.nn.bias_add(net,bias)
          if(layer.batch_normalization):
             net=tf.layers.batch_normalization(net,training=is_training) 
          net=layer.activation(net)
          print net.get_shape()  
      if isinstance(layer,pool):
        with tf.name_scope("pool"+str(i)):
          net=layer.Type(net, layer.kernel, layer.stride, layer.padding,name="pool")
          print net.get_shape()
      if isinstance(layer,flatten):
          #net=tf.reshape(net, [-1,net.shape[1]*net.shape[2]*net.shape[3]])
        with tf.name_scope("flatten"+str(i)):
          net=tf.contrib.layers.flatten(net)
          print net.get_shape()
      if isinstance(layer,dropout):
        with tf.name_scope('dropout'+str(i)):
          if(is_training):
            net=tf.nn.dropout(net,layer.keep_prob)
      if isinstance(layer,full_connect):
        with tf.name_scope('fc'+str(i)):
          weights=tf.Variable(layer.ini_type([int(net.shape[-1]),layer.shape],stddev=0.1),name="weights")
          bias = tf.Variable(tf.constant(0.0,shape=[layer.shape]),name="bias")
          #net=tf.nn.bias_add(tf.matmul(net,weights),bias)
          net=tf.layers.dense(inputs=net,units=layer.shape,activation=None)
          print net.get_shape()
  return net 



