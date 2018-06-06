import csv
from layer import nvidia_layer
import tensorflow as tf
import numpy as np
import random
import cv2 
from scipy.misc import imread, imresize
import os
import gc 
import pandas as pd
import matplotlib.pyplot as plt

'''---------------parameters------------------'''
base_dir='/home/tiger/deeptesla/'
epochs=9
min_batch=20#20*3=60
l2_para=0.000001
f=file('log.txt','a+')


def pre_process(img):
  crop_img=img[150:572,0:1280]
  im_resize=cv2.resize(crop_img,(200,66),interpolation=cv2.INTER_CUBIC)
  img_flip=cv2.flip(im_resize, 1)
  img_bright = cv2.cvtColor(im_resize, cv2.COLOR_BGR2HSV)
  brightness = random.randint(300,999)*0.001
  img_bright[:, :, 2] = img_bright[:, :, 2] * brightness
  img_bright=cv2.cvtColor(img_bright, cv2.COLOR_HSV2RGB)
  return im_resize,img_flip,img_bright




#plt.imshow(img_bright)
#plt.show()
frames=[]
wheels=[]
for i in range(epochs):
  csv_dir=base_dir+'epoch0'+str(i+1)+'_steering.csv'
  img_dir=base_dir+'epoch0'+str(i+1)+'/'
  csv_reader=csv.reader(open(csv_dir))
  for row in csv_reader:
    if(row[2] !='wheel'):
      img_path=img_dir+str(int(row[1])+1)+'.jpg'
      frames.append(img_path)
      wheels.append(row[2])
'''----------shuffle the data-----------------'''
c=list(zip(frames,wheels))
random.shuffle(c)
frames[:],wheels[:]=zip(*c)
del c
gc.collect()
'''-----------build the graph-----------------'''
x=tf.placeholder(tf.float32,[None,66,200,3])
y_=tf.placeholder(tf.float32,[None,1])
y=nvidia_layer(x)
with tf.variable_scope("mseloss"):
   mse=tf.reduce_mean(tf.square(y_-y))
with tf.variable_scope("l2loss"):
   l2=tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
with tf.variable_scope("loss"):
   loss=mse+l2_para*l2
'''-----------add bn key to graph-------------'''
update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
global_step = tf.get_variable('global_step', [], dtype=tf.int32,initializer=tf.constant_initializer(0), trainable=False)
with tf.control_dependencies(update_ops):
  train_step=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
var_list = tf.trainable_variables()
if global_step is not None:
    var_list.append(global_step)
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list)
'''----------start train-----------------------'''
with tf.Session() as sess:
 sess.run(tf.global_variables_initializer())
  #saver.restore(sess, tf.train.latest_checkpoint('.'))
 sess.run(tf.global_variables_initializer())
 sess.run(tf.local_variables_initializer())
 for j in range(5):
   pixs=np.zeros(shape=[min_batch*3,66,200,3])
   labels=np.zeros(shape=[min_batch*3,1])
   flag=0;
   for k in range(0, len(wheels)):      
     frame=cv2.imread(frames[k])
     if(frame.shape==(720,1280,3)):
        flag+=1
        frame=frame.astype(np.float32)
        resize_img,flip_img,bright_img=pre_process(frame)
        pixs[flag*3,:,:,:]=resize_img
        pixs[flag*3+1,:,:,:]=bright_img
        pixs[flag*3+2,:,:,:]=flip_img
        human_in=float(wheels[k])
        labels[flag*3,:]=human_in
        labels[flag*3+1,:]=human_in
        labels[flag*3+2,:]=0-human_in
        if(flag==min_batch-1):
          flag=0
          #labels=labels.astype(np.float32)
          sess.run(train_step,feed_dict={x:pixs,y_:labels})
          Loss=loss.eval(feed_dict={x:pixs,y_:labels}) 
          Mse=mse.eval(feed_dict={x:pixs,y_:labels}) 
          stri='loss='+str(Loss)+'   '+'mse='+str(Mse)
          print(stri) 
          f.write('loss='+str(Loss)+'\n') 
 saver.save(sess,"nvidia_end_end")
































