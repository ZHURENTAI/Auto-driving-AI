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
base_dir='/home/tiger/deeptesla/'
f=file('test.txt','a+')
csv_dir=base_dir+'epoch10_steering.csv'
img_dir=base_dir+'epoch10/'
csv_reader=csv.reader(open(csv_dir))
wheels=[]
cap=cv2.VideoCapture('/home/tiger/deeptesla/epoch10_front.mkv')


def pre_process(img):
  crop_img=img[150:572,0:1280]
  im_resize=cv2.resize(crop_img,(200,66),interpolation=cv2.INTER_CUBIC)
  img_flip=cv2.flip(im_resize, 1)
  img_bright = cv2.cvtColor(im_resize, cv2.COLOR_BGR2HSV)
  brightness = random.randint(300,999)*0.001
  img_bright[:, :, 2] = img_bright[:, :, 2] * brightness
  img_bright=cv2.cvtColor(img_bright, cv2.COLOR_HSV2RGB)
  return im_resize,img_flip,img_bright

for row in csv_reader:
  if(row[2] !='wheel'):
    wheels.append(row[2])


x=tf.placeholder(tf.float32,[None,66,200,3])
y_=tf.placeholder(tf.float32,[None,1])
y=nvidia_layer(x)
loss=tf.reduce_mean(tf.sqrt(tf.square(y_-y)))
'''-----------add bn key to graph-------------'''
saver = tf.train.Saver()
with tf.Session() as sess:
 sess.run(tf.global_variables_initializer())
 saver.restore(sess, tf.train.latest_checkpoint('.'))
 pixs=np.zeros(shape=[1,66,200,3])
 labels=np.zeros(shape=[1,1])
 for k in range(0, len(wheels)):      
     ret,frame=cap.read()
     if(frame.shape==(720,1280,3)):
        frame=frame.astype(np.float32)
        human_in=float(wheels[k])
        resize_img,flip_img,bright_img=pre_process(frame)
        pixs[0,:,:,:]=resize_img
        if(wheels[k]>=0):
          labels[0,:]=1
        if(wheels[k]<0):
          labels[0,:]=0
        Y_=y.eval(feed_dict={x:pixs,y_:labels}) 
        Loss=loss.eval(feed_dict={x:pixs,y_:labels}) 
        print('loss=',Loss) 
        if(Y_>=0):
           f.write(str(1)+'\n') 
        if(Y_<0):
           f.write(str(0)+'\n') 

