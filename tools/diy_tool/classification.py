#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : classification.py
## Authors    : zhluo@aries
## Create Time: 2017-03-23:14:22:22
## Description:
## 
##

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#display plots in this notebook
%matplotlib inline

#set display defaults
plt.rcParams['figure.figsize'] = (10, 10) #large images
plt.rcParams['image.interpolation'] = 'nearest' # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray' # use grayscale otput rather than a (potentially misleading)

# load caffe
import sys
caffe_root = '/home/zhluo/work/deep_compression/diy_caffe'
sys.path.insert(0, caffe_root + 'python')

import caffe

# load caffemodel
import os
caffe_model_root = '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
if os.path.isfile(caffe_root + caffe_model_root):
    print('CaffeNet found')
else:
    print('please download pre-trained caffemodel...\n eg: ./scripts/download_model_binary.py models/bvlc_reference_caffenet.caffemodel')

# load and set net 
caffe.set_mode_cpu()
prototxt = '/models/bvlc_reference_caffenet/deploy.prototxt'
model_def = caffe_root + prototxt
model_weights = caffe_root + caffe_model_root

net = caffe.Net(model_def, model_weights, caffe.TEST)

# load the mean imagenet image(as distributed with caffe) for subtraction
imagenet_mean = '/python/caffe/imagenet/ilsvrc_2012_mean.npy'
mu = np.load(caffe_root + imagenet_mean)
mu = mu.mean(1).mean(1) # average over pixels to obtain the mean(BGR) pixel values
print('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1)) # move image channels to outermost dimension 
transformer.set_mean('data', mu) # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255) # rescale form [0,1] to [0,255]
transformer.set_channel_swap('data', (2,1,0)) # swap channels from RGB to BGR

net.blobs['data'].reshape(50, 2, 227, 227)

image_root = '/examples/images/cat.jpg'
image = caffe.io.load_image(caffe_root + image_root)
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)

