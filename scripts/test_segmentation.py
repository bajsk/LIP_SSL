#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import sys

caffe_root = os.path.dirname(os.path.realpath(__file__)) + "/../caffe_ssl/"
sys.path.insert(0, caffe_root + '/build/install/python')

import caffe
caffe.set_mode_cpu()
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from utils import padding

classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
           'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
           'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
           'rightShoe']

if __name__ == '__main__':

    MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../human/"
    model_path = MODEL_PATH + "/config/attention/deploy.prototxt"
    weight_path = MODEL_PATH + "/model/attention/attention+ssl.caffemodel"
    label_colours = cv2.imread("human_parsing.png", 1).astype(np.uint8)
    
    mean = [104.008, 116.669, 122.675]
    
    net = caffe.Net(model_path, weight_path, caffe.TEST)
    
    input_shape = net.blobs['data'].data.shape

    output_shape = net.blobs['fc8_interp'].data.shape

    imgfile = os.path.dirname(os.path.realpath(__file__)) + "/../images/frame0006.jpg"
    img = cv2.imread(imgfile, 1)
    img = padding(img, 1)
    img = img.astype(np.float32)
    img -= mean
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    data = img.transpose((2, 0, 1))
    net.blobs['data'].data[0, ...] = data
    out = net.forward()

    prediction = net.blobs['fc8_mask'].data[0, ...][0]
    prediction = prediction[1 : 481, 1 : 641]
    prediction = prediction.astype(np.int8)
    prediction = cv2.merge((prediction, prediction, prediction))

    prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)

    label_colours_bgr = label_colours[..., ::-1]
    cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
    
    cv2.imwrite("prediction.png", prediction_rgb)        
