#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import sys

caffe_root = os.path.dirname(os.path.realpath(__file__)) + "/../caffe_ssl/"
sys.path.insert(0, caffe_root + '/build/install/python')

import caffe
caffe.set_mode_gpu()
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
           'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
           'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
           'rightShoe']

import numpy as np
import cv_bridge
from sensor_msgs.msg import Image

MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../human/"
model_path = MODEL_PATH + "/config/attention/deploy.prototxt"
weight_path = MODEL_PATH + "/model/attention/attention+ssl.caffemodel"
label_colours = cv2.imread("human_parsing.png", 1).astype(np.uint8)

def segmentation_image(req):
    
