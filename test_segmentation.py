import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys

caffe_root = os.getcwd() + "/code/"
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_cpu()
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
           'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
           'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
           'rightShoe']

if __name__ == '__main__':

    model_path = "/home/btran/publicWorkspace/dev/LIP_SSL/human/config/attention/deploy.prototxt"
    weight_path = "/home/btran/publicWorkspace/dev/LIP_SSL/human/model/attention/attention+ssl.caffemodel"

    mean = [104.008, 116.669, 122.675]
    
    net = caffe.Net(model_path, weight_path, caffe.TEST)
    
    input_shape = net.blobs['data'].data.shape

    output_shape = net.blobs['fc8_interp'].data.shape

    label_colours = cv2.imread("attention.png").astype(np.uint8)
    
    imgfile = './temp.png'
    img = cv2.imread(imgfile, 1)
    img = img.astype(np.float32)
    img -= mean
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    data = img.transpose((2, 0, 1))
    net.blobs['data'].data[0, ...] = data
    out = net.forward()

    prediction = net.blobs['fc8_mask'].data[0, ...][0]
    cv2.imwrite("prediction.png", prediction)    
    