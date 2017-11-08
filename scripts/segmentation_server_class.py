#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import sys
import cv_bridge
from sensor_msgs.msg import Image
import cv2
import rospy
from human_parsing.srv import *
from utils import padding, overlay

caffe_root = os.path.dirname(os.path.realpath(__file__)) + "/../caffe_ssl/"
sys.path.insert(0, caffe_root + '/build/install/python')

import caffe

sys.path.append('/usr/local/lib/python2.7/site-packages')

class SegmentationServerClass():
    
    def __init__(self, gpu_id = None):
        MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../human/"
        self.model_path = MODEL_PATH + "/config/attention/deploy.prototxt"
        self.weight_path = MODEL_PATH + "/model/attention/attention+ssl.caffemodel"
        self.label_colours = cv2.imread("human_parsing.png", 1).astype(np.uint8)
        self.br = cv_bridge.CvBridge()
        self.gpu_id = gpu_id
        self.mean = [104.008, 116.669, 122.675]
        self.net = None

        s = rospy.Service("human_parsing", HumanParsing, self.handle_human_parsing)
        rospy.spin()
    
    def handle_human_parsing(self, req):
        if self.net == None:
            try:
                self.net = caffe.Net(self.model_path, self.weight_path, caffe.TEST)
            except:
                rospy.logerr("Error, cannot load deep_net to the GPU")
                self.net =None
                self.service_queue -=1
                return HumanParsingResponse()
        
        try:
            img = self.br.imgmsg_to_cv2(req.rgb_img, desired_encoding = "bgr8")
            if self.gpu_id >= 0:
                caffe.set_mode_gpu()
                caffe.set_device(self.gpu_id)
            else:
                caffe.set_mode_cpu()
            
            img_padding = padding(img, 1)
            img_padding = img_padding.astype(np.float32)
            img_padding -= self.mean
            data = img_padding.transpose((2, 0, 1))
            self.net.blobs['data'].data[0, ...] = data
            out = self.net.forward()
            prediction = self.net.blobs['fc8_mask'].data[0, ...][0]
            prediction = prediction[1 : 481, 1 : 641]
            prediction = prediction.astype(np.int8)
            prediction = cv2.merge((prediction, prediction, prediction))
            prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
            label_colours_bgr = self.label_colours[..., ::-1]
            cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
            overlay_img = overlay(img, prediction_rgb)
            cv2.imwrite(os.path.dirname(os.path.realpath(__file__)) + "/../images/overlay.png", overlay_img)
            segmentation_img_msg = self.br.cv2_to_imgmsg(prediction_rgb, encoding = "bgr8")
            
            return HumanParsingResponse(segmentation_img = segmentation_img_msg)

        except cv_bridge.CvBridgeError as e:
            rospy.logerr("CVBridge exception %s",e)
            return HumanParsingResponse()
        
if __name__ == "__main__":
    rospy.init_node("human_parsing_server")
    SegmentationServerClass(gpu_id = 0)
