#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import rospy
from human_parsing.srv import HumanParsing
import cv2
import cv_bridge
import os

import sys
caffe_root = os.path.dirname(os.path.realpath(__file__)) + "/../caffe_ssl/"
sys.path.insert(0, caffe_root + '/build/install/python')

import caffe
GPU_ID = 0
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)


def human_parsing_client(img_msg):
    print ("temp")
    rospy.wait_for_service("/human_parsing")
    print ("temp2")

    try:
        human_parsing = rospy.ServiceProxy("human_parsing", HumanParsing)
        resp1 = human_parsing(img_msg)
        print ("Done")
        return resp1.segmentation_img
    except rospy.ServiceException, e:
        print ("Service call failed: %s" %e)

if __name__=="__main__":

    imgfile = os.path.dirname(os.path.realpath(__file__)) + "/../images/frame0001.jpg"
    img = cv2.imread(imgfile)
    br = cv_bridge.CvBridge()
    img_msg = br.cv2_to_imgmsg(img, encoding = "bgr8")

    segmentation_img_msg = human_parsing_client(img_msg)
    segmentation_img = br.imgmsg_to_cv2(segmentation_img_msg, desired_encoding = "bgr8")

    cv2.imwrite("segmentation_client.png", segmentation_img)
