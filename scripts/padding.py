#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import cv2
import os

def padding(img, alpha):
    h, w = img.shape[:2]
    base_size = h + alpha, w + alpha, 3
    base=np.zeros(base_size, dtype = np.uint8)
    cv2.rectangle(base, (0, 0), (w + alpha, h + alpha),(0, 0, 0))
    base[alpha : h + alpha, alpha : w + alpha] = img
    # cv2.imwrite("temp.jpg", base)
    return base
    
if __name__=="__main__":    
    imgfile = os.path.dirname(os.path.realpath(__file__)) + "/../images/frame0006.jpg"
    img = cv2.imread(imgfile)
    padding(img, 5)
    
