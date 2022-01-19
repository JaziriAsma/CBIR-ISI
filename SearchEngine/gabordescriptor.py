"""
Created on Sun Dec 29 19:45:23 2019
@author: Dell
"""

import cv2
import numpy as np

class GaborDescriptor:
    
    def build_filters(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters
    
    def process(self,img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum
    
    def extract_features(self,img):
 
        filters=self.build_filters()
        res1=self.process(img,filters)

        return res1
       