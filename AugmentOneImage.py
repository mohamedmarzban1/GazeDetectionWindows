# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:26:07 2019

@author: mfm160330
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:47:48 2019

@author: mfm160330
"""
import os
import pandas as pd
import csv
import cv2
import numpy as np


### ==== A function that checks if a label is present in a list ==== ###
def MyListCheck (MyList, value):
    for x in MyList:
        if x == value:
            return True
    return False


def AugmentData(ImagePath, writeLoc, num, br_add, br_scale, sat_add, sat_scale):
    
    Face_array = cv2.imread(ImagePath)  # convert to array                


    Face_HSV = cv2.cvtColor(Face_array, cv2.COLOR_BGR2HSV)

    
    random_FaceCrop_coefficient = np.random.randint(low = 1, high = 5, size = num) ## generates random integer vales from 1 to 5
    br_add_rand = np.random.randint(low = -br_add, high = br_add, size = num) #random brightness additive
    br_scale_rand = np.random.uniform(low =1-br_scale, high=1+br_scale, size = num) #random brightness scale
    sat_add_rand = np.random.randint(low = -br_add, high = br_add, size = num) #random saturation additive
    sat_scale_rand = np.random.uniform(low =1-sat_scale, high=1+sat_scale, size = num) #random saturation scale
    count = 0
    maxLim = 255
    minLim = 0
    for i1 in range(num):
        
        # face augmentation
        Face_HSV = Face_HSV[random_FaceCrop_coefficient[i1]:] #Face_array[vCropS:vCropE, hCropS:hCropE]
        h, s, v = cv2.split(Face_HSV)
        v = np.clip((v * br_scale_rand[i1]) + br_add_rand[i1], minLim, maxLim, out = v)
        s = np.clip((s * sat_scale_rand[i1]) + sat_add_rand[i1], minLim, maxLim, out = s)        
        Face_hsv_editted = cv2.merge((h, s, v))
        Face_RGB = cv2.cvtColor(Face_hsv_editted, cv2.COLOR_HSV2BGR)

  
        
        ImageIDNew =  "Aug%03d.jpg" %  (count)

        cv2.imwrite(writeLoc +'/' +ImageIDNew, Face_RGB) 

        
        count = count + 1

    
    
    return 

###========== Intialize parameters ==============###
ImagePath = "G:/ContGazeImages/ContLabelledFace/2019-6-21/C2019June21Face_c32_f3391.jpg"
writeLoc = "C:/AdasData/Collection"
brightness_additive_max = 10
brightness_scale_max = 0.05
sat_add_max = 10
sat_scale_max = 0.05
#Sat_Max_val = 30


num = 1

AugmentData(ImagePath, writeLoc, num, brightness_additive_max, brightness_scale_max, sat_add_max, sat_scale_max)
        
        
        
    
####=========================================================####

