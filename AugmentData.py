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


def AugmentData(row, num, br_add, br_scale, sat_add, sat_scale):
    ImageID = str(row['ImageID'])
    ImagePath = str(row['ImagePath'])
    Face_array = cv2.imread(os.path.join(ImagePath,'Face','F'+ImageID) )  # convert to array                
    Left_array = cv2.imread(os.path.join(ImagePath,'Leye','L'+ImageID) ) 
    Right_array = cv2.imread(os.path.join(ImagePath,'Reye','R'+ImageID) ) 
    
    row['ImagePath'] = AugmentedDataLoc
    
    Face_HSV = cv2.cvtColor(Face_array, cv2.COLOR_BGR2HSV)
    Left_HSV = cv2.cvtColor(Left_array, cv2.COLOR_BGR2HSV)
    Right_HSV = cv2.cvtColor(Right_array, cv2.COLOR_BGR2HSV)
    
    random_FaceCrop_coefficient = np.random.randint(low = 1, high = 5, size = num) ## generates random integer vales from 1 to 5
    random_EyeCrop_coefficient = np.random.randint(low = 1, high = 2, size = num) ## generates random integer vales from 1 to 3
    #random_brightness_coefficient = np.random.uniform(low = 0.95, high = 1.05 , size = num)   ## generates value between 0.95 and 1.05
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
        
        #left eye augmentation
        Left_HSV = Left_HSV[random_EyeCrop_coefficient[i1]:] #Face_array[vCropS:vCropE, hCropS:hCropE]
        h_l, s_l, v_l = cv2.split(Left_HSV)        
        v_l = np.clip((v_l * br_scale_rand[i1])+br_add_rand[i1], minLim, maxLim, out=v_l)
        s_l = np.clip((s_l * sat_scale_rand[i1]) + sat_add_rand[i1], minLim, maxLim, out = s_l)
        Left_hsv_editted = cv2.merge((h_l, s_l, v_l))
        Left_RGB = cv2.cvtColor(Left_hsv_editted, cv2.COLOR_HSV2BGR)
        
        # right eye augmentation
        Right_HSV = Right_HSV[random_EyeCrop_coefficient[i1]:] #Face_array[vCropS:vCropE, hCropS:hCropE]
        h_r, s_r, v_r = cv2.split(Right_HSV)        
        v_r = np.clip((v_r * br_scale_rand[i1])+br_add_rand[i1], minLim, maxLim, out=v_r)
        s_r = np.clip((s_r * sat_scale_rand[i1]) + sat_add_rand[i1], minLim, maxLim, out = s_r)
        Right_hsv_editted = cv2.merge((h_r, s_r, v_r))
        Right_RGB = cv2.cvtColor(Right_hsv_editted, cv2.COLOR_HSV2BGR)        
        
        ImageIDNew = ImageID[:-4] + "Aug%03d.jpg" %  (count)
        row['ImageID'] = ImageIDNew

        cv2.imwrite(AbsWriteLocSubF + "/F" + ImageIDNew, Face_RGB) 
        cv2.imwrite(AbsWriteLocSubL + "/L" + ImageIDNew, Left_RGB) 
        cv2.imwrite(AbsWriteLocSubR + "/R" + ImageIDNew, Right_RGB) 
        
        count = count + 1
        with open(NewAugmentedFile, 'a+') as csv_output:
            filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
            filewriter.writerow(row)
    
    
    return 

###========== Intialize parameters ==============###
InputFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNineV3Exclude2Labels.csv" ##input
NewAugmentedFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/AugmentedNineV3.csv" ##output
AugmentedDataLoc = "G:/AugmnetedHSVv3/"
brightness_additive_max = 10
brightness_scale_max = 0.05
sat_add_max = 10
sat_scale_max = 0.05
#Sat_Max_val = 30
###==============================================###
FolderNames = ['Face', 'Leye', 'Reye']

# ========= Read the input ID file ==============#
AllLabeledImagesFile = pd.read_csv(InputFile, sep='\t')
AllLabeledImagesFile = AllLabeledImagesFile.T
numRowsInput = AllLabeledImagesFile.shape[1]
# ===============================================#

#========= Create Augmented File and write header ===========#
csv_output = open(NewAugmentedFile, 'w+')
header = "DataSetID\tImagePath\tImageID\tElevClass\tAzimClass\tElev\tAzim\n"
csv_output.write(header)

#======= Create face and Eyes folders for augmented data ======#
AbsWriteLocSubF =  AugmentedDataLoc + "/" + FolderNames[0]
AbsWriteLocSubL =  AugmentedDataLoc + "/" + FolderNames[1]
AbsWriteLocSubR =  AugmentedDataLoc + "/" + FolderNames[2]   
for f in FolderNames:
    try:           
        os.makedirs(AugmentedDataLoc + "/" + f)
    except OSError:  
        print("Creation of the directory %s failed" % AugmentedDataLoc)
        
num = 0
#=======  =====#
for indx in range(numRowsInput): 
    row = AllLabeledImagesFile[indx]
    with open(NewAugmentedFile, 'a+') as csv_output:
        filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
        filewriter.writerow(row)
    Elev = float(row['Elev'])
    Azim = float(row['Azim'])
    ElevClass = int(row['ElevClass'])
    AzimClass = int(row['AzimClass'])
    if ElevClass == 0:
        num = 8
    elif ElevClass == 1:
        num = 5
    elif ElevClass == 2 or ElevClass == 12:
        num = 3
    elif ElevClass == 11 or MyListCheck ([2,3,5,6,7,8,33,34,35,36,37], AzimClass):
        num = 1
    #num = AugmentTimes[ElevClass]
    
    if num > 0:
        AugmentData(row, num, brightness_additive_max, brightness_scale_max, sat_add_max, sat_scale_max)
        
        
        
    
####=========================================================####

