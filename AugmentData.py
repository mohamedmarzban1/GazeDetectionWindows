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



def AugmentData(row, num):
    ImageID = str(row['ImageID'])
    ImagePath = str(row['ImagePath'])
    Face_array = cv2.imread(os.path.join(ImagePath,'Face','F'+ImageID) )  # convert to array                
    Left_array = cv2.imread(os.path.join(ImagePath,'Leye','L'+ImageID) ) 
    Right_array = cv2.imread(os.path.join(ImagePath,'Reye','R'+ImageID) ) 
    
    row['ImagePath'] = AugmentedDataLoc
    
    #cv2.imwrite(AbsWriteLocSubF + "/F"+ImageID[:-4] + "Aug%03d.jpg" %  (count), faceArrayAug)
    #cv2.imwrite(AbsWriteLocSubL + "/L"+ImageID[:-4] + "Aug%03d.jpg" %  (count), LeftArrayAug)
    #cv2.imwrite(AbsWriteLocSubR + "/R"+ImageID[:-4] + "Aug%03d.jpg" %  (count), RightArrayAug)
    
    try:
        Face_HLS = cv2.cvtColor(Face_array,cv2.COLOR_RGB2HLS)  ## Convert Face RGB image to HLS 
        Left_HLS = cv2.cvtColor(Left_array,cv2.COLOR_RGB2HLS)  ## Convert left eye RGB image to HLS 
        Right_HLS = cv2.cvtColor(Right_array,cv2.COLOR_RGB2HLS)  ## Convert right eye RGBimage to HLS
    except: 
        zz =1  
        
        
    
    #Face_HLS = np.array(Face_HLS, dtype = np.float64)
    #Left_HLS = np.array(Left_HLS, dtype = np.float64)
    #Right_HLS = np.array(Right_HLS, dtype = np.float64)

    
    random_FaceCrop_coefficient = np.random.randint(low = 1, high = 5, size = num) ## generates random integer vales from 1 to 5
    random_EyeCrop_coefficient = np.random.randint(low = 1, high = 3, size = num) ## generates random integer vales from 1 to 3
    random_brightness_coefficient = np.random.uniform(low = 0.95, high = 1.05 , size = num)   ## generates value between 0.95 and 1.05
    
    count = 0
    for i1 in range(num):
        
        Face_HLS = Face_HLS[random_FaceCrop_coefficient[i1]:] #Face_array[vCropS:vCropE, hCropS:hCropE]
        Left_HLS = Left_HLS [random_EyeCrop_coefficient[i1]:] 
        Right_HLS = Right_HLS [random_EyeCrop_coefficient[i1]:]
        
        Face_HLS = np.array(Face_HLS, dtype = np.float64)
        Left_HLS = np.array(Left_HLS, dtype = np.float64)
        Right_HLS = np.array(Right_HLS, dtype = np.float64)

        Face_HLS[:,:,1] = Face_HLS[:,:,1]*random_brightness_coefficient[i1] ## scale pixel values up or down for channel 1(Lightness)
        Face_HLS[:,:,1][Face_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255    
        Face_HLS = np.array(Face_HLS, dtype = np.uint8)    
        Face_RGB = cv2.cvtColor(Face_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    
        Left_HLS[:,:,1] = Left_HLS[:,:,1]*random_brightness_coefficient[i1] ## scale pixel values up or down for channel 1(Lightness)
        Left_HLS[:,:,1][Left_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255    
        Left_HLS = np.array(Left_HLS, dtype = np.uint8)    
        Left_RGB = cv2.cvtColor(Left_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    
        Right_HLS[:,:,1] = Right_HLS[:,:,1]*random_brightness_coefficient[i1] ## scale pixel values up or down for channel 1(Lightness)
        Right_HLS[:,:,1][Right_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255    
        Right_HLS = np.array(Right_HLS, dtype = np.uint8)    
        Right_RGB = cv2.cvtColor(Right_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB   
        
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
InputFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNine.csv" ##input
NewAugmentedFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/AugmentedNine.csv" ##output
AugmentedDataLoc = "G:/AugmentedData/"
AugmentTimes = [13,6,2,0,0,0,0,0,0,1,2,7,11,19]
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
        

#=======  =====#
for indx in range(numRowsInput): 
    row = AllLabeledImagesFile[indx]
    with open(NewAugmentedFile, 'a+') as csv_output:
        filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
        filewriter.writerow(row)
    Elev = float(row['Elev'])
    Azim = float(row['Azim'])
    ElevClass = int(row['ElevClass'])
    num = AugmentTimes[ElevClass]
    if num > 0:
        AugmentData(row, num)
        
        
        
    
####=========================================================####

