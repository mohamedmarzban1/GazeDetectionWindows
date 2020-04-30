# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:18:04 2019
# Applies face and eyes detection to the labelled images and keeps only the succeeded detected face and eyes 
# And keeps count of number of detections and number of misdetections.

@author: mfm160330
"""


from skimage import io
import face_alignment
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import cv2
import glob # to read all files in a given directory
import ntpath 
import csv
ntpath.basename("a/b/c")
import os
import pandas as pd


def path_leaf(path): # extracts a file name from a given path 
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
##########################################################################
#============= intialize parameters =================================#
###########################################################################
ImageReadLocation = 'G:/ContGazeImages/ContLabelledFace/2019-11-20-001'#'C:/AdasData/Labelled Face/LF 2018-10-14'
InputIDfilePath = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D2019-11-20-001/ContGaze/IntermContGazeAngles.csv'
DataSet = 'CFE2019-11-20-001'
fileType = '.jpg'
vCroppedS = 80   # start vertical pixel after the image was already cropped
hCroppedS = 500 
############################################################################

WriteDir = 'G:/ContGazeImages/FaceAndEyes' #'C:/AdasData/FaceAndEyes'
WriteLocation = WriteDir +'/' + DataSet #'C:/AdasData/FaceAndEyes/FE2018-12-1'
FolderNames = ['Face', 'Leye', 'Reye']
is2D = 0 #1:2D 0:3D
FinalFormatIDFile =  'AnglesIDfileCurrBack.csv'
FinalIDFilePath = WriteLocation + '/' + FinalFormatIDFile

topMargin = 100 #Face Margins
bottomMargin = 10
HorzMargin = 10
EHorzMar = 15 #Eyes Margins
EVertMar = 15

#============= Run the 2D/3D face alignment ===============================#.
if is2D:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)
else:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0', flip_input=True)

#============ Loop over all the data of 1 subject =========================#

AbsWriteLocSubF =  WriteLocation + "/" + FolderNames[0]
AbsWriteLocSubL =  WriteLocation + "/" + FolderNames[1]
AbsWriteLocSubR =  WriteLocation + "/" + FolderNames[2]

for f in FolderNames:
    try:           
        os.makedirs(WriteLocation + "/" + f)
    except OSError:  
        print ("Creation of the directory %s failed" % WriteLocation)
        
countFailed = 0
AllLabeledImagesFile = pd.read_csv(InputIDfilePath, sep='\t')
AllLabeledImagesFile = AllLabeledImagesFile.T
numImages = AllLabeledImagesFile.shape[1]

# ===== Open Final format ID file and add a header ======#
csv_output = open(FinalIDFilePath, 'w+')
header = "DataSetID\tImageID\tRho\tElev\tAzim\tXcom\tYcom\tZcom\tXtarget\tYtarget\tZtarget\tFace_X1\tFace_Y1\tFace_X2\tFace_Y2\tLEye_X1\tLEye_Y1\tLEye_X2\tLEye_Y2\tREye_X1\tREye_Y1\tREye_X2\tREye_Y2\tlabels\n"
csv_output.write(header)


    
for indx in range(numImages): 
    row = AllLabeledImagesFile[indx]
    ReadImageName = str(row['ImageID'])
    filename = ImageReadLocation +"/" + str(row['ImageID'])
    frame = cv2.imread(filename)
    
    preds = fa.get_landmarks(frame)
    if preds is None:
        countFailed += 1
        file = open(WriteLocation+"/CFaceNotDetected.txt", "a+") 
        file.write("{}-     {} \n".format(countFailed,ReadImageName))
        file.close()
        continue
        

    MaxX, MaxY, MinX, MinY, diffX, diffY, faceArea = [], [], [], [], [], [], []
    for i1 in range(len(preds)):
        # Detect largest Face based on landmarks
        MaxX.append (int( np.min([np.max(preds[i1][:,0]) + HorzMargin, frame.shape[1]]) ))
        MaxY.append (int( np.min([np.max(preds[i1][:,1]) + bottomMargin, frame.shape[0]]) ))
        MinX.append (int( np.max([np.min(preds[i1][:,0]) - HorzMargin ,0]) ))
        MinY.append (int( np.max([np.min(preds[i1][:,1]) - topMargin ,0]) ))
        diffX.append(MaxX[i1] - MinX[i1])
        diffY.append(MaxY[i1] - MinY[i1])
        faceArea.append(diffX[i1]*diffY[i1])
    
    indxL = faceArea.index(max(faceArea)) # index of the largest face
    FaceLoc = [MinX[indxL]+hCroppedS, MinY[indxL]+vCroppedS, MaxX[indxL]+hCroppedS, MaxY[indxL]+vCroppedS]  #X_left, Y_Upper, X_Right, Y_lower  

    
    # Detect Left Eye based on landmarks
    Lpreds = preds[indxL][36:42,:]
    LMaxX = int( np.min([np.max(Lpreds[:,0]) + EHorzMar, frame.shape[1]]) )
    LMaxY = int( np.min([np.max(Lpreds[:,1]) + EVertMar, frame.shape[0]]) )
    LMinX = int( np.max([np.min(Lpreds[:,0]) - EHorzMar ,0]) )
    LMinY = int( np.max([np.min(Lpreds[:,1]) - EVertMar ,0]) )
    LMinXY = [LMinX, LMinY]
    
    LeftLoc = [LMinX+hCroppedS, LMinY+vCroppedS, LMaxX+hCroppedS, LMaxY+vCroppedS]  #X_left, Y_Upper, X_Right, Y_lower  

        
    # Detect right eye based on landmarks
    Rpreds = preds[indxL][42:48,:]
    RMaxX = int( np.min([np.max(Rpreds[:,0]) + EHorzMar, frame.shape[1]]) )
    RMaxY = int( np.min([np.max(Rpreds[:,1]) + EVertMar, frame.shape[0]]) )
    RMinX = int( np.max([np.min(Rpreds[:,0]) - EHorzMar ,0]) )
    RMinY = int( np.max([np.min(Rpreds[:,1]) - EVertMar ,0]) )
    #RMinXY = [RMinX, RMinY]
    RightLoc = [RMinX+hCroppedS, RMinY+vCroppedS, RMaxX+hCroppedS, RMaxY+vCroppedS]  #X_left, Y_Upper, X_Right, Y_lower  


    # Crop Face and Eyes
    FaceCropped = frame[MinY[indxL]:MaxY[indxL],MinX[indxL]:MaxX[indxL]]
    LCropped = frame[LMinY:LMaxY,LMinX:LMaxX]
    RCropped = frame[RMinY:RMaxY,RMinX:RMaxX]
        
    NewImageName = ReadImageName
    cv2.imwrite(AbsWriteLocSubF + "/F" + NewImageName, FaceCropped)
    cv2.imwrite(AbsWriteLocSubL + "/L" + NewImageName, LCropped)
    cv2.imwrite(AbsWriteLocSubR + "/R" + NewImageName, RCropped)
        
    # write IDs to a CSV file
    with open(FinalIDFilePath, 'a+') as csv_output:
        filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
        filewriter.writerow([str(row['DataSetID']), ReadImageName, str(row['Rho']), str(row['Elev']), str(row['Azim']), str(row['Xcom']), str(row['Ycom']), str(row['Zcom']), str(row['BigTagX']), str(row['BigTagY']), str(row['BigTagZ']), str(FaceLoc[0]), str(FaceLoc[1]), str(FaceLoc[2]), str(FaceLoc[3]), str(LeftLoc[0]), str(LeftLoc[1]), str(LeftLoc[2]), str(LeftLoc[3]), str(RightLoc[0]), str(RightLoc[1]), str(RightLoc[2]), str(RightLoc[3]), 'nan'])


