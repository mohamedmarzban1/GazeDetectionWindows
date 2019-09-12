# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 22:21:20 2019

@author: mfm160330
"""
#from skimage import io
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


def path_leaf(path): # extracts a file name from a given path 
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

#============= intialize parameters =================================#
isAllF = 1 #1:All files in dirc. 0: single file 
ReadLocation = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/Labeled data Face/LF 2019-7-15'
WriteDir = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/FaceAndEyes'
DataSet = 'FE2019-7-15'
vCroppedS = 180
hCroppedS = 500
fileType = '.jpg'
#=====================================================================#

WriteLocation = WriteDir +'/' + DataSet #'C:/AdasData/FaceAndEyes/FE2018-12-1'
FolderNames = ['Face', 'Leye', 'Reye']
is2D = 0 #1:2D 0:3D
idFileName = 'id.csv'


topMargin = 100 #Face Margins
bottomMargin = 10
HorzMargin = 10
EHorzMar = 15 #Eyes Margins
EVertMar = 15

#============= Run the 2D/3D face alignment ===============================#
if is2D:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)
else:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0', flip_input=True)

#============ Loop over all the data of 1 subject =========================#
subDirs = next(os.walk(ReadLocation))[1]

csv_header = "DataSetID, ImageID, Label, Face_X1, Face_Y1, Face_X2, Face_Y2, LEye_X1, LEye_Y1, LEye_X2, LEye_Y2, REye_X1, REye_Y1, REye_X2, REye_Y2"
csvfile = open (WriteLocation+'/'+idFileName, 'w+') 
csvfile.write(csv_header+"\n")

for subDir in subDirs:
    ReadLocSub = ReadLocation + "/" + subDir
    WriteLocSub = WriteLocation + "/" + subDir
    AbsWriteLocSubF =  WriteLocSub + "/" + FolderNames[0]
    AbsWriteLocSubL =  WriteLocSub + "/" + FolderNames[1]
    AbsWriteLocSubR =  WriteLocSub + "/" + FolderNames[2]
    
    for f in FolderNames:
        try:           
            os.makedirs(WriteLocSub + "/" + f)
        except OSError:  
            print ("Creation of the directory %s failed" % WriteLocSub)
            
    count = 0
    countFailed = 0
    for filename in glob.glob((ReadLocSub+ '/*' + fileType)): #assuming jpg
        count += 1
        ReadImageName = path_leaf(filename)  
        frame = cv2.imread(filename)
        preds = fa.get_landmarks(frame)
        if preds is None:
            countFailed += 1
            file = open(WriteLocation+"/FaceNotDetected.txt", "a+") 
            file.write("{}- SubDir = {}      {} \n".format(countFailed,subDir,ReadImageName))
            file.close()
            continue
        
        # Detect largest Face based on landmarks
        MaxX, MaxY, MinX, MinY, diffX, diffY, faceArea = [], [], [], [], [], [], []
        for i1 in range(len(preds)):
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
        RightLoc = [RMinX+hCroppedS, RMinY+vCroppedS, RMaxX+hCroppedS, RMaxY+vCroppedS]  #X_left, Y_Upper, X_Right, Y_lowe

        # Crop Face and Eyes
        FaceCropped = frame[MinY[indxL]:MaxY[indxL],MinX[indxL]:MaxX[indxL]]
        LCropped = frame[LMinY:LMaxY,LMinX:LMaxX]
        RCropped = frame[RMinY:RMaxY,RMinX:RMaxX]
        
        NewImageName = ReadImageName[:-4]+"U%03d.jpg" %  (count)
        cv2.imwrite(AbsWriteLocSubF+"/F"+NewImageName, FaceCropped)
        cv2.imwrite(AbsWriteLocSubL+"/L"+NewImageName, LCropped)
        cv2.imwrite(AbsWriteLocSubR+"/R"+NewImageName, RCropped)
        
        # write IDs to a CSV file
        with open(WriteLocation+'/'+idFileName, 'a+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
            filewriter.writerow([DataSet, NewImageName, subDir, str(FaceLoc[0]), str(FaceLoc[1]), str(FaceLoc[2]), str(FaceLoc[3]), str(LeftLoc[0]), str(LeftLoc[1]), str(LeftLoc[2]), str(LeftLoc[3]), str(RightLoc[0]), str(RightLoc[1]), str(RightLoc[2]), str(RightLoc[3])])



