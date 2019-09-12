# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:32:14 2019

@author: mfm160330
"""


import numpy as np
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

face_cascade = cv2.CascadeClassifier('C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS/Spyder files/opencv LBP cascade Profile Faces/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS/Spyder files/opencv LBP cascade Profile Faces/data/haarcascades_cuda/haarcascade_eye.xml')
isAllF = 1 #1:All files in dirc. 0: single file 
ReadLocation = 'G:/NaturaliticPics/N2019-6-14' #'G:/NaturaliticPics/TestNoFaces'
WriteLocation = 'G:/NaturaliticPics/N2019-6-14Viola' #'G:/NaturaliticPics/TestNoFaces'#
FolderNames = ['Face', 'Leye', 'Reye']
fileType = '.jpg'
topMargin = 100 #Face Margins
bottomMargin = 10
HorzMargin = 10
EHorzMar = 15 #Eyes Margins
EVertMar = 15

#============ Write Locations =========================#
AbsWriteLocSubF =  WriteLocation + "/" + FolderNames[0]
AbsWriteLocSubL =  WriteLocation + "/" + FolderNames[1]
AbsWriteLocSubR =  WriteLocation + "/" + FolderNames[2]
#=========== Create 3 folders (Face, LEye, REye) ==========================#
for f in FolderNames:
    try:           
        os.makedirs(WriteLocation + "/" + f)
    except OSError:  
        print ("Creation of the directory %s failed" % f)
#==========================================================================#

#============== Detect largert face and eyes ======================#
count = 0
countFailed = 0
for filename in glob.glob((ReadLocation+ '/*' + fileType)): 
    count += 1
    ReadImageName = path_leaf(filename)  
    frame = cv2.imread(filename)
    
    # detect faces 
    faces = face_cascade.detectMultiScale(frame, 1.3, 5) ### check the inputs of this fn
    try:
        a = faces[0]
    except IndexError:
        print("Face Not detected \n")
        countFailed += 1
        file = open(WriteLocation+"/FaceNotDetected.txt", "a+") 
        file.write("{}- {} \n".format(countFailed,ReadImageName))
        file.close()
        continue
    
    x_a ,y_a, w_a, h_a, faceArea_a = [], [], [], [], []
    for (x,y,w,h) in faces:
        #roi_color = img[y:y+h, x:x+w]
        x_a.append(x)
        y_a.append(y)
        w_a.append(w)
        h_a.append(h)
        faceArea = w*h
        faceArea_a.append(faceArea)
    
    indxL = faceArea_a.index(max(faceArea_a)) # index of the largest face
    FaceCropped = frame[y_a[indxL]:y_a[indxL]+h_a[indxL],x_a[indxL]:x_a[indxL]+w_a[indxL]]
    cv2.imwrite(AbsWriteLocSubF+"/F"+ReadImageName[:-4]+"U%03d.jpg" %  (count), FaceCropped)
