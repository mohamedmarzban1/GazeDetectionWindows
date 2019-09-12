"""
Created on Thu Nov  8 19:14:19 2018

@author: mfm160330
"""
import numpy as np
import cv2
import os


##===== Intialize the start and end minute/second to be extracted from the video ===# 
FirstFrameFlag = 1
MinuteStart = 0
SecStart = 0#41

MinuteEnd = 4
SecEnd = 55#24

#ReadVideoLocation = 'F:/TestDrive2Oct18/Face Camera'
VideoName = 'G:/Multi-sensors gaze Data Collection/Drive 2019-7-23/Face/GH030193.MP4'#'G:/Multi-sensors gaze Data Collection/TestDrive2018-12-03/Face/F4.MP4'
WriteLocation = 'G:/FixedGazeImages/Gaze points Data/G2019-7-23'
ImagesName= 'D2019-7-23'
FR = 60

#==== Intializing the crop values in HD===================#
cropFlag = 1 
vCropS = 80 #180 #(face14Oct18: 50, face1Dec: 100   mirror:150 ) #Vertical Crop Start
vCropE = 1000 #(face14Oct18: 1000,  mirror:950) #Vertical crop end
hCropS = 500 #(face14Oct18: 500,  mirror:800)
hCropE = 1550 #(face14Oct18: 1600, mirror:1650)


##========Read Video =======##
cap = cv2.VideoCapture(VideoName)
if FirstFrameFlag != 1:
    msStart = (MinuteStart*60 + SecStart)*1000
    cap.set(cv2.CAP_PROP_POS_MSEC,msStart)
FrameEnd = (MinuteEnd*60 + SecEnd) * FR

##===== Specifing start point using frame number ===# 
#frame_no = 57600
#cap.set(1,frame_no)

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
ret2 = True
while(cap.isOpened() & ret2):
    _, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if frame is None:
        count += 1
        continue
    
    if cropFlag:
        frameCropped = frame[vCropS:vCropE, hCropS:hCropE]
    else:
        frameCropped = frame
    
    #cv2.imshow('frame',frame) # PlZ comment me before running (To display the image)
    #cv2.imshow('frameCropped',frameCropped) # PlZ comment me before running (To display the image)
    #cv2.waitKey(0) # PlZ comment me before running 
    
    CurrFrameNum = cap.get(cv2.CAP_PROP_POS_FRAMES) #current frame number
    secName = np.mod(SecStart+np.floor(count/60),60)
    minName = MinuteStart + np.floor(SecStart/60 + count/3600)
    SecFracName = np.mod(count,60) #SecFracName is 1/60 (1 frame) seconds
    #cv2.imwrite(WriteLocation+"/"+ImagesName+"-%d-%d-%d-%d.jpg" % (count, minName, secName, SecFracName) , frame)
    cv2.imwrite(WriteLocation+"/"+ImagesName+"-%d-%d-%d-%d.jpg" % (count, minName, secName, SecFracName) , frameCropped)
    ret2 = bool(FrameEnd - CurrFrameNum)
    count+=1

cap.release()
cv2.destroyAllWindows()
print("Finished without Errors")

