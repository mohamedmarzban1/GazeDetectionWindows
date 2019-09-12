# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:14:38 2018

@author: mfm160330
"""

import cv2

##===== Intialize the start and end minute/second to be extracted from the video ===# 
MinuteStart = 1
SecStart = 6

MinuteEnd = 1
SecEnd = 15

FR = 60
###########################################################




StartSec = MinuteStart*60 + SecStart
EndSec = MinuteEnd*60 + SecEnd
### Splitting a video using ffmpeg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("G:/Multi-sensors gaze Data Collection/CalibrationData2019-5-1/face/GH010170.MP4", StartSec, EndSec, targetname="C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/CalFace2019-5-1CalSmall.mp4")





### Splitting the video using openCV
#msStart = (MinuteStart*60 + SecStart)*1000
#FrameEnd = (MinuteEnd*60 + SecEnd) * FR
#cap = cv2.VideoCapture('GH010009.MP4')
#cap.set(cv2.CAP_PROP_POS_MSEC,msStart)

#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 


#ret, frame = cap.read()

#MyFourCC = cv2.VideoWriter_fourcc('X','V','I','D') 
#video_out = cv2.VideoWriter('RoadTraceAprilTagXVID.avi', MyFourCC, FR, (width,height), 1)

#ret, ret2 = True, True
#while(cap.isOpened() & ret & ret2):
#    CurrFrameNum= cap.get(cv2.CAP_PROP_POS_FRAMES) #current frame number
#    ret, frame = cap.read()
#    ret2 = bool(FrameEnd - CurrFrameNum)

#    if ret & ret2:
#        video_out.write(frame)

#cap.release()
#cv2.destroyAllWindows()
#print("Finished without Errors")