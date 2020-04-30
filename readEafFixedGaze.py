# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:10:33 2020

@author: mfm160330
"""

import pympi
import cv2
import numpy as np
import os

#===============Get start and end frames of the continuous ==================#
def WriteStartEndFramesCont():
    start_cont = eafob.get_annotation_data_for_tier(tier_name_cont)[0][0] # start time of Cont. in ms
    end_cont = eafob.get_annotation_data_for_tier(tier_name_cont)[0][1] # end time of Cont. in ms
    for annotation in eafob.get_annotation_data_for_tier(tier_name_cont):
        if start_cont > annotation[0]:
            start_cont = annotation[0]
        if end_cont < annotation[1]:
            end_cont = annotation[1]
        
    start_cont_frame_num = (start_cont/1000)*60
    end_cont_frame_num = (end_cont/1000)*60
    print("start frame number cont = ",start_cont_frame_num)
    print("end frame number cont = ",end_cont_frame_num,"\n")
    numContGazeFrames = end_cont_frame_num - start_cont_frame_num
    print("===================================================================")

    #======= Back ================#
    back_start_cont = round(start_cont_frame_num + back_offset*60)
    back_end_cont = round(end_cont_frame_num + back_offset*60)
    print("back frame number start = " , back_start_cont)
    print("back frame number end = " , back_end_cont)

    #======= Road ================#
    road_start_cont = round(start_cont_frame_num + road_offset*60)
    road_end_cont = round(end_cont_frame_num + road_offset*60)
    print("road frame number start = " , road_start_cont)
    print("road frame number end = " , road_end_cont)

    #======= Face ================#
    face_start_cont = round(start_cont_frame_num + face_offset*60)
    face_end_cont = round(end_cont_frame_num + face_offset*60)
    print("face frame number start = " , face_start_cont)
    print("face frame number end = " , face_end_cont)
    print("Total number of frames = ", numContGazeFrames)
    
    #============= Write cont start and end frames ======================#

    #outputFile = open(text_output_file,"w")

    #outputFile.writelines("back offset in seconds = " + str(back_offset) +"\n") 
    #outputFile.writelines("road offset in seconds = " + str(road_offset) +"\n") 
    #outputFile.writelines("face offset in seconds = " + str(face_offset) +"\n") 

    #outputFile.writelines("start frame number cont = " + str(start_cont_frame_num) +"\n") 
    #outputFile.writelines("end frame number cont = " + str(end_cont_frame_num) +"\n")

    #outputFile.writelines("back frame number start = " + str(back_start_cont) +"\n")
    #outputFile.writelines("back frame number end = " + str(back_end_cont) +"\n") 

    #outputFile.writelines("road frame number start = " + str(road_start_cont) +"\n")
    #outputFile.writelines("road frame number end = " + str(road_end_cont) +"\n")

    #outputFile.writelines("face frame number start = " + str(face_start_cont) +"\n")
    #outputFile.writelines("face frame number end = " + str(face_end_cont) +"\n")
    #outputFile.writelines("Total number of frames = "+ str(numContGazeFrames) +"\n")

    #outputFile.close()
    return start_cont_frame_num, end_cont_frame_num



def createFolders(fixed_write_location,Categories_dic):
    allDicValues = Categories_dic.values()
    
    
    for subDir in allDicValues:
        try:
            WriteLocSub = fixed_write_location + subDir
            os.makedirs(WriteLocSub)
        except OSError:  
            print ("Creation of the directory %s failed" % WriteLocSub)

def Get_annotations_FrameNums(annotations_fixed, face_offset):
    annotations_fixed_frameNum = []
    for i in range(len(annotations_fixed)):
        annotations_fixed_frameNum.append((round(annotations_fixed[i][0]*60/1000 + face_offset*60),  round(annotations_fixed[i][1]*60/1000 + face_offset*60), annotations_fixed[i][2]))
    return annotations_fixed_frameNum
        

#============Intialize parameters ======================#
DataSet = '2019-11-20-001'
back_offset = 05.508
road_offset = 03.363
face_offset = 0

text_output_file_name = 'StartAndEndContFrameNums.txt'
output_file_path = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D'+DataSet+'/ContGaze/'
text_output_file = output_file_path + text_output_file_name
Eaf_file_path = 'G:/Multi-sensors gaze Data Collection/' + DataSet + '/annotations.eaf' 
tier_name_cont = 'continuous_gaze' 





eafob = pympi.Elan.Eaf(Eaf_file_path)

if tier_name_cont not in eafob.get_tier_names():
    print('WARNING!!!.. tier {}. is not present in elan file'.formart(tier_name_cont))

start_cont_frame_num, end_cont_frame_num = WriteStartEndFramesCont()

#============================================================================#

#====== fixed gaze file ===================
    
 
   
fixed_images_name= '2019-11-20_001'

FR = 60
cropFlag = 1 
vCropS = 80 
vCropE = 1000 
hCropS = 500
hCropE = 1550 
face_video = 'G:/Multi-sensors gaze Data Collection/'+DataSet+'/face.mp4'#'G:/Multi-sensors gaze Data Collection/TestDrive2018-12-03/Face/F4.MP4'
fixed_write_location = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/Labeled data Face/'+DataSet+'/'
tier_name_fixed = 'gaze_markers'

Categories_dic = {4:"a- 4", 1:"b- 1", 8:"c- 8", 2:"d- 2", 13:"e- 13", 5:"f- 5", 9:"g- 9", 11:"h- 11", 6:"i- 6", 20:"j- 20", 19:"k- 19", 18:"l- 18", 21:"m- 21", 17:"n- 17", 16:"o- 16", 14:"p- 14", 3:"q- 3", 7:"r- 7", 10:"s- 10", 12:"t- 12" ,15:"u- 15" } 

createFolders(fixed_write_location,Categories_dic)

# ========= Get start and end time of fixed gaze driving ======== #
# Note that the start time of fixed gaze driving has to be higher than end_cont
annotations_fixed = eafob.get_annotation_data_for_tier(tier_name_fixed) # start time of Cont. in ms
annotations_fixed_face_frameNum = Get_annotations_FrameNums(annotations_fixed, face_offset)



start_face_fixed = float('inf')  # start frame number in face video
end_face_fixed = annotations_fixed_face_frameNum[0][1]
for i, annotation in enumerate(annotations_fixed_face_frameNum):
    if annotation[0]>end_cont_frame_num and start_face_fixed > annotation[0]:
        start_face_fixed = annotation[0]
        start_fixed_index = i # for ordered annotations
    if annotation[1]>end_cont_frame_num and end_face_fixed < annotation[1]:
        end_face_fixed = annotation[1]

annotations_fixed_face_frameNum = annotations_fixed_face_frameNum[start_fixed_index::]


num_annot_fixed = len(annotations_fixed_face_frameNum)


##========Read Video =======##
cap = cv2.VideoCapture(face_video)
###===== test that it reads the first frame =====###
#ret1, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)

ret = True
frame_count = 0
j = 0
cap.set(cv2.CAP_PROP_POS_MSEC,annotations_fixed[43][0])
ret1, frame = cap.read()
cv2.imwrite(fixed_write_location+"/"+fixed_images_name+"-%d-test3333.jpg" % (start_face_fixed) , frame)


cap.set(cv2.CAP_PROP_POS_FRAMES, start_face_fixed)
ret1, frame = cap.read()
while frame is None:
    ret1, frame = cap.read()
cv2.imwrite(fixed_write_location+"/"+fixed_images_name+"-%d-test.jpg" % (start_face_fixed) , frame)

cap2 = cv2.VideoCapture(face_video)
while frame_count < start_face_fixed:
    _, frame = cap2.read()
    if frame is None:
        continue
    frame_count += 1
cv2.imwrite(fixed_write_location+"/"+fixed_images_name+"-%d-test22.jpg" % (start_face_fixed) , frame)



while(cap.isOpened() & ret):
#    while frame_count < start_face_fixed:
#        _, frame = cap.read()
#        if frame is None:
#            continue
#        frame_count += 1
    
    


    
    if j >= num_annot_fixed:
        # finished all fixed gaze annotations
        break
    
    if frame_count == (annotations_fixed_face_frameNum[j][0] + face_offset*60):
        # write all images till the end of the annotations range
        label = int(annotations_fixed[j][2])
        WriteLocSub = fixed_write_location + Categories_dic[label]
        while frame_count <= (annotations_fixed[j][1] + face_offset*60):
            _ , frame = cap.read()
            
            if cropFlag:
                frameCropped = frame[vCropS:vCropE, hCropS:hCropE]
            else:
                frameCropped = frame
            frame_count += 1
            
            secName = np.mod(np.floor(frame_count/60),60)
            minName = np.floor(frame_count/3600)
            SecFracName = np.mod(frame_count,60) #SecFracName is 1/60 (1 frame) seconds
            #cv2.imwrite(WriteLocation+"/"+ImagesName+"-%d-%d-%d-%d.jpg" % (count, minName, secName, SecFracName) , frame)
            cv2.imwrite(WriteLocSub+"/"+fixed_images_name+"-%d-%d-%d-%d.jpg" % (frame_count, minName, secName, SecFracName) , frameCropped)
        
        





