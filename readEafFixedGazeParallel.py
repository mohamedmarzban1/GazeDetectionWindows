# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:10:33 2020

@author: mfm160330
"""

import pympi
import cv2
import numpy as np
import os
#########################################################################################
#===============Get start and end frames of the continuous gaze ==================#
def WriteStartEndFramesCont():
    start_cont = eafob.get_annotation_data_for_tier(tier_name_cont)[0][0] # start time of Cont. in ms
    end_cont = eafob.get_annotation_data_for_tier(tier_name_cont)[0][1] # end time of Cont. in ms
    for annotation in eafob.get_annotation_data_for_tier(tier_name_cont):
        if start_cont > annotation[0]:
            start_cont = annotation[0]
        if end_cont < annotation[1]:
            end_cont = annotation[1]
        
    start_cont_frame_num = (start_cont/1000)*FR
    end_cont_frame_num = (end_cont/1000)*FR
    print("start frame number cont = ",start_cont_frame_num)
    print("end frame number cont = ",end_cont_frame_num,"\n")
    numContGazeFrames = end_cont_frame_num - start_cont_frame_num
    print("===================================================================")

    #======= Back ================#
    back_start_cont = round(start_cont_frame_num + back_offset*FR)
    back_end_cont = round(end_cont_frame_num + back_offset*FR)
    print("back frame number start = " , back_start_cont)
    print("back frame number end = " , back_end_cont)

    #======= Road ================#
    road_start_cont = round(start_cont_frame_num + road_offset*FR)
    road_end_cont = round(end_cont_frame_num + road_offset*FR)
    print("road frame number start = " , road_start_cont)
    print("road frame number end = " , road_end_cont)

    #======= Face ================#
    face_start_cont = round(start_cont_frame_num + face_offset*FR)
    face_end_cont = round(end_cont_frame_num + face_offset*FR)
    print("face frame number start = " , face_start_cont)
    print("face frame number end = " , face_end_cont)
    print("Total number of frames = ", numContGazeFrames)
    
    #============= Write cont start and end frames ======================#

    outputFile = open(text_output_file,"w")
    
    outputFile.writelines("Offsets (" + dataset + ") \n")
    outputFile.writelines("============== \n")

    outputFile.writelines("back offset in seconds = " + str(back_offset) +"\n") 
    outputFile.writelines("road offset in seconds = " + str(road_offset) +"\n") 
    outputFile.writelines("face offset in seconds = " + str(face_offset) +"\n") 
    
    outputFile.writelines("Continuous Gaze\n")
    outputFile.writelines("=================\n")

    outputFile.writelines("cont start frame number cont = " + str(start_cont_frame_num) +"\n") 
    outputFile.writelines("cont end frame number cont = " + str(end_cont_frame_num) +"\n")

    outputFile.writelines("cont back frame number start = " + str(back_start_cont) +"\n")
    outputFile.writelines("cont back frame number end = " + str(back_end_cont) +"\n") 

    outputFile.writelines("cont road frame number start = " + str(road_start_cont) +"\n")
    outputFile.writelines("cont road frame number end = " + str(road_end_cont) +"\n")

    outputFile.writelines("cont face frame number start = " + str(face_start_cont) +"\n")
    outputFile.writelines("cont face frame number end = " + str(face_end_cont) +"\n")
    outputFile.writelines("cont Total number of frames = "+ str(numContGazeFrames) +"\n")
    
    outputFile.writelines("====================================================\n")

    outputFile.close()
    return start_cont, end_cont, face_start_cont, face_end_cont
##########################################################################################
    
####################################################################################
# get start and end frames of the fixed gaze
def WriteStartEndFramesFixed():
    outputFile = open(text_output_file,"a")
    outputFile.writelines("Fixed Gaze\n")
    outputFile.writelines("=================\n")
    
    outputFile.writelines("fixed synchronized frame number start = " + str(annotations_fixed[start_fixed_index][0]*FR/1000) +"\n")
    outputFile.writelines("fixed synchronized frame number end = " + str(annotations_fixed[-1][1]*FR/1000) +"\n")
    
    outputFile.writelines("fixed face frame number start = " + str(annotations_fixed_frame_nums_driving[0][0]) +"\n")
    outputFile.writelines("fixed face frame number end = " + str(annotations_fixed_frame_nums_driving[-1][1]) +"\n")
    
    outputFile.writelines("fixed back frame number start = " + str(round(annotations_fixed[start_fixed_index][0]*FR/1000 + back_offset*FR)) +"\n")
    outputFile.writelines("fixed back frame number end = " + str(round(annotations_fixed[-1][1]*FR/1000 + back_offset*FR)) +"\n")
    
    outputFile.writelines("fixed back AprilTags frame number start = " + str(round(annotations_fixed[start_fixed_index][0]*FR/1000 + back_offset*FR)-AprilTags_extra_frames) +"\n")
    outputFile.writelines("fixed back AprilTags frame number end = " + str(round(annotations_fixed[-1][1]*FR/1000 + back_offset*FR) + AprilTags_extra_frames) +"\n")
    
    outputFile.close()


#############################################################################################################    
    
def CreateParentFolders(write_location):
    # create Dataset directory in OutputFiles
    try:
        os.makedirs(write_location)
    except: 
        print ("Creation of the directory %s failed" % write_location)
    
    ## create Fixed gaze directory
    #try:
    #    os.makedirs(write_location + "/FixedGaze")
    #except: 
    #    print ("Creation of the directory %s /FixedGaze failed" % write_location)    

    ## create cont gaze directory
    #try:
    #    os.makedirs(write_location + "/ContGaze")
    #except: 
    #    print ("Creation of the directory %s /ContGaze failed" % write_location) 
    
    
# afunction that creates directories for each label in fixed gaze and creates their parent directory 
def createFolders(fixed_write_location,Categories_dic):
    allDicValues = Categories_dic.values()  
    try:
        os.makedirs(fixed_write_location)
    except: 
        print ("Creation of the directory %s failed" % fixed_write_location)
    
    for subDir in allDicValues:
        try:
            WriteLocSub = fixed_write_location + '/' + subDir
            os.makedirs(WriteLocSub)
        except OSError:  
            print ("Creation of the directory %s failed" % WriteLocSub)

def Get_annotations_FrameNums(annotations_fixed, face_offset, FR):
    annotations_fixed_frameNum = []
    for i in range(len(annotations_fixed)):
        annotations_fixed_frameNum.append((round(annotations_fixed[i][0]*FR/1000 + face_offset*FR),  round(annotations_fixed[i][1]*FR/1000 + face_offset*FR), annotations_fixed[i][2]))
    return annotations_fixed_frameNum
        

def getVideoOffsets(csv_annotation_file):
    with open(csv_annotation_file,"r") as file:
        for i in range(7):
            data = file.readline()
            if data.find('back')>0:
                try:
                    back_offset = float(data[data.find('TIME_ORIGIN') + 13:data.find('/>')-2])/1000 #in sec
                except:
                    back_offset = 0
                    print("WARNING: back offset is ZERO")
            if data.find('road')>0:
                try:
                    road_offset = float(data[data.find('TIME_ORIGIN') + 13:data.find('/>')-2])/1000 # in sec
                except:
                    road_offset = 0
                    print("WARNING: road offset is ZERO")
            elif data.find('face')>0:
                try:
                    face_offset = float(data[data.find('TIME_ORIGIN') + 13:data.find('/>')-2])/1000 #in sec
                except:
                    face_offset = 0
                    print("WARNING: face offset is ZERO")

    return back_offset, road_offset, face_offset

#============ Intialize parameters ======================#
dataSets = ["2020-03-12-001"]#["2020-02-10-001", "2020-02-14-001", "2020-02-22-001", "2020-03-12-001"]
fixed_images_name_All = ["2020-03-12_001"] #["2020-02-10_001", "2020-02-14_001", "2020-02-22_001", "2020-03-12_001"]#["2020-01-18_001", "2020-01-24_001", "2020-01-25_001", "2020-01-27_001", "2020-01-28_001"]
writeFileOnly = 0 # if this flag is set, we won't generate new fixed gaze images

FR = 59.939
## For Fixed gaze
cropFlag = 1 
vCropS = 80 
vCropE = 1000 
hCropS = 500
hCropE = 1550
AprilTags_extra_frames = 100 # run more number of AprilTag frames as a safety margin
Categories_dic = {4:"a- 4", 1:"b- 1", 8:"c- 8", 2:"d- 2", 13:"e- 13", 5:"f- 5", 9:"g- 9", 11:"h- 11", 6:"i- 6", 20:"j- 20", 19:"k- 19", 18:"l- 18", 21:"m- 21", 17:"n- 17", 16:"o- 16", 14:"p- 14", 3:"q- 3", 7:"r- 7", 10:"s- 10", 12:"t- 12" ,15:"u- 15" } 

for i0, dataset in enumerate(dataSets):
    print("processing subject", dataset)
    fixed_images_name = fixed_images_name_All[i0]
    ## For both cont and fixed gaze
    text_output_file_name = 'StartAndEndFrameNums.txt'
    output_file_path = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D'+dataset 
    text_output_file = output_file_path + '/' + text_output_file_name
    Eaf_file_path = 'D:/Dataset/' + dataset + '/annotations.eaf' 
    csv_annotation_file = 'D:/Dataset/' + dataset + '/annotations.csv' 
    tier_name_cont = 'continuous_gaze' 
    ## For Fixed gaze
    face_video = 'D:/Dataset/'+dataset+'/face.mp4'
    fixed_write_location = 'D:/FixedGazeImages/Labeled data Face/'+dataset
    tier_name_fixed = 'gaze_markers'

    #==============================================================#
    eafob = pympi.Elan.Eaf(Eaf_file_path)

    if not os.path.exists(csv_annotation_file):
        eafob.to_file(csv_annotation_file)


    #======= Get offset times  =========#
    back_offset, road_offset, face_offset =  getVideoOffsets(csv_annotation_file)

    #======== write continuous gaze start and end times ========#
    # create Dataset directory in OutputFiles
    CreateParentFolders(output_file_path)

    ##########
    #e = eafob.get_linked_files()
    #j = eafob.get_full_time_interval()
    #k = eafob.get_cv_descriptions(1)

    if tier_name_cont not in eafob.get_tier_names():
        print('WARNING!!!.. tier {}. is not present in elan file'.formart(tier_name_cont))

    start_cont, end_cont, start_cont_frame_num, face_end_cont = WriteStartEndFramesCont()

    #====== Extarct fixed gaze frames ===================#
    createFolders(fixed_write_location,Categories_dic)

    # ========= Get start and end time of fixed gaze driving ======== #
    # Note that the start time of fixed gaze driving has to be higher than end_cont
    annotations_fixed = eafob.get_annotation_data_for_tier(tier_name_fixed) # start time of Cont. in ms
    annotations_fixed_frame_nums =  Get_annotations_FrameNums(annotations_fixed, face_offset, FR)



    start_face_fixed = float('inf')  # start frame # in face video
    end_face_fixed = annotations_fixed_frame_nums[0][1]
    for i, annotation in enumerate(annotations_fixed_frame_nums):
        if annotation[0]>face_end_cont and start_face_fixed > annotation[0]:
            start_face_fixed = annotation[0]
            start_fixed_index = i # for ordered annotations
        if annotation[1]>face_end_cont and end_face_fixed < annotation[1]:
            end_face_fixed = annotation[1]

    annotations_fixed_frame_nums_driving = annotations_fixed_frame_nums [start_fixed_index::]

    WriteStartEndFramesFixed()

    if not writeFileOnly:
        num_annot_driving = len(annotations_fixed_frame_nums_driving)

        ##========Read Video and write frames =======##
        cap = cv2.VideoCapture(face_video)
        #cap.set(cv2.CAP_PROP_POS_FRAMES, annotations_fixed_frame_nums_driving[0][0])
        #ret1, frame = cap.read()
        #cv2.imwrite(fixed_write_location + "/" + fixed_images_name + "__%dms.jpg" % (annotations_fixed_frame_nums_driving[0][0]) , frame)


        for i in range(num_annot_driving):
            cap.set(cv2.CAP_PROP_POS_FRAMES, annotations_fixed_frame_nums_driving[i][0])
            label =  int(annotations_fixed_frame_nums_driving[i][2])
            WriteLocSub = fixed_write_location + '/' + Categories_dic[label]
            currFrameNum = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
            while (currFrameNum <= annotations_fixed_frame_nums_driving[i][1]):
                ret1, frame = cap.read()
                while frame is None:
                    ret1, frame = cap.read()
            
                if cropFlag:
                    frameCropped = frame[vCropS:vCropE, hCropS:hCropE]
                else:
                    frameCropped = frame
            
                # N.B: the accurate number here is the currFrameNum, min, sec and secFrac are not accurate coz the FR should be 59.939
                currFrameNum = cap.get(cv2.CAP_PROP_POS_FRAMES)
                secName = np.mod(np.floor(currFrameNum/60),60)
                minName = np.floor(currFrameNum/3600)
                SecFracName = np.mod(currFrameNum,60) #SecFracName is 1/60 (1 frame) seconds
                cv2.imwrite(WriteLocSub + "/" + fixed_images_name +"-%d-%d-%d-%d.jpg" % (currFrameNum, minName, secName, SecFracName) , frameCropped)
        





