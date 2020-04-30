# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:33:34 2020

This is a test file to ensure we can read EAF files

@author: mfm160330
"""

import pympi

DataSet = '2019-11-20-001'
back_offset = 05.508
road_offset = 03.363
face_offset = 0

text_output_file_name = 'StartAndEndContFrameNums.txt'
output_file_path = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D'+DataSet+'/ContGaze/'
text_output_file = output_file_path + text_output_file_name
Eaf_file_path = 'G:/Multi-sensors gaze Data Collection/' + DataSet + '/annotations.eaf' 
tier_name = 'continuous_gaze' #tier_names = ['continuous_gaze', 'gaze_markers']

eafob = pympi.Elan.Eaf(Eaf_file_path)

if tier_name not in eafob.get_tier_names():
    print('WARNING!!!.. tier {}. is not present in elan file'.formart(tier_name))


start_cont = eafob.get_annotation_data_for_tier(tier_name)[0][0] # start time of Cont. in ms
end_cont = eafob.get_annotation_data_for_tier(tier_name)[0][1] # end time of Cont. in ms
for annotation in eafob.get_annotation_data_for_tier(tier_name):
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

#=============================#


outputFile = open(text_output_file,"w")

outputFile.writelines("back offset in seconds = " + str(back_offset) +"\n") 
outputFile.writelines("road offset in seconds = " + str(road_offset) +"\n") 
outputFile.writelines("face offset in seconds = " + str(face_offset) +"\n") 

outputFile.writelines("start frame number cont = " + str(start_cont_frame_num) +"\n") 
outputFile.writelines("end frame number cont = " + str(end_cont_frame_num) +"\n")

outputFile.writelines("back frame number start = " + str(back_start_cont) +"\n")
outputFile.writelines("back frame number end = " + str(back_end_cont) +"\n") 

outputFile.writelines("road frame number start = " + str(road_start_cont) +"\n")
outputFile.writelines("road frame number end = " + str(road_end_cont) +"\n")

outputFile.writelines("face frame number start = " + str(face_start_cont) +"\n")
outputFile.writelines("face frame number end = " + str(face_end_cont) +"\n")

outputFile.writelines("Total number of frames = "+ str(numContGazeFrames) +"\n")

outputFile.close()











