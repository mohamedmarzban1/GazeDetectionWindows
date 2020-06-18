# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:05:46 2020

@author: mfm160330
"""

import cv2
import pandas as pd
import os

## Extracts the face images from videos correponding to the labeled angles, crop these images and write them


def CreateImagesFolder(write_location):
    # create Dataset directory in OutputFiles
    try:
        os.makedirs(write_location)
    except: 
        print ("Creation of the directory %s failed" % write_location)
    

def getVideoOffsets(OffsetsTxtFile):
    with open(OffsetsTxtFile,"r") as file:
        for i in range(16):
            data = file.readline()

            if data.find('cont face frame number start')>=0:
                face_offset = int(data[data.find('=') + 1::]) # in frame numbers

    return face_offset

#######################################################
#### Change Before Every New Run #######################
######################################################
dataSets =  ["2019-05-22-001"] #["2019-06-21-001", "2019-07-10-001", "2019-10-30-001", "2019-07-19-001"]  #["2019-12-06-001", "2020-01-18-001", "2020-01-25-001", "2020-01-27-001", "2020-01-28-001", "2020-02-01-001", "2020-02-07-001", "2020-02-08-001", "2020-02-10-001", "2020-02-22-001", "2020-03-12-001"]  #["2019-11-14-001", "2019-11-19-002", "2019-11-20-001", "2019-11-22-001", "2019-11-25-001"]              #["2020-02-10-001", "2020-02-14-001", "2020-02-22-001", "2020-03-12-001", "2019-12-06-001"] # "2019-11-25-001" #"2019-11-14-001", "2019-11-22-001", 
DataSetIDs = ["C2019May22_001"] #["C2019June21_001Face", "C2019July10_001Face", "C2019Oct30_001Face", "2019July19_001"] #["C2020Feb10_001Face" "C2020Nov19_002Face", "C2019Nov20_001Face", "C2019Nov22_001Face", "C2019Nov25_001Face"] #["C2019Dec06_001", "C2020Jan18_001", "C2020Jan25_001", "C2020Jan27_001", "C2020Jan28_001", "C2020Feb01_001", "C2020Feb07_001", "C2020Feb08_001", "C2020Feb10_001", "C2020Feb22_001", "C2020March12_001"]  #["C2019Nov14_001Face" "C2019Nov19_002Face", "C2019Nov20_001Face", "C2019Nov22_001Face", "C2019Nov25_001Face"]                               # ["C2020Feb10_001Face", "C2020Feb14_001Face", "C2020Feb22_001Face", "C2020March12_001Face", "C2019Dec06_001Face"] #"C2019Nov25_001Face"  #"C2019Nov14_001Face", "C2019Nov22_001Face",
skipFrames = 5 #downsampleing by skipFrames + 1

######################################################################

for i0, dataset in enumerate(dataSets):
    print("processing dataset ", dataset)
    faceVideoPath = 'G':/Dataset/' + dataset +  '/ContGaze/FContGaze.mp4' #'/face.mp4' #
    contGazeOutputFilesLoc ='C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D'+dataset+'/ContGaze/'
    contGazeImagesLoc = 'D:/ContGazeImages/ContLabelledFace/'+dataset
    OffsetsTxtFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D"+dataset+"/StartAndEndFrameNums.txt"

    DataSetID = DataSetIDs[i0]
    startFaceFrame = 1 #getVideoOffsets(OffsetsTxtFile) # start frame num in the video (N.B. all previous frames should Not be counted)
    vCropS = 80
    vCropE = 1000
    hCropS = 500
    hCropE = 1550   
    fileType = '.jpg' #'.png'
    FR = 59.939
#######################################################

    IntialGazeFilePath = contGazeOutputFilesLoc + 'ContGazeIntialLabelsAllFrames_32.csv'  #input file
    IntermGazeFilePath = contGazeOutputFilesLoc + 'IntermContGazeAngles_32V2.csv' #outputFile

    CreateImagesFolder(contGazeImagesLoc)
    
    IntialLabelFile = pd.read_csv(IntialGazeFilePath, sep='\t')
    csv_length = len(IntialLabelFile) -1 
    print("ContGaze frame length is",csv_length)
    IntialLabelFile = IntialLabelFile.T
    
    video = cv2.VideoCapture(faceVideoPath)
    
    # skip all images before continuous part in face video
    #for i in range(startFaceFrame-1):
    #    _, _ = video.read()
    video.set(cv2.CAP_PROP_POS_FRAMES, startFaceFrame)
    success, image = video.read() #first useful image in the video
    #cv2.imwrite(contGazeImagesLoc+"/test2.jpg",image)

    csv_output = open(IntermGazeFilePath, 'w+')
    header = "DataSetID\tImageID\tRho\tElev\tAzim\tXcom\tYcom\tZcom\tBigTagX\tBigTagY\tBigTagZ\n"
    csv_output.write(header)
    
    cont_index = int(IntialLabelFile[0]['frame id']) # index of frame number in back camera #starts at zero
    UsefulCount = 0
    face_index = startFaceFrame #index of the faceframe
    
    

    while cont_index < csv_length:   

        lastFrameUsedFlag = 0
        if  str(IntialLabelFile[cont_index]['BigTagX']) != 'nan': 
            UsefulCount = UsefulCount + 1 
            row = IntialLabelFile[cont_index]
            imagecropped = image[vCropS:vCropE, hCropS:hCropE]
            imageNameInit = DataSetID+"_c%d_f%d" % (UsefulCount, face_index)
            cv2.imwrite(contGazeImagesLoc+'/'+imageNameInit+fileType, imagecropped )  # writes captured image
            csv_output.write(str(DataSetID) + '\t' + str(imageNameInit+fileType) + '\t' + str(row['r']) + '\t' + str(row['theta']) + '\t' + str(row['phi']) + '\t' + str(row['comX']) + '\t' + str(row['comY']) + '\t' + str(row['comZ']) + '\t' + str(row['BigTagX']) + '\t' + str(row['BigTagY']) + '\t' + str(row['BigTagZ']) + '\n')
            lastFrameUsedFlag = 1
        
        if lastFrameUsedFlag:
            cont_index = cont_index + 1 + skipFrames
            for i in range(skipFrames + 1):
                success, image = video.read()
                face_index += 1
        else:
            cont_index = cont_index + 1
            success, image = video.read()
            face_index += 1
            


print("finished processing all datasets ")



    

