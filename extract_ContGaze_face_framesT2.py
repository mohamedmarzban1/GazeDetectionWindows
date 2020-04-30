# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:05:46 2020

@author: mfm160330
"""

import argparse
import cv2
import csv
import pandas as pd

## Extracts the face images from videos correponding to the labeled angles, crop these images and write them

#######################################################
#### Change Before Every New Run #######################
######################################################
faceVideoPath = 'G:/Multi-sensors gaze Data Collection/2019-11-20-001/face.mp4'
contGazeOutputFilesLoc ='C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D2019-11-20-001/ContGaze/'
contGazeImagesLoc = 'G:/ContGazeImages/ContLabelledFace/2019-11-20-001'
DataSetID = 'C2019Nov20Face'
startFaceFrame = 13680 # start sec in the video (N.B. all previous frames should Not be counted)
vCropS = 80
vCropE = 1000
hCropS = 500
hCropE = 1550
fileType = '.jpg' #'.png'
#######################################################

IntialGazeFilePath = contGazeOutputFilesLoc + 'ContGazeIntialLabelsAllFrames.csv'  #input file
IntermGazeFilePath = contGazeOutputFilesLoc + 'IntermContGazeAngles.csv' #outputFile

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("face_video_path",      help="Face Video Path", type=str)
    parser.add_argument("face_csv_data",        help="Data CSV File", type=str)
    parser.add_argument("face_video_offset",    help="Video sync offset back - face. This will be the same as the start frame of the back camera data stream.", type=int)
    parser.add_argument("road_video_offset",    help="Video sync offset back - road", type=int)

    return parser.parse_args()

def open_csv(face_csv_path):
    face_csv = open(face_csv_path, 'r')
    return csv.reader(face_csv, delimiter='\t')

def main():
    faceOffset = 0;
    roadOffset = 0;
    print_imgs(faceOffset, roadOffset)

def print_imgs(face_offset, road_offset):

    IntialLabelFile = pd.read_csv(IntialGazeFilePath, sep='\t')
    csv_length = len(IntialLabelFile)
    print("ContGaze frame length is",csv_length)
    IntialLabelFile = IntialLabelFile.T
    
    video = cv2.VideoCapture(faceVideoPath)
    
    # skip all images before continuous part in face video
    for i in range(startFaceFrame-1):
        _, _ = video.read()
    success, image = video.read() #first useful image in the video
    

    csv_output = open(IntermGazeFilePath, 'w+')
    header = "DataSetID\tImageID\tRho\tElev\tAzim\tXcom\tYcom\tZcom\tBigTagX\tBigTagY\tBigTagZ\n"
    csv_output.write(header)
    
    cont_index = IntialLabelFile[0]['frame id']
    UsefulCount = 0
    face_index = startFaceFrame
    while True:   

        if cont_index >= csv_length:
            print("Hit last frame in CSV")
            break

        cont_index = cont_index + 1
        if  str(IntialLabelFile[cont_index]['BigTagX']) != 'nan': 
            UsefulCount = UsefulCount + 1 
            row = IntialLabelFile[cont_index]
            imagecropped = image[vCropS:vCropE, hCropS:hCropE]
            imageNameInit = DataSetID+"_c%d_f%d" % (UsefulCount, face_index)
            cv2.imwrite(contGazeImagesLoc+'/'+imageNameInit+fileType, imagecropped )  # writes captured image
            csv_output.write(str(DataSetID) + '\t' + str(imageNameInit+fileType) + '\t' + str(row['r']) + '\t' + str(row['theta']) + '\t' + str(row['phi']) + '\t' + str(row['comX']) + '\t' + str(row['comY']) + '\t' + str(row['comZ']) + '\t' + str(row['BigTagX']) + '\t' + str(row['BigTagY']) + '\t' + str(row['BigTagZ']) + '\n')

        #cv2.imwrite(contGazeImagesLoc+'/'+DataSetID+"_c%d_f%d" % (UsefulCount, face_index), imagecropped )+fileType  # writes captured image 
        #csv_output.write(str(DataSetID) + '\t' + str(DataSetID+"_c%d_f%d"+fileType  % (UsefulCount, face_index)) + '\t' + str(row['r']) + '\t' + str(row['theta']) + '\t' + str(row['phi']) + '\t' + str(row['comX']) + '\t' + str(row['comY']) + '\t' + str(row['comZ']) + '\t' + str(row['BigTagX']) + '\t' + str(row['BigTagY']) + '\t' + str(row['BigTagZ']) + '\n')

        #   print("Printing frame %d (road) | %d (back) | %d (IntialLabelFile)" % (road_index, back_index, face_index))
        #   print('Combined data line:', IntialLabelFile[face_data_index])

        success,image = video.read()
        face_index += 1


if __name__ == '__main__':
    main()