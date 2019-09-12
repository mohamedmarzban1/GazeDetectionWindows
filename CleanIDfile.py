# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 00:13:06 2019
@author: mfm160330

Go over all rows in a csv file, if the ID was not found as a file in the a specific folder, it will be ignored.
If the ID was found, the row will be added to a another CSV file.
"""
import os
import pandas as pd
import csv


#===== Intialize parameters ============#
InputIdFilePath = "G:/ContGazeImages/FaceAndEyes/CFE2019-7-19/CFE2019-7-19FinalFormatIDFile.csv"  #input
ImageReadLocation = "G:/ContGazeImages/FaceAndEyes/CFE2019-7-19/Face" #input
IDFileCleaned = "G:/ContGazeImages/FaceAndEyes/CFE2019-7-19/CFE2019-7-19IDFileCleaned.csv" #output


#====== Open the cleaned ID file =========#
csv_output = open(IDFileCleaned, 'w+')
header = "DataSetID\tImageID\tRho\tElev\tAzim\tXcom\tYcom\tZcom\tXtarget\tYtarget\tZtarget\tFace_X1\tFace_Y1\tFace_X2\tFace_Y2\tLEye_X1\tLEye_Y1\tLEye_X2\tLEye_Y2\tREye_X1\tREye_Y1\tREye_X2\tREye_Y2\tlabels\n"
csv_output.write(header)

# ========= Read the input ID file ==============#
AllLabeledImagesFile = pd.read_csv(InputIdFilePath, sep='\t')
AllLabeledImagesFile = AllLabeledImagesFile.T
numRowsInput = AllLabeledImagesFile.shape[1]


for indx in range(numRowsInput): 
    row = AllLabeledImagesFile[indx]
    ImageID = str(row['ImageID'])
    filename = ImageReadLocation +"/F" + ImageID
    FileExists = os.path.isfile(filename)
    if FileExists:
        with open(IDFileCleaned, 'a+') as csv_output:
            filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
            filewriter.writerow(row)




