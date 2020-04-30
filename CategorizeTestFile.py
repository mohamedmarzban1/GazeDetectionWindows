# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:21:04 2019
This file is used to categorize a test file into 3 categories (1-contiuous Test, 2-discrete test, 3-ExcludedMarkersTest).
@author: mfm160330
"""
import pandas as pd
import csv
import numpy as np 

### ==== A function that checks if a label is present in a list ==== ###
def MyListCheck (MyList, label):
    for x in MyList:
        if x == label:
            return True
    return False

#======== Intialize parameters ===========#
FilesPath = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles'
TestIDfile = "ElevenAllTestX7.csv"
inputFile = FilesPath +'/'+ TestIDfile

OutputFixedNewMarkers = FilesPath  +'/'+ "ElevenFixedNewMarkersTestX7.csv"
OutputFixed =  FilesPath  +'/'+ "ElevenFixedTestX7.csv"
OutputCont = FilesPath  +'/'+ "ElevenContTestX7.csv"
ExcludeLabels = ["q- 3", "f- 5"] # exclude those markers from training


#====== Open 3 test files and write =========#
header = "DataSetID\tImagePath\tImageID\tElevClass\tAzimClass\tElev\tAzim\n"
# Flag 0
csv_outputFixedNewMarkers = open(OutputFixedNewMarkers, 'w+')
csv_outputFixedNewMarkers.write(header)

#Flag 1
csv_outputFixed = open(OutputFixed, 'w+')
csv_outputFixed.write(header)

#Flag 2
csv_outputCont = open(OutputCont, 'w+')
csv_outputCont.write(header)


#==== Read and concatenate all ID files =========#
dfAngles = pd.DataFrame()

dfInput = pd.read_csv(inputFile, sep='\t')
dfInput = dfInput.T
for j in range(dfInput.shape[1]):
    row = dfInput[j]
    DataSetID = str(row['DataSetID'])
    
    if DataSetID[0] == 'C':
        writeFlag = 2
    else:
        ImagePath = str(row['ImagePath'])
        label = ImagePath[-4:]
        if MyListCheck (ExcludeLabels, label):
            writeFlag = 0
        else:
            writeFlag = 1

    if writeFlag == 0:
        with open(OutputFixedNewMarkers, 'a+') as csv_output:
            filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
            filewriter.writerow(row)
    elif writeFlag == 1:
        with open(OutputFixed, 'a+') as csv_output2:
            filewriter2 = csv.writer(csv_output2, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
            filewriter2.writerow(row)             
    elif writeFlag == 2:
        with open(OutputCont, 'a+') as csv_output3:
            filewriter3 = csv.writer(csv_output3, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
            filewriter3.writerow(row)  

