# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:50:57 2019

@author: mfm160330

1- Combine more than 1 ID file
2- Convert Angles to labels

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


#===== Intialize parameters ============#
### Fixed Gaze:
ReadLocation1 = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/FaceAndEyes" #input
Sub1 =  ["FE2019-7-10"]#["FE2019-7-10", "FE2019-7-11"] #["FE2018-12-1", "FE2018-12-3", "FE2019-5-22", "FE2019-5-30", "FE2019-6-11", "FE2019-6-14", "FE2019-7-9", "FE2019-7-15" , "FE2019-7-23"]  #["FE2019-7-10", "FE2019-7-11"] # ##input

###Cont Gaze:
ReadLocation2 = "G:/ContGazeImages/FaceAndEyes" #input
Sub2 =  ["CFE2019-7-10"]#["CFE2019-7-10", "CFE2019-7-11"] #["CFE2019-5-22", "CFE2019-5-30", "CFE2019-6-11", "CFE2019-6-14", "CFE2019-6-21", "CFE2019-7-9", "CFE2019-7-15" , "CFE2019-7-19" , "CFE2019-7-23"] 
idFile = "AnglesIDfile.csv"

DenseClassificationFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNineValidV3.csv" #"C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNine.csv" #output
#validAndTestFlag = 1
#ValidFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNineValidV3.csv"

#===============================================#

ContDownSample = 4 #downsample contgaze images by 4
ExcludeLabels = []#["q- 3", "f- 5"] # exclude those markers from training

# Dense classificiation Parameters:
# Make sure (ElevEnd -Elevstart)/res is an integar 
ElevStart = 76 # in degrees
ElevEnd = 100 #in degrees
AzimStart = -46 # in degrees
AzimEnd = 26 # in degrees
res = 2 #Resolution of Elevation and Azimuth Angles classes in degrees
#===================================#

numElevClasses = (ElevEnd - ElevStart)/res + 2
numAzimClasses = (AzimEnd - AzimStart)/res + 2

#====== Open a new Dense Classification file =========#
csv_output = open(DenseClassificationFile, 'w+')
header = "DataSetID\tImagePath\tImageID\tElevClass\tAzimClass\tElev\tAzim\n"
csv_output.write(header)

#if validAndTestFlag:
#    csv_output2 = open(ValidFile, 'w+')
#    csv_output2.write(header)


#==== Read and concatenate all ID files =========#
dfAngles = pd.DataFrame()
Sub1_idx = 0
#DataSetID, ImagePath, ImageID = [], [], []
#ElevClass, AzimClass = [], []
for i in range(len(Sub1) + len(Sub2)):
    if i < len(Sub1):
        InputIdFilePath = ReadLocation1+'/'+Sub1[i]+'/'+idFile
        Sub1_idx = Sub1_idx + 1
    else:
        InputIdFilePath = ReadLocation2+'/'+Sub2[i-Sub1_idx]+'/'+idFile
    
    dfInput = pd.read_csv(InputIdFilePath, sep='\t')
    #dfInput.index = range(dfInput.shape[0])
    dfInput = dfInput.T
    #next(csvfile) #skip heading
    for j in range(dfInput.shape[1]):
        #if not ''.join(row).strip():
        #    continue # ignore the blank lines
        if (i < len(Sub1) or j % ContDownSample == 0):
        
            row = dfInput[j]
            label = str(row['labels'])
            if MyListCheck (ExcludeLabels, label):
                # if the label is in exludelabels, continue the for loop without writing it
                continue
            
            DataSetID = str(row['DataSetID'])
            if i < len(Sub1):
                ImagePath = ReadLocation1 +'/'+Sub1[i]+'/'+str(row['labels'])
            else:
                ImagePath = ReadLocation2 +'/'+Sub2[i-Sub1_idx]
            ImageID = str(row['ImageID'])       
            Elev = float(row['Elev'])*180/np.pi
            Azim = float(row['Azim'])*180/np.pi
            
        
    
            if Elev < ElevStart:
                ElevClass = 0 
            elif Elev > ElevEnd:
                ElevClass = numElevClasses-1
            else:
                ElevClass = np.ceil((Elev-ElevStart)/res)    
                #print('ElevClass =',ElevClass, '\n')    
    
    
            if Azim < AzimStart:
                AzimClass = 0
            elif Azim > AzimEnd:
                AzimClass = numAzimClasses-1
            else:
                AzimClass = np.ceil((Azim-AzimStart)/res)    
                #print('AzimClass =',AzimClass[i], '\n')    

            if np.isnan(AzimClass):
                #doNothing
                print('---------nan value found--------------')
#            elif (i == 2 and j <4000 and validAndTestFlag):
#                with open(ValidFile, 'a+') as csv_output2:
#                    filewriter2 = csv.writer(csv_output2, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
#                    filewriter2.writerow([DataSetID, ImagePath, ImageID, str(ElevClass), str(AzimClass), str(Elev), str(Azim)])             
            else:
                with open(DenseClassificationFile, 'a+') as csv_output:
                    filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
                    filewriter.writerow([DataSetID, ImagePath, ImageID, str(ElevClass), str(AzimClass), str(Elev), str(Azim)])

        



