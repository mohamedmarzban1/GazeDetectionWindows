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


#===== Intialize parameters ============#
### Fixed Gaze:
ReadLocation1 = [] # "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/FaceAndEyes" #input
Sub1 = [] #["FE2018-12-3", "FE2018-12-1", "FE2019-5-22", "FE2019-5-30"] #["FE2019-5-22"] #["FE2019-5-30"] ##input
idFile1 = [] #["AnglesIDfile.csv", "AnglesIDfile.csv", "AnglesIDfile.csv", "AnglesIDfile.csv"] #["AnglesIDfile.csv", "AnglesIDfile.csv"]

###Cont Gaze:
ReadLocation2 = "G:/ContGazeImages/FaceAndEyes"
Sub2 = ["CFE2019-7-10", "CFE2019-7-11"] # ["CFE2019-7-9", "CFE2019-5-30", "CFE2019-6-11", "CFE2019-7-11"] #[]#["CFE2019-7-10"] #
idFile2 = ["AnglesIDfile.csv", "AnglesIDfile.csv"] #["CFE2019-7-9IDFileCleaned.csv", "CFE2019-5-30FinalFormatIDFile.csv", "CFE2019-6-11FinalFormatIDFile.csv", "CFE2019-7-11IDFileCleaned.csv"]#["CFE2019-7-10FinalFormatIDFile.csv"] #["CFE2019-7-9IDFileCleaned.csv"] 

DenseClassificationFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNineV3TestTemporalNotDownsampled.csv" #output
#===============================================#

# Dense classificiation Parameters:
# Make sure (ElevEnd -Elevstart)/res is an integar 
ContDownSample = 1 #downsample contgaze images by 4
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
header = "DataSetID\tImagePath\tImageID\tElevClass\tAzimClass\tElev\tAzim\tImageCount\n"
csv_output.write(header)


#==== Read and concatenate all ID files =========#
dfAngles = pd.DataFrame()
Sub1_idx = 0
#DataSetID, ImagePath, ImageID = [], [], []
#ElevClass, AzimClass = [], []
for i in range(len(Sub1) + len(Sub2)):
    if i < len(Sub1):
        InputIdFilePath = ReadLocation1+'/'+Sub1[i]+'/'+idFile1[i]
        Sub1_idx = Sub1_idx + 1
    else:
        InputIdFilePath = ReadLocation2+'/'+Sub2[i-Sub1_idx]+'/'+idFile2[i-Sub1_idx]
    
    dfInput = pd.read_csv(InputIdFilePath, sep='\t')
    #dfInput.index = range(dfInput.shape[0])
    dfInput = dfInput.T
    #next(csvfile) #skip heading
    for j in range(dfInput.shape[1]):
        
        if (j % ContDownSample == 0):
            #if not ''.join(row).strip():
            #    continue # ignore the blank lines
            row = dfInput[j]
            DataSetID = str(row['DataSetID'])
            if i < len(Sub1) :
                ImagePath = ReadLocation1 +'/'+Sub1[i]+'/'+str(row['labels'])
            else:
                ImagePath = ReadLocation2 +'/'+Sub2[i-Sub1_idx]
            ImageID = str(row['ImageID'])
            Elev = float(row['Elev'])*180/np.pi
            Azim = float(row['Azim'])*180/np.pi
            TmpSplit = ImageID.split("_c")
            ImageCount = int(TmpSplit[1].split("_f")[0])

    
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
                z =1 #doNothing
                print('---------nan value found--------------')
            else:
                with open(DenseClassificationFile, 'a+') as csv_output:
                    filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
                    filewriter.writerow([DataSetID, ImagePath, ImageID, str(ElevClass), str(AzimClass), str(Elev), str(Azim), ImageCount])

        



