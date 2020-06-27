# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:50:57 2019

@author: mfm160330

1- Combine more than 1 ID file


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
    
#Categories = ["a- 4", "b- 1", "c- 8", "d- 2", "e- 13", "f- 5", "g- 9", "h- 11", "i- 6", "j- 20", "k- 19", "l- 18", "m- 21", "n- 17", "o- 16", "p- 14", "q- 3", "r- 7", "s- 10", "t- 12" ,"u- 15" ] 
ReadLocation1 =  "D:/FixedGazeImages/FaceAndEyes" ## input
Sub1 = ['FE2018-12-03-001', 'FE2019-05-22-001', 'FE2019-05-30-001', 'FE2019-06-11-001', 'FE2019-06-14-001', 'FE2019-07-09-001', 'FE2019-07-10-001', 'FE2019-07-11-001', 'FE2019-07-15-001', 'FE2019-07-23-001', 'FE2019-10-03-001', 'FE2019-10-03-002', 'FE2019-11-05-001', 'FE2019-11-14-001', 'FE2019-11-19-002', 'FE2019-11-20-001', 'FE2019-11-22-001', 'FE2019-11-25-001', 'FE2019-12-06-001', 'FE2020-01-18-001', 'FE2020-01-24-001', 'FE2020-01-25-001', 'FE2020-01-27-001', 'FE2020-01-28-001', 'FE2020-02-01-001', 'FE2020-02-07-001', 'FE2020-02-08-001', 'FE2020-02-10-001', 'FE2020-02-14-001', 'FE2020-02-22-001', 'FE2020-03-12-001']
idFile1 =   "AnglesIDfileV4.csv" ##input

###Cont Gaze:
#AllDirs = os.listdir(ReadLocation1)
ReadLocation2 = "D:/ContGazeImages/FaceAndEyes"  ##input
Sub2 = ['CFE2019-05-22-001', 'CFE2019-06-11-001', 'CFE2019-06-14-001', 'CFE2019-06-21-001', 'CFE2019-07-09-001', 'CFE2019-07-10-001', 'CFE2019-07-11-001', 'CFE2019-07-15-001', 'CFE2019-07-19-001', 'CFE2019-07-23-001', 'CFE2019-08-27-001', 'CFE2019-10-30-001', 'CFE2019-10-31-001', 'CFE2019-11-14-001', 'CFE2019-11-19-002', 'CFE2019-11-20-001', 'CFE2019-11-22-001', 'CFE2019-11-25-001', 'CFE2019-12-06-001', 'CFE2020-01-18-001', 'CFE2020-01-24-001', 'CFE2020-01-25-001', 'CFE2020-01-27-001', 'CFE2020-01-28-001', 'CFE2020-02-01-001', 'CFE2020-02-07-001', 'CFE2020-02-08-001', 'CFE2020-02-10-001', 'CFE2020-02-14-001', 'CFE2020-02-22-001', 'CFE2020-03-12-001']
idFile2 = "AnglesIDfileV4.csv" ##input

DenseClassificationFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/AllAnglesZ1.csv" #"G:/ContGazeImages/FaceAndEyes/CombineCont2019-7-19.csv" #output
#validAndTestFlag = 1
#ValidFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNineValidV3.csv"
#===============================================#

ContDownSample = 1#100 #downsample contgaze images by 4
FixedDownSample = 1
ExcludeLabels = [] # ["a- 4", "b- 1", "c- 8", "d- 2", "e- 13", "g- 9", "h- 11", "i- 6", "j- 20", "k- 19", "l- 18", "m- 21", "n- 17", "o- 16", "p- 14", "r- 7", "s- 10", "t- 12" ,"u- 15" ] #["q- 3", "f- 5"] #[] # exclude those markers from training

# Dense classificiation Parameters:
# Make sure (ElevEnd -Elevstart)/res is an integar 
ElevSeparatorAngle = -32 # Angle that differentiates dashboard area from gear shifter area
ElevStart = -24 # in degrees
ElevEnd = 20 #in degrees

AzimSeparatorAngle = -54 # Angle that differentiates mirror area from side window area
AzimStart = -26 # in degrees
AzimEnd = 64 # in degrees

res = 2 #Resolution of Elevation and Azimuth Angles classes in degrees
#===================================#

numElevClasses = (ElevEnd - ElevStart)/res + 2 + 1 # We added an extra elevation class to differentiate speedometer area from gear shifter area  
numAzimClasses = (AzimEnd - AzimStart)/res + 2 + 1 # We added an extra elevation class to differentiate mirror area from side window area 



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
MinElevArray, MaxElevArray, MinAzimArray, MaxAzimArray = [], [], [], []
#DataSetID, ImagePath, ImageID = [], [], []
#ElevClass, AzimClass = [], []
for i in range(len(Sub1) + len(Sub2)):
    if i < len(Sub1):
        InputIdFilePath = ReadLocation1+'/'+Sub1[i]+'/'+idFile1
        Sub1_idx = Sub1_idx + 1
    else:
        InputIdFilePath = ReadLocation2+'/'+Sub2[i-Sub1_idx]+'/'+idFile2
    
    dfInput = pd.read_csv(InputIdFilePath, sep='\t')
    #dfInput.index = range(dfInput.shape[0])
    dfInput = dfInput.T
    #next(csvfile) #skip heading
    MyMinElev, MyMinAzim = float('inf'), float('inf')
    MyMaxElev, MyMaxAzim = -float('inf'), -float('inf')
    for j in range(dfInput.shape[1]):
        #if not ''.join(row).strip():
        #    continue # ignore the blank lines

        if ((i < len(Sub1) and j % FixedDownSample == 0) or  (i >= len(Sub1) and j % ContDownSample == 0)):
        
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
            Azim = float(row['Azim'])*180/np.pi #flipping the y-axis
            
                
            
            if Elev < ElevSeparatorAngle:
                ElevClass = 0 
            elif Elev < ElevStart:
                ElevClass = 1 
            elif Elev > ElevEnd:
                ElevClass = numElevClasses-1
            else:
                ElevClass = np.ceil((Elev-ElevStart)/res)+1   
                #print('ElevClass =',ElevClass, '\n')    
    
            if Azim < AzimSeparatorAngle:
                AzimClass = 0
            elif Azim < AzimStart:
                AzimClass = 1
            elif Azim > AzimEnd:
                AzimClass = numAzimClasses-1
            else:
                AzimClass = np.ceil((Azim-AzimStart)/res)+1    

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


csv_output.close()

