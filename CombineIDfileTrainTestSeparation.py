# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:25:37 2020

@author: mfm160330
"""
"""

1- Combine more than 1 ID file
2- Convert Angles to labels
3- Data separation

"""

import pandas as pd
import csv
import numpy as np 
import os
from random import randint

### ==== A function that checks if a label is present in a list ==== ###
def MyListCheck (MyList, label):
    for x in MyList:
        if x == label:
            return True
    return False


def getSepratorContGaze(ReadLocation2, sub_2):
    #Sep = ContSeparation[i-Sub1_idx] #separation frame between trainning and testing
    allFiles = os.listdir(ReadLocation2 + '/' + sub_2 + '/Face')
    numImages = len(allFiles)
    maxNum = 0
    for i in range(len(allFiles)):
        imageName =  allFiles[i]
        SplittedID = imageName.split("_f")[0]
        maxNum = max(int(SplittedID.split("_c")[1]), maxNum)
    Sep = int(maxNum - numImages + 0.9*numImages)
    return Sep


def getSeparatorFixedGaaze(ReadLocation1, sub_1):
    startLabel = randint(1, 21)
    labelName = CategoriesDict[startLabel]
    allFiles = os.listdir(ReadLocation1 + '/' + sub_1 + '/' + labelName + '/Face')
    
    Sep = 0 
    for i in range(len(allFiles)):
        imageName90 = allFiles[i]
        SplittedID = imageName90.split("-")
        SplittedID.reverse()
        Sep = max(int(SplittedID[FrameIndexNumberRev]), Sep)
    
    return Sep
        


#### ====== A fuctaion that determines whether this example, a trainning or test or validation
def TrainOrFixedOrFixedExcludedOrCont (i,n,ImageID,Sep,TestLabels,TestFlag,label):
    #OutputFileIndx = 0: testFixedFile, 1: testFixedExcludeMarkersFile , 2:testContFile , 3:TrainAllFile
    if (i<n):
        #fixedGaze #OutputFileIndx = 0,1,3
        
        if MyListCheck (TestLabels,label):
            OutputFileIndx = 1 #testFixedExcludeMarkersFile
            
        else:
            if TestFlag:
                OutputFileIndx = 0
            else:
                SplittedID = ImageID.split("-")
                SplittedID.reverse()
                frameNum = int(SplittedID[FrameIndexNumberRev])
                if frameNum < Sep:
                    OutputFileIndx = 3 #testCont
                else:
                    OutputFileIndx = 0
        
    else:
        #contGaze #OutputFileIndx = 2,3
        if TestFlag:
            OutputFileIndx = 2
        else:    
            SplittedID = ImageID.split("_f")[0]
            frameNum = int(SplittedID.split("_c")[1])
            if frameNum < Sep:
                OutputFileIndx = 3
            else:
                OutputFileIndx = 2
    
    return OutputFileIndx

#===== Intialize parameters ============#
TestFlag = 0    #1: test subjects, 0: Train subjects
splitUniqueID = 'Z1' 
#OffsetFixed = 100 # separation bet. Train and test data (same subj) # useless if TestFlag = 1
#OffsetCont = 100 #separattion bet Train and test data (same subj) #useless if TestFlag = 1
CategoriesDict = {4:"a- 4", 1:"b- 1", 8:"c- 8", 2:"d- 2", 13:"e- 13", 5:"f- 5", 9:"g- 9", 11:"h- 11", 6:"i- 6", 20:"j- 20", 19:"k- 19", 18:"l- 18", 21:"m- 21", 17:"n- 17", 16:"o- 16", 14:"p- 14", 3:"q- 3", 7:"r- 7", 10:"s- 10", 12:"t- 12", 15:"u- 15"} 

### Fixed Gaze:
ReadLocation1 = "D:/FixedGazeImages/FaceAndEyes" # "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/ReCalibrationOutputs"
Sub1 = ['FE2018-12-03-001', 'FE2019-05-22-001', 'FE2019-05-30-001', 'FE2019-06-11-001', 'FE2019-06-14-001', 'FE2019-07-09-001', 'FE2019-07-15-001', 'FE2019-10-03-001', 'FE2019-10-03-002', 'FE2019-11-05-001', 'FE2019-11-19-002', 'FE2019-11-22-001', 'FE2019-11-25-001', 'FE2019-12-06-001', 'FE2020-01-18-001', 'FE2020-01-24-001', 'FE2020-01-27-001', 'FE2020-01-28-001', 'FE2020-02-01-001', 'FE2020-02-07-001', 'FE2020-02-08-001', 'FE2020-02-10-001', 'FE2020-02-14-001', 'FE2020-02-22-001', 'FE2020-03-12-001']
idFile1 =   "AnglesIDfileV4.csv" # "fg/AnglesId_Fixed_Gaze_Calib.csv"
FixedSeparation = [22532, 23038, 13500, 10524, 15100, 16213, 23774, 21042, 19250, 18243, 17343, 18343, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000]
FrameIndexNumberRev = 3 #   In image name, after which dash is the frame number located (Reveresed image name)


###Cont Gaze:
ReadLocation2 = "D:/ContGazeImages/FaceAndEyes" #"C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/ReCalibrationOutputs" #input
Sub2 = ['CFE2019-05-22-001', 'CFE2019-06-11-001', 'CFE2019-06-14-001', 'CFE2019-06-21-001', 'CFE2019-07-09-001', 'CFE2019-07-15-001', 'CFE2019-07-19-001', 'CFE2019-08-27-001', 'CFE2019-10-30-001', 'CFE2019-10-31-001', 'CFE2019-11-19-002', 'CFE2019-11-22-001', 'CFE2019-11-25-001', 'CFE2019-12-06-001', 'CFE2020-01-18-001', 'CFE2020-01-24-001', 'CFE2020-01-27-001', 'CFE2020-01-28-001', 'CFE2020-02-01-001', 'CFE2020-02-07-001', 'CFE2020-02-08-001', 'CFE2020-02-10-001', 'CFE2020-02-14-001', 'CFE2020-02-22-001', 'CFE2020-03-12-001'] 
idFile2 = "AnglesIDfileV4.csv"
ContSeparation = []#[1100 1300 1500]#[11095, 8800, 9685, 9826, 10228, 9747, 6839, 12000, 8200, 9000, 9000, 9000, 9000, 9000, 9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000]


if TestFlag:
    Name = 'DifferentSubj'
else:
    Name = 'SameSubj'

OutputPath = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles"

testFixedFile = OutputPath + "/" +"Test" + Name + "Fixed" + splitUniqueID + ".csv"
testFixedExcludeMarkersFile = OutputPath + "/" +"Test" + Name + "FixedExcludeMarkers" + splitUniqueID + ".csv"
testContFile = OutputPath + "/" +"Test" + Name + "Cont" + splitUniqueID + ".csv"
trainAllFile = OutputPath + "/" +"Train" + Name + "All" + splitUniqueID + ".csv" #will be useful if TestFlag = 0

#===============================================#

ContDownSample = 2 #downsample contgaze images by 4
TestLabels = ["b- 1", "q- 3" ,"r- 7", "t- 12"] # exclude those markers from training

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

#====== Open trainning, testing and validation files and write =========#
header = "DataSetID\tImagePath\tImageID\tElevClass\tAzimClass\tElev\tAzim\n"
csv_output = open(testFixedFile, 'w+')
csv_output.write(header)

csv_output2 = open(testFixedExcludeMarkersFile, 'w+')
csv_output2.write(header)

csv_output3 = open(testContFile, 'w+')
csv_output3.write(header)

if not TestFlag:
    csv_output4 = open(trainAllFile, 'w+')
    csv_output4.write(header)


#==== Read and concatenate all ID files =========#
dfAngles = pd.DataFrame()
Sub1_idx = 0
MinElevArray, MaxElevArray, MinAzimArray, MaxAzimArray = [], [], [], []

#======= loop over all files ===============#
for i in range(len(Sub1) + len(Sub2)):
    if i < len(Sub1):
        InputIdFilePath = ReadLocation1+'/'+Sub1[i]+'/'+idFile1
        sub_1 = Sub1[i]
        Sep = getSeparatorFixedGaaze(ReadLocation1, sub_1)#FixedSeparation[i] #separation frame between trainning and testing
        Sub1_idx = Sub1_idx + 1
             
    else:
        InputIdFilePath = ReadLocation2+'/'+Sub2[i-Sub1_idx]+'/'+idFile2
        sub_2 = Sub2[i-Sub1_idx]
        Sep = getSepratorContGaze(ReadLocation2, sub_2)

    dfInput = pd.read_csv(InputIdFilePath, sep='\t')
    dfInput = dfInput.T
    
    #========= Loop over all rows in 1 file ==============#
    for j in range(dfInput.shape[1]):
        row = dfInput[j]
        ImageID = str(row['ImageID'])
        label = str(row['labels'])
        OutputFileIndx = TrainOrFixedOrFixedExcludedOrCont (i,len(Sub1),ImageID,Sep,TestLabels,TestFlag,label)
        #0: testFixedFile, 1: testFixedExcludeMarkersFile , 2:testContFile , 3:TrainAllFile
        

        if (i < len(Sub1) or j % ContDownSample == 0):
            
            
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

            writtenRow = [DataSetID, ImagePath, ImageID, str(ElevClass), str(AzimClass), str(Elev), str(Azim)]
            if np.isnan(AzimClass):
                #doNothing
                print('---------nan value found--------------')
            elif OutputFileIndx == 0:
                with open(testFixedFile, 'a+', newline='' ) as csv_output:
                    filewriter = csv.writer(csv_output, delimiter='\t')            
                    filewriter.writerow(writtenRow)
            elif OutputFileIndx == 1:
                with open(testFixedExcludeMarkersFile, 'a+', newline='' ) as csv_output2:
                    filewriter2 = csv.writer(csv_output2, delimiter='\t')            
                    filewriter2.writerow(writtenRow)             
            elif OutputFileIndx == 2:
                with open(testContFile, 'a+', newline='' ) as csv_output3:
                    filewriter3 = csv.writer(csv_output3, delimiter='\t')            
                    filewriter3.writerow(writtenRow)  
            elif OutputFileIndx == 3 and TestFlag == 0:
                with open(trainAllFile, 'a+', newline='' ) as csv_output4:
                    filewriter4 = csv.writer(csv_output4, delimiter='\t')            
                    filewriter4.writerow(writtenRow)  

csv_output.close()
csv_output2.close()
csv_output3.close()
if not TestFlag:
    csv_output4.close()



