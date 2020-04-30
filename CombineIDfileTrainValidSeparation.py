# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:56:27 2019

@author: mfm160330


1- Combine more than 1 ID file
2- Convert Angles to labels
3- N.B: the y-axis is flipped in this file
4- Data separation

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


#### ====== A fuctaion that determines whether this example, a trainning or test or validation
def TrainOrValidate (i,n,ImageID,trainValidFlag,Sep):
    if (i<n):
        #fixedGaze
        SplittedID = ImageID.split("-")
        SplittedID.reverse()
        frameNum = int(SplittedID[FrameIndexNumberRev])
        if frameNum > Sep:
            trainValidFlag = 1
        
    else:
        #contGaze
        SplittedID = ImageID.split("_f")[0]
        frameNum = int(SplittedID.split("_c")[1])
        if frameNum > (Sep + ContSeparationMargin):
            trainValidFlag = 1
        elif (frameNum > Sep and frameNum < (Sep + ContSeparationMargin)):
            trainValidFlag = 100 # a buffer frame between trainning and validation should be unused
    
    return trainValidFlag

#===== Intialize parameters ============#
### Fixed Gaze:
ReadLocation1 =  "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/FaceAndEyes" # "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/ReCalibrationOutputs"
Sub1 =  ["FE2018-12-1", "FE2018-12-3", "FE2019-5-22", "FE2019-5-30", "FE2019-6-14", "FE2019-7-9", "FE2019-7-11", "FE2019-7-15", "FE2019-7-23","FE2019-10-03", "FE2019-10-31"]
idFile1 =   "AnglesIDfile.csv" # "fg/AnglesId_Fixed_Gaze_Calib.csv"
FixedSeparation = [float('inf'), float('inf'), 13400, float('inf'), 24640, float('inf'), 15113, 18940, 17143, float('inf'), float('inf')]
FrameIndexNumberRev = 3 #   In image name, after which dash is the frame number located (Reveresed image name)


###Cont Gaze:
ReadLocation2 = "G:/ContGazeImages/FaceAndEyes"#"C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/ReCalibrationOutputs" #input
Sub2 =  ["CFE2019-5-22", "CFE2019-5-30", "CFE2019-6-14", "CFE2019-6-21", "CFE2019-7-9", "CFE2019-7-11","CFE2019-7-15", "CFE2019-7-19", "CFE2019-7-23"] 
idFile2 = "AnglesIDfile.csv"
ContSeparation = [11095, 11000, 9685, 9826, 11000, 9747, float('inf'), 11400, 8900, float('inf'), float('inf')]

OutputPath = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles"
TrainningFile = OutputPath + "/" + "ElevenTrainX10.csv" #"C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNine.csv" #output
ValidationFile = OutputPath + "/" + "/ElevenValidX10.csv" #"C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNine.csv" #output
TwoMarkersTestFile = OutputPath + "/" + "/ElevenTwoMarkersTestX10.csv"
ContSeparationMargin = 200 # need to keep this ContSeparationMargin frames unused between trainning and validation for continuous

#===============================================#
FixedDownSample = 2 #downsample FixedGaze images by 2
ContDownSample = 4 #downsample contgaze images by 4
TestLabels = ["q- 3", "f- 5"] # exclude those markers from training

# Dense classificiation Parameters:
# Make sure (ElevEnd -Elevstart)/res is an integar 
ElevStart = 78 # in degrees
ElevEnd = 102 #in degrees
AzimStart = -30 # in degrees
AzimEnd = 52 # in degrees
res = 2 #Resolution of Elevation and Azimuth Angles classes in degrees
#===================================#

numElevClasses = (ElevEnd - ElevStart)/res + 2 + 1 # We added an extra elevation class to differentiate speedometer area from gear shifter area  
numAzimClasses = (AzimEnd - AzimStart)/res + 2 + 1 # We added an extra elevation class to differentiate mirror area from side window area 

ElevSeparatorAngle = 120 # Angle that differentiates dashboard area from gear shifter area
AzimSeparatorAngle = -40 # Angle that differentiates mirror area from side window area

#====== Open trainning, testing and validation files and write =========#
header = "DataSetID\tImagePath\tImageID\tElevClass\tAzimClass\tElev\tAzim\n"
csv_output = open(TrainningFile, 'w+')
csv_output.write(header)

csv_output3 = open(ValidationFile, 'w+')
csv_output3.write(header)

csv_outputTest = open(TwoMarkersTestFile, 'w+')
csv_outputTest.write(header)


#==== Read and concatenate all ID files =========#
dfAngles = pd.DataFrame()
Sub1_idx = 0
MinElevArray, MaxElevArray, MinAzimArray, MaxAzimArray = [], [], [], []

for i in range(len(Sub1) + len(Sub2)):
    if i < len(Sub1):
        InputIdFilePath = ReadLocation1+'/'+Sub1[i]+'/'+idFile1
        Sub1_idx = Sub1_idx + 1
        Sep = FixedSeparation[i] #separation frame between trainning and validation
        
    else:
        InputIdFilePath = ReadLocation2+'/'+Sub2[i-Sub1_idx]+'/'+idFile2
        Sep = ContSeparation[i-Sub1_idx] #separation frame between trainning and validation

    
    dfInput = pd.read_csv(InputIdFilePath, sep='\t')
    #dfInput.index = range(dfInput.shape[0])
    dfInput = dfInput.T
    #next(csvfile) #skip heading
    MyMinElev, MyMinAzim = float('inf'), float('inf')
    MyMaxElev, MyMaxAzim = -float('inf'), -float('inf')
    
    for j in range(dfInput.shape[1]):
        trainValidFlag = 0  #0: trainning example, 1:valid example
        row = dfInput[j]
        ImageID = str(row['ImageID'])
        trainValidFlag = TrainOrValidate (i,len(Sub1),ImageID,trainValidFlag,Sep)

        if ((i < len(Sub1) and j % FixedDownSample == 0) or  (i >= len(Sub1) and j % ContDownSample == 0)):
            
        
            label = str(row['labels'])
            if MyListCheck (TestLabels, label):
                # if the label is in exludelabels, continue the for loop without writing it
                trainValidFlag = 2
            
            DataSetID = str(row['DataSetID'])
            if i < len(Sub1):
                ImagePath = ReadLocation1 +'/'+Sub1[i]+'/'+str(row['labels'])
            else:
                ImagePath = ReadLocation2 +'/'+Sub2[i-Sub1_idx]
            ImageID = str(row['ImageID'])       
            Elev = float(row['Elev'])*180/np.pi
            Azim = - float(row['Azim'])*180/np.pi #flipping the y-axis
            
            
            if Elev < MyMinElev:
                MyMinElev = Elev
            if Elev > MyMaxElev:
                MyMaxElev = Elev
                
            if Azim < MyMinAzim:
                MyMinAzim = Azim
            if Azim > MyMaxAzim:
                MyMaxAzim = Azim
                
            
            if Elev < ElevStart:
                ElevClass = 0 
            elif Elev > ElevEnd:
                if Elev < ElevSeparatorAngle:
                    ElevClass = numElevClasses-2
                else:
                    ElevClass = numElevClasses-1
            else:
                ElevClass = np.ceil((Elev-ElevStart)/res)    
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
            elif trainValidFlag == 0:
                with open(TrainningFile, 'a+') as csv_output:
                    filewriter = csv.writer(csv_output, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
                    filewriter.writerow(writtenRow)        
            elif trainValidFlag == 1:
                with open(ValidationFile, 'a+') as csv_output3:
                    filewriter3 = csv.writer(csv_output3, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
                    filewriter3.writerow(writtenRow)  
            elif trainValidFlag == 2:
                with open(TwoMarkersTestFile, 'a+') as csv_outputTest:
                    filewriter3 = csv.writer(csv_outputTest, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
                    filewriter3.writerow(writtenRow)  
             

    MinElevArray.append(MyMinElev)  
    MaxElevArray.append(MyMaxElev)
    MinAzimArray.append(MyMinAzim)
    MaxAzimArray.append(MyMaxAzim)    



