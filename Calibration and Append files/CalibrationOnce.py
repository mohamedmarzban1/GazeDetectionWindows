# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:20:49 2019

@author: mfm160330
Calibration for back data
"""
import csv
import numpy as np
import pickle

#====== read ID file, Shuffle it, create pathes for train and test data sets =========#
MyFileName = 'AprilBackCalib2019-6-20.csv'
ReadLoc = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/calibration2019-6-20'
OutputFileName = "BackCalibAll2019-6-20.pickle"
NULL_Marker = 2222
#UniqueIDs = list(range(301,321+1))
doorTags = [203, 277, 314, 316, 317, 318]
TimeIncludeStart = np.array([[0,0],[3,12]])  #Include frames between those times in door tags [min,sec]
TimeIncludeEnd =  np.array([[2,55],[7,19]])  
TagConsider = 100 # Consider only all tags above this value
idFile = ReadLoc+'/'+MyFileName

#### ===== Read detection IDs, hamming distance error, x,y,z of the tag  from CSV ==== ####
xCartesian = []
yCartesian = []
zCartesian = []
hamDistErrs = []
detIDs = []
countTest = 0
with open(idFile, "r") as csvfile:
    next(csvfile) #skip heading
    readCSV = csv.reader(csvfile, delimiter='\t')
    for frameNum,detID,hamDistErr,dist,x,y,z,yaw,pitch,roll in readCSV:
        countTest = countTest + 1
        detIDs.append(int(detID))
        hamDistErrs.append(int(hamDistErr))
        xCartesian.append(float(x))
        yCartesian.append(float(y))
        zCartesian.append(float(z))

#### ====== Put all uniques detected Tags in TagIDs ===== ####
TagIDs = np.unique(detIDs)  # get all detected tags
index = np.argwhere(TagIDs == NULL_Marker) 
TagIDs = np.delete(TagIDs, index) # remove the null marker from unique list of detected tags
index2 =  np.argwhere(TagIDs < TagConsider)
TagIDs = np.delete(TagIDs, index2) # remove the calibration tags in the car

### === Transform the columns to numpy arrays ==== ####
detIDs = np.array(detIDs) # All detected tags (column 1 in the AprilTag output)
xCartesian = np.array(xCartesian) 
yCartesian = np.array(yCartesian)
zCartesian = np.array(zCartesian)
hamDistErrs = np.array(hamDistErrs)

### ==== Intialize all the variables to all zeros ==== ####
xAvg = np.zeros([1, len(TagIDs)]) 
yAvg = np.zeros([1, len(TagIDs)]) 
zAvg = np.zeros([1, len(TagIDs)]) 
numElem = np.zeros([1, len(TagIDs)]) 

##############################################
## find the range of rows in the csv file where door is open
framesCount = 1
RowCount = 0
framesIncludeStart = np.matmul(TimeIncludeStart, np.array([60*60,60]))
framesIncludeEnd = np.matmul(TimeIncludeEnd, np.array([60*60,60]))
framesIncludeStartEnd = np.sort(np.concatenate((framesIncludeStart, framesIncludeEnd)))

IndxRowIncludedStartEnd = []
numSegments = framesIncludeStart.shape[0] # number of video segments to be conidered
loopVal = 0 
for i1 in  range(detIDs.shape[0]): #framesIncludeStart.shape[0]:
    RowCount = RowCount + 1
    currentDetID = detIDs[i1]
    if currentDetID == NULL_Marker:
        framesCount = framesCount + 1
    
       
    if framesCount == framesIncludeStartEnd[loopVal]:
        loopVal = loopVal + 1
        IndxRowIncludedStartEnd.append(RowCount)
        
    if loopVal == framesIncludeStartEnd.shape[0]:
        break


### ===== Put each frame start and end in 1 list  ===== ####
i=0
RangeIndxRowIncluded=[]
while i<len(IndxRowIncludedStartEnd):
  RangeIndxRowIncluded.append(IndxRowIncludedStartEnd[i:i+2])
  i+=2    

#########################################
for i1 in range(len(TagIDs)):
    
    #####==== If the Tag is on the car's door, remove the amount of time where the door was open ====#####
    if (set([TagIDs[i1]]) & set(doorTags)):
        IndxRowIncluded = np.array([], dtype=int)
        for i2 in range(len(RangeIndxRowIncluded)):
            startFrameIndx = RangeIndxRowIncluded[i2][0]
            endFrameIndx = RangeIndxRowIncluded[i2][1]
            IndxRowIncluded = np.append(IndxRowIncluded,np.arange(startFrameIndx,endFrameIndx))
        detIDsIncluded  = detIDs[IndxRowIncluded]
        xCartesianIncluded = xCartesian[IndxRowIncluded] 
        yCartesianIncluded = yCartesian[IndxRowIncluded]
        zCartesianIncluded = zCartesian[IndxRowIncluded]
        hamDistErrsIncluded = hamDistErrs[IndxRowIncluded]
                
    else:
        # === Not a door tag ====#
        detIDsIncluded = detIDs
        xCartesianIncluded = xCartesian
        yCartesianIncluded = yCartesian
        zCartesianIncluded = zCartesian
        hamDistErrsIncluded = hamDistErrs
                
    ### === Remove Tags having high hamming distance error from LISTs === ###
    index3 = np.argwhere(hamDistErrsIncluded > 0) 
    hamDistErrsIncluded = np.delete(hamDistErrsIncluded, index3)
    detIDsIncluded = np.delete(detIDsIncluded, index3) # All detected tags (column 1 in the AprilTag output)
    xCartesianIncluded = np.delete(xCartesianIncluded, index3) 
    yCartesianIncluded = np.delete(yCartesianIncluded, index3)
    zCartesianIncluded = np.delete(zCartesianIncluded, index3)
    

    
    IDOneTag = np.array(np.where(detIDsIncluded == TagIDs[i1]))
    if IDOneTag.shape[1] == 0:
        # All detections of this tag do not meet the requirements (door was open or high hamming errors)
        continue
    else:
        xOneTag = xCartesianIncluded[IDOneTag]
        yOneTag = yCartesianIncluded[IDOneTag]
        zOneTag = zCartesianIncluded[IDOneTag]

        # Get 95 percentile and 5 percentile to remove outliers from data before averaging
        x95Perc = np.percentile(xOneTag,95)   
        x5Perc = np.percentile(xOneTag,5)
        y95Perc = np.percentile(yOneTag,95)
        y5Perc = np.percentile(yOneTag,5)
        z95Perc = np.percentile(zOneTag,95)
        z5Perc = np.percentile(zOneTag,5)

        xOneTag[np.logical_and(xOneTag>=x5Perc,xOneTag<x95Perc)]
        yOneTag[np.logical_and(yOneTag>=y5Perc,yOneTag<y95Perc)]
        zOneTag[np.logical_and(zOneTag>=z5Perc,zOneTag<z95Perc)]
            
        xAvg[0,i1] = np.average(xOneTag)
        yAvg[0,i1] = np.average(yOneTag)
        zAvg[0,i1] = np.average(zOneTag)
        numElem[0,i1] = xOneTag.shape[1]
        
print("detected IDs = ",TagIDs)
print("x = ",xAvg)
print("y = ",yAvg)
print("z= ",zAvg)
print("number of detected tags for each ID is", numElem)

# Save the data
pickle_out = open(OutputFileName,"wb")
pickle.dump(TagIDs, pickle_out)
pickle.dump(xAvg, pickle_out)
pickle.dump(yAvg, pickle_out)
pickle.dump(zAvg, pickle_out)
pickle.dump(numElem, pickle_out)
pickle_out.close()



zz= 1
