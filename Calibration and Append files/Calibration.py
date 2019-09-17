# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:34:46 2019

@author: mfm160330

This file is used for calibrating the cameras (so far Back camera) every time a drive is performed
# check that the calibration label values make sense
"""

import csv
import numpy as np
import pickle
import rmsd

#====== read ID file, Shuffle it, create pathes for train and test data sets =========#
MyFileName = 'MohamedDrivBackClib.csv'
ReadLoc = 'G:/Multi-sensors gaze Data Collection/TestDrive2018-12-1'
NULL_Marker = 2222
doorTags = [314, 316, 317, 318]
TimeIncludeStart = np.array([[1,23],[2,19]])  #Include frames between those times in door tags [min,sec]
TimeIncludeEnd =  np.array([[1,58],[4,25]])  
TagConsider = 101 # Consider only all tags above this value
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
TagIDs = np.unique(detIDs)  # get all detected tags in sorted manner
index = np.argwhere(TagIDs == NULL_Marker) 
TagIDs = np.delete(TagIDs, index) # remove the null marker from unique list of detected tags
index2 =  np.argwhere(TagIDs < TagConsider)
TagIDs = np.delete(TagIDs, index2) # remove the calibration tags in the car
index2 = np.argwhere(TagIDs == 203)
TagIDs = np.delete(TagIDs, index2) # remove the calibration tags in the car




### === Transform the columns to numpy arrays ==== ####
detIDs = np.array(detIDs) # All detected tags (column 1 in the AprilTag output)
xCartesian = np.array(xCartesian) 
yCartesian = np.array(yCartesian)
zCartesian = np.array(zCartesian)
hamDistErrs = np.array(hamDistErrs)

### ==== Intialize all the variables in current video to all zeros ==== ####
xAvgCurrent = np.zeros([len(TagIDs),1]) 
yAvgCurrent = np.zeros([len(TagIDs),1]) 
zAvgCurrent = np.zeros([len(TagIDs),1]) 
numElem = np.zeros([len(TagIDs),1]) 


for i1 in range(len(TagIDs)):
    

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
            
        xAvgCurrent[i1,0] = np.average(xOneTag).T #X value of the label in current video
        yAvgCurrent[i1,0] = np.average(yOneTag).T
        zAvgCurrent[i1,0] = np.average(zOneTag).T
        numElem[i1,0] = xOneTag.shape[1]

XYZcurrent = np.hstack((xAvgCurrent, yAvgCurrent, zAvgCurrent))
print("detected IDs = ",TagIDs)
print("x = ",xAvgCurrent)
print("y = ",yAvgCurrent)
print("z= ",zAvgCurrent)
print("number of detected tags for each ID is", numElem)

#======== load the saved labels locations ===========#
pickle_in = open("BackCalibrationAll.pickle","rb")
labelIDsUni = pickle.load(pickle_in)

index = np.argwhere(labelIDsUni < 300)
labelIDsRef = labelIDsUni[index]   # detected labels in reference video

XlabelRef = pickle.load(pickle_in)[0][index] # X values of the labels in reference video
YlabelRef = pickle.load(pickle_in)[0][index] # Y values of the labels in ref video
ZlabelRef = pickle.load(pickle_in)[0][index]
XYZref = np.hstack((XlabelRef, YlabelRef, ZlabelRef))


DoubleIndx = np.argwhere(labelIDsRef == TagIDs.T) # Indx of Tags detected in ref video and current video
XYZref = XYZref[ DoubleIndx[:,0],:] 
XYZcurrent = XYZcurrent[ DoubleIndx[:,1],:] 

##=============== rmsd.kabsh requires the point sets to be of size m x D, ====================== ###
# where m is the number of points, D is the cartesian dimension "D=3 in our case"
C_curr = rmsd.centroid (XYZcurrent)
C_ref = rmsd.centroid (XYZref)

XYZcurr_centered = XYZcurrent - C_curr  #XYZ current after centering 
XYZref_centered = XYZref - C_ref

R = rmsd.kabsch(XYZcurr_centered, XYZref_centered) # the optimal rotation matrix to rotate XYZcurrent to XYZref

#CurrUpdate = np.dot(avg_mesh - C_curr, R) + C_ref
## ========================= ###

##### save R, C_curr, C_ref to use in labeling script 
pickle_out = open("KabaschRotTrans"+".pickle","wb")
pickle.dump(R, pickle_out)
pickle.dump(C_curr, pickle_out)
pickle.dump(C_ref, pickle_out)
pickle.dump(XYZcurrent, pickle_out)
pickle.dump(numElem,pickle_out)
pickle_out.close()



zz= 1
