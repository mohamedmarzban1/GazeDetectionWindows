# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:36:37 2019
This file is used to Enhance the estimation by using the temporal correlation in the post processing stage
@author: mfm160330
"""

import numpy as np
from keras.models import load_model
import csv
import cv2
import os
from random import shuffle 
import re # for spliting with multiple delimeters
import scipy.io as sio
import numpy.matlib




def TemporalEstimation(y_soft):
    #num_prev = y_soft.shape[0]
    #numClasses = y_soft.shape[1]
    y_soft_predicted = np.average(y_soft, axis = 0) # predicted soft angle
    return y_soft_predicted

# a function that performs temporal estimation through exponential forgetting factor weights
def TempEstForgetFactor(y_soft, beta = 0.95):
    num_frame = y_soft.shape[0] # num_frame = num_prev + 1
    W_i = np.expand_dims (np.power(beta,np.arange(0, num_frame)), axis =0) # weights
    W_i_sum = np.sum(W_i)
    #W_i = np.matlib.repmat(W_i, numClasses, 1 ).T
    y_soft_predicted = np.matmul(W_i,y_soft)/W_i_sum # predicted soft angle
    return y_soft_predicted
    
    
  #======= A function that takes the imageID and the index of the previous temporal image and returns the new ID
def getImageIDfromList(ImageID,ImageCount,i):
    if i == 0:
        return ImageID
    else:
        ImageCount = ImageCount-i
        for ID_t in TestIDs:
            if int(ID_t[7]) == ImageCount:
                print("GotImageFromList", '\n')
                return ID_t[2]

    print("==================== image Not found ====================================", '\n')
    return 2

#======= A function that takes the imageID and the index of the previous temporal image and returns the new ID
def getImageID(ImageID, ImageCount, i):
    if i == 0:
        return ImageID
    else:
        ImageCount = ImageCount-i
        
        ImageF_Count_tmp = ImageID.split("_f")[1]
        ImageF_Count, testFileType = ImageF_Count_tmp.split(".")
        ImageF_Count = int(ImageF_Count)-i
        ImageID_t = ImageID.split("_c")[0]+"_c" + str(ImageCount) + "_f" + str(ImageF_Count) +"."+ testFileType
        return ImageID_t


# ===== A function that takes an ID as input and number of previous frames, extract images preprocess them and returns a numpy array and their Labels ===#
def MyPrepareDataTemporal(elligibleImageID, num_prev):
    X_Face, X_LEye, X_REye = [], [], [] 
    y_Elev, y_Azim = [], [], 
    DataSetID, ImagePath, ImageID, ElevClass, AzimClass, Elev, Azim, ImageCount = elligibleImageID
    ImageCount = int(ImageCount)
    X_Face, X_LEye, X_REye = [], [], [] 
    y_Elev, y_Azim = [], [], 
    ImageIDList = []
    for i in range(num_prev+1):
        ImageID_t = getImageID(ImageID, ImageCount, i) # imageIDtemporal
        FullFaceID = ImagePath+'/Face/'+'F'+ImageID_t            
        Face_array = cv2.imread(FullFaceID)  # convert to array
        try:
            X_Face.append(cv2.resize(Face_array, (FaceResize, FaceResize))/255)  # resize to normalize data size and rescale it
        except:
            ImageID_t = getImageIDfromList(ImageID,ImageCount,i)
            FullFaceID = ImagePath+'/Face/'+'F'+ImageID_t            
            Face_array = cv2.imread(FullFaceID)  # convert to array
        #if Face_array.all() == None:
        #print(ImageID_t, '/n')
        Left_array = cv2.imread(os.path.join(ImagePath,'Leye','L'+ImageID_t) ) 
        Right_array = cv2.imread(os.path.join(ImagePath,'Reye','R'+ImageID_t) ) 

        X_LEye.append(cv2.resize(Left_array, (EyeResize, EyeResize))/255)  
        X_REye.append(cv2.resize(Right_array, (EyeResize, EyeResize))/255)
        y_Elev.append(ElevClass)
        y_Azim.append(AzimClass)
        ImageIDList.append(ImageID_t)
        
    X_Face = np.array(X_Face).reshape(-1,FaceResize,FaceResize,3)
    X_LEye = np.array(X_LEye).reshape(-1,EyeResize,EyeResize,3)
    X_REye = np.array(X_REye).reshape(-1,EyeResize,EyeResize,3)
        
    return X_Face, X_REye, X_LEye, y_Elev, y_Azim, ImageIDList        

    
# ==== A function that takes Test IDs as input and number of previous frames for temporal investigations 
#and returns the elligible frames IDs for temporal investigations ===#
def extractElligibleTemporalFrames(TestIDs, num_prev):
    elligibleImageIDs = []
    count = 0
    frameCountPrev = -1 # to check for num_prev
    lastEligFrameNum = -10 # last Eligible frame number ## to check for ContDownSample
    for i, ID in enumerate(TestIDs):
        ImageID = str(ID[2])
        TmpSplit = ImageID.split("_f")
        #frameCount = int(TmpSplit[1].split(fileType)[0])
        frameCountTmp = re.split(fileTypes,TmpSplit[1])
        frameCount = int(frameCountTmp[0])
        
        if frameCount-frameCountPrev  == 1:
            count = count + 1
        else:
            count = 0
        
        if count > num_prev and frameCount-lastEligFrameNum >= ContDownSample:
            elligibleImageIDs.append(ID)
            lastEligFrameNum = frameCount
        
        frameCountPrev = frameCount
    
    return elligibleImageIDs

# ========== Accuracy calculation function ====================== #
def AccuracyCal(y_truth, y_pred): 
    count1 = 0
    for i1, j1 in zip(y_truth, y_pred):
        if i1 == j1:
            count1 = count1 + 1
    Accuracy = count1/len(y_pred)
    return Accuracy

# ========== Double Resolution Accuracy calculation  ====================== #
def DoubleResAccuracy(y_truth, y_pred_soft): 
    y_predSorted =  np.flip(np.argsort(y_pred_soft, axis =1), axis=1)
    Y_double_res = y_predSorted[:,:2]

    count3 = 0
    for i3, j3, k3 in zip(y_truth, Y_double_res[:,0], Y_double_res[:,1]):
        if (i3 == j3) or (i3 == k3):
            count3 = count3 + 1
    Acc_2 = count3/len(y_truth)
    return Acc_2

# ======= Accuracy for highest N continuous points ===== #
def AccHigestN(y_truth, y_pred_soft,N):
    numTestSamples = len(y_truth) 
    maxIndx = np.zeros((numTestSamples,1), dtype=int) # index corresponding to the start of the highest N values
    maxConfidence = np.zeros((numTestSamples,1))
    maxConf_sep = np.zeros((numTestSamples,N)) #maximum confidence, each resolution separetly
    CorrCount = 0 # counting the correctly classified values
    
    numClasses = y_pred_soft.shape[1]-N+1
    for i1 in range(numTestSamples):
        
        for i2 in range(numClasses):
            confidence = np.sum(y_pred_soft[i1,i2:i2+N]) # current confidence
            if  confidence> maxConfidence[i1]:
                maxIndx[i1] = i2
                maxConfidence[i1] = confidence
                
                
        maxConf_sep[i1,:] = y_pred_soft[i1, np.arange(maxIndx[i1],maxIndx[i1]+N)]
        
        if maxIndx[i1]<=y_truth[i1]<+maxIndx[i1]+N:
            CorrCount = CorrCount + 1
     
    return CorrCount/numTestSamples
 
#=============Intialize Parameters =================#    
SavedModel = 'mySavedModels/run15SimpleNetwork.h5'
testDataSetFile = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNineV3TestTemporalNotDownsampled.csv'  #DenseTemp2019-7-10.csv #DenseTest2019-5-30Fixed.csv'
# a Non-downsampled ID file should be used here

numTestSamples = 10000
#========== temporal Correlation parameters ===============#
num_prev = 20 # number of previous frames used to account for temporal correlation
SaveFileName = 'LdSvdTemprun15Exp20Beta0.8'
ContDownSample = 4  # To make sure we don't test very similar frames (Each two franmes to be tested have to be at least ContDownSample frames apart)
beta = 0.8

#======= Dense classificiation and temporal Parameters ==========#
numElevClasses = 14 #number of Elevation Angles classes, 1) theta<=-45 2) -45<theta<=-43 3) -43<theta<=-41 .... 47) 45<theta
numAzimClasses = 38 #number of Azimuth Angles classes, 1) phi<=-90 2) -90<phi<=-88 3) -43<theta<=-41 .... 92) 90<phi
softLabels = 1 #transform the hard labels into soft ones to penalize errors differently 
IsEyes = 1
fileTypes = '.jpg|.png'

FaceResize = 224
EyeResize = 64
MyBatchSize = 32




##============== load dataset ================##
##N.B.: for now, data set is loaded manully by loading .spydata file
TestIDs = []
with open(testDataSetFile, "r") as csvfile:
   readCSV = csv.reader(csvfile, delimiter='\t')
   next(csvfile) #skip heading
   for row in readCSV:
       if not ''.join(row).strip():
           continue # ignore the blank lines
       TestIDs.append(row)
##============================================##
       


#====== Test temporal (each frame and its separate images separetly) =====#
y_Elev_truth, y_Azim_truth = [], []
elligibleImageIDs = extractElligibleTemporalFrames(TestIDs, num_prev) 
shuffle(elligibleImageIDs)
NumIDs = min(numTestSamples, len(elligibleImageIDs))
print("Number of Elligible IDs = ", NumIDs, "\n")
elligibleImageIDs = elligibleImageIDs[0:NumIDs]
numEligibleIDs = len(elligibleImageIDs)

#=========== load model =================#
model_final = load_model(SavedModel)
print('finished loading model \n')
##### summarize model
#model_final.summary()
#===========================================================#


#=============Testing =============================#
y_Elev_soft_t = np.empty([numEligibleIDs, numElevClasses]) # Soft Elevation estimation with temporal
y_Azim_soft_t = np.empty([numEligibleIDs, numAzimClasses])
y_Elev_soft = np.empty([numEligibleIDs, numElevClasses]) # soft Elevation estimation
y_Azim_soft = np.empty([numEligibleIDs, numAzimClasses])    
for i in range(numEligibleIDs):
    X_F_test_b, X_R_test_b, X_L_test_b, y_Elev_truth_b, y_Azim_truth_b, ImageIDList = MyPrepareDataTemporal(elligibleImageIDs[i], num_prev) # data of all the temporal segment
    [y_Elev_soft_b, y_Azim_soft_b] = model_final.predict([X_F_test_b, X_R_test_b, X_L_test_b]) # predicting values for all batch
    y_Elev_soft_t[i] = TempEstForgetFactor(y_Elev_soft_b, beta)#TemporalEstimation(y_Elev_soft_b) # Predicted elevation angle estimation for the eligible frame after temporal combinning
    y_Azim_soft_t[i] = TempEstForgetFactor(y_Azim_soft_b, beta)#TemporalEstimation(y_Azim_soft_b) # Predicted azimuth angle estimation for the eligible frame after temporal combinning
    y_Elev_truth.append(float(y_Elev_truth_b[0]))
    y_Azim_truth.append(float(y_Azim_truth_b[0]))
    y_Elev_soft[i] = y_Elev_soft_b[0]
    y_Azim_soft[i] = y_Azim_soft_b[0]

#=============================================================================#

#==============Accuracy Evaluation ===================================#
y_Elev_pred_t = np.argmax(y_Elev_soft_t, axis=1) # predicted Elevation Angle with temporal
y_Azim_pred_t = np.argmax(y_Azim_soft_t, axis=1) 
y_Elev_pred = np.argmax(y_Elev_soft, axis=1) # predicted Elevation Angle normally
y_Azim_pred = np.argmax(y_Azim_soft, axis=1)


sio.savemat(SaveFileName,{'y_Elev_truth':y_Elev_truth,'y_Elev_soft':y_Elev_soft,'y_Azim_truth':y_Azim_truth,'y_Azim_soft':y_Azim_soft, 'y_Elev_soft_t':y_Elev_soft_t, 'y_Azim_soft_t':y_Azim_soft_t, 'beta':beta})


# accuracy for temporal
ElevAccuracy_t = AccuracyCal(y_Elev_truth, y_Elev_pred_t)
print('Elevation Accuracy with temporal = ', ElevAccuracy_t, "\n")
AzimAccuracy_t = AccuracyCal(y_Azim_truth, y_Azim_pred_t)
print('Azimuth Accuracy with temporal = ', AzimAccuracy_t, "\n")      

# Accuracy (No temporal)
ElevAccuracy = AccuracyCal(y_Elev_truth, y_Elev_pred)
print('Elevation Accuracy = ', ElevAccuracy, "\n")
AzimAccuracy = AccuracyCal(y_Azim_truth, y_Azim_pred)
print('Azimuth Accuracy = ', AzimAccuracy, "\n")   


## Elevation and Azimuth Accuracy for double resolution (temporal)
Elev_acc_2_t = DoubleResAccuracy(y_Elev_truth, y_Elev_soft_t)
print("Elevation Accuracy double resolution (with temporal) = ", Elev_acc_2_t, "\n")
Azim_acc_2_t = DoubleResAccuracy(y_Azim_truth, y_Azim_soft_t)
print("Azimuth Accuracy double resolution (with temporal) = ", Azim_acc_2_t, "\n")

## Elevation and Azimuth Accuracy for double resolution (Notemporal)
Elev_acc_2 = DoubleResAccuracy(y_Elev_truth, y_Elev_soft)
print("Elevation Accuracy double resolution = ", Elev_acc_2, "\n")
Azim_acc_2 = DoubleResAccuracy(y_Azim_truth, y_Azim_soft)
print("Azimuth Accuracy double resolution = ", Azim_acc_2, "\n")

## With temporal
Elev_acc_highest5_t = AccHigestN (y_Elev_truth, y_Elev_soft_t,5)
print("Elevation Accuracy 10deg resolution (with temporal) = ", Elev_acc_highest5_t, "\n")
Azim_acc_highest5_t = AccHigestN (y_Azim_truth, y_Azim_soft_t,5)
print("Azimuth Accuracy 10deg resolution (with temporal)= ", Azim_acc_highest5_t, "\n")

## No Temporal
Elev_acc_highest5 = AccHigestN (y_Elev_truth, y_Elev_soft,5)
print("Elevation Accuracy 10deg resolution = ", Elev_acc_highest5, "\n")
Azim_acc_highest5 = AccHigestN (y_Azim_truth, y_Azim_soft,5)
print("Azimuth Accuracy 10deg resolution = ", Azim_acc_highest5, "\n")
