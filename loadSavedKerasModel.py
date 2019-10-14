# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:06:00 2019
Load a saved model and test data and perform testing for your data 

@author: mfm160330
"""
#Don't forget to load your data set
import numpy as np
from keras.models import load_model
import csv
import cv2
import os
from random import shuffle 



# ===== A function that takes the batch IDs as inputs, extract images preprocess them and returns a numpy array and their Labels ===#
def MyPrepareData (batch_IDs):

    X_Face, X_LEye, X_REye = [], [], [] 
    y_Elev, y_Azim = [], [], 
    for DataSetID, ImagePath, ImageID, ElevClass, AzimClass, _, _ in batch_IDs:
        FullFaceID = ImagePath+'/Face/'+'F'+ImageID
        Face_array = cv2.imread(FullFaceID)  # convert to array
        #if Face_array == None:
        #    print('can not read image '+os.path.join(ImagePath+'Face','F'+ImageID)+'\n')
        #    continue
        Left_array = cv2.imread(os.path.join(ImagePath,'Leye','L'+ImageID) ) 
        Right_array = cv2.imread(os.path.join(ImagePath,'Reye','R'+ImageID) ) 
        X_Face.append(cv2.resize(Face_array, (FaceResize, FaceResize))/255)  # resize to normalize data size and rescale it
        X_LEye.append(cv2.resize(Left_array, (EyeResize, EyeResize))/255)  
        X_REye.append(cv2.resize(Right_array, (EyeResize, EyeResize))/255)
        y_Elev.append(ElevClass)
        y_Azim.append(AzimClass)
        
    X_Face = np.array(X_Face).reshape(-1,FaceResize,FaceResize,3)
    X_LEye = np.array(X_LEye).reshape(-1,EyeResize,EyeResize,3)
    X_REye = np.array(X_REye).reshape(-1,EyeResize,EyeResize,3)
    
    #y_Elev = list(map(float, y_Elev))
    #y_Azim = list(map(float, y_Azim))
    return X_Face, X_REye, X_LEye, y_Elev, y_Azim        


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
    
            

#### 
SavedModel = 'mySavedModels/run15SimpleNetwork.h5'
testDataSetFile = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseNineTestV3.csv'#DenseTest2019-5-30Fixed.csv'

FaceResize = 224
EyeResize = 64
MyBatchSize = 32
NumUsedTest = 2000 # number of test samples


#==== Dense classificiation Parameters ======#
numElevClasses = 14 #number of Elevation Angles classes, 1) theta<=-45 2) -45<theta<=-43 3) -43<theta<=-41 .... 47) 45<theta
numAzimClasses = 38 #number of Azimuth Angles classes, 1) phi<=-90 2) -90<phi<=-88 3) -43<theta<=-41 .... 92) 90<phi
softLabels = 1 #transform the hard labels into soft ones to penalize errors differently 
IsEyes = 1

# load model
print('Started loading Model \n')
model_final = load_model(SavedModel)
print('finished loading model \n')
# summarize model
model_final.summary()
# load dataset
##N.B.: for now, data set is loaded manully by loading .spydata file
TestIDs = []
with open(testDataSetFile, "r") as csvfile:
   readCSV = csv.reader(csvfile, delimiter='\t')
   next(csvfile) #skip heading
   for row in readCSV:
       if not ''.join(row).strip():
           continue # ignore the blank lines
       TestIDs.append(row)
shuffle(TestIDs)
#TestIDs = TestIDs[0:NumUsedTest] 

# Test in batches

num_t = int(np.floor(len(TestIDs)/MyBatchSize)) #number of test iterations
num_t_s = num_t*MyBatchSize #number of actual test samples 
y_Elev_truth, y_Azim_truth = [], []
y_Elev_soft = np.empty([num_t_s,numElevClasses])
y_Azim_soft = np.empty([num_t_s,numAzimClasses])
#y_Elev_soft, y_Azim_soft = [], [] 
for i in range(num_t):
    TestIDsBatch = TestIDs[i*MyBatchSize:(i+1)*MyBatchSize]
    X_F_test_b, X_R_test_b, X_L_test_b, y_Elev_truth_b, y_Azim_truth_b = MyPrepareData (TestIDsBatch) #test values
    y_Elev_truth = y_Elev_truth +  list(map(float, y_Elev_truth_b))
    y_Azim_truth = y_Azim_truth + list(map(float, y_Azim_truth_b))
    [y_Elev_soft_b, y_Azim_soft_b] = model_final.predict([X_F_test_b, X_R_test_b, X_L_test_b]) # predictions for Test data
    y_Elev_soft[i*MyBatchSize:(i+1)*MyBatchSize,:] = y_Elev_soft_b
    y_Azim_soft[i*MyBatchSize:(i+1)*MyBatchSize,:] = y_Azim_soft_b
    
y_Elev_pred = np.argmax(y_Elev_soft, axis=1)
y_Azim_pred = np.argmax(y_Azim_soft, axis=1)


#predict output
#print('started predicting model \n')
#y_Elev_truth = list(map(float, y_Elev_truth))
#y_Azim_truth = list(map(float, y_Azim_truth))
#[y_Elev_soft, y_Azim_soft] = model_final.predict([X_Face_test, X_REye_test, X_LEye_test], batch_size=32) # predictions for Test data
#print('finished predicting model \n')

# evaluate the model
#score = model.evaluate(X, Y, verbose=0)
y_Elev_pred = np.argmax(y_Elev_soft, axis=1)
y_Azim_pred = np.argmax(y_Azim_soft, axis=1)

ElevAccuracy = AccuracyCal(y_Elev_truth, y_Elev_pred)
print('Elevation Accuracy = ', ElevAccuracy, "\n")

AzimAccuracy = AccuracyCal(y_Azim_truth, y_Azim_pred)
print('Azimuth Accuracy = ', AzimAccuracy, "\n")      


## Elevation and Azimuth Accuracy for double resolution
Elev_acc_2 = DoubleResAccuracy(y_Elev_truth, y_Elev_soft)
print("Elevation Accuracy double resolution = ", Elev_acc_2, "\n")
  
Azim_acc_2 = DoubleResAccuracy(y_Azim_truth, y_Azim_soft)
print("Azimuth Accuracy double resolution = ", Azim_acc_2, "\n")


Elev_acc_highest4 = AccHigestN (y_Elev_truth, y_Elev_soft,4)
print("Elevation Accuracy 8deg resolution = ", Elev_acc_highest4, "\n")

Azim_acc_highest4 = AccHigestN (y_Azim_truth, y_Azim_soft,4)
print("Azimuth Accuracy 8deg resolution = ", Azim_acc_highest4, "\n")


Elev_acc_highest5 = AccHigestN (y_Elev_truth, y_Elev_soft,5)
print("Elevation Accuracy 10deg resolution = ", Elev_acc_highest5, "\n")

Azim_acc_highest5 = AccHigestN (y_Azim_truth, y_Azim_soft,5)
print("Azimuth Accuracy 10deg resolution = ", Azim_acc_highest5, "\n")



Elev_acc_highest6 = AccHigestN (y_Elev_truth, y_Elev_soft,6)
print("Elevation Accuracy 12deg resolution = ", Elev_acc_highest6, "\n")

Azim_acc_highest6 = AccHigestN (y_Azim_truth, y_Azim_soft,6)
print("Azimuth Accuracy 12deg resolution = ", Azim_acc_highest6, "\n")


Elev_acc_highest7 = AccHigestN (y_Elev_truth, y_Elev_soft,7)
print("Elevation Accuracy 14deg resolution = ", Elev_acc_highest7, "\n")

Azim_acc_highest7 = AccHigestN (y_Azim_truth, y_Azim_soft,8)
print("Azimuth Accuracy 12deg resolution = ", Azim_acc_highest7, "\n")