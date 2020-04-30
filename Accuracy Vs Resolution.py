# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:48:27 2019

In this file we draw the accuracy vs resolution and save values to a .mat file
@author: mfm160330
"""

import matplotlib as plt
import scipy.io as sio
import numpy as np

SaveFileName = 'matFiles/AccVsResTestSameSubjectsContX12.mat'#'AccVsResRun16TempExp20Beta0.8.mat'
temporalFlag = 0

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
 


if temporalFlag:
    AccuracyElevArray_t = np.empty([numElevClasses,1], dtype = float)
    AccuracyAzimArray_t = np.empty([numAzimClasses,1], dtype = float)

AccuracyElevArray = np.empty([numElevClasses,1], dtype = float)
AccuracyAzimArray = np.empty([numAzimClasses,1], dtype = float)
for i in range(numElevClasses):
    AccuracyElevArray[i] = AccHigestN(y_Elev_truth, y_Elev_soft,i+1)
    
               
for i2 in range(numAzimClasses):
    AccuracyAzimArray[i2] = AccHigestN(y_Azim_truth, y_Azim_soft,i2+1)
    
    
if temporalFlag:
    for i3 in range(numElevClasses):
        AccuracyElevArray_t[i3] = AccHigestN(y_Elev_truth, y_Elev_soft_t,i3+1)
    
               
    for i4 in range(numAzimClasses):
        AccuracyAzimArray_t[i4] = AccHigestN(y_Azim_truth, y_Azim_soft_t,i4+1)


if temporalFlag:
    sio.savemat(SaveFileName,{'AccuracyElevArray':AccuracyElevArray,'AccuracyAzimArray':AccuracyAzimArray,'AccuracyElevArray_t':AccuracyElevArray_t,'AccuracyAzimArray_t':AccuracyAzimArray_t,'y_Elev_truth':y_Elev_truth,'y_Elev_soft':y_Elev_soft,'y_Azim_truth':y_Azim_truth,'y_Azim_soft':y_Azim_soft, 'y_Elev_soft_t':y_Elev_soft_t, 'y_Azim_soft_t':y_Azim_soft_t})
else:    
    sio.savemat(SaveFileName,{'AccuracyElevArray':AccuracyElevArray,'AccuracyAzimArray':AccuracyAzimArray,'y_Elev_truth':y_Elev_truth,'y_Elev_soft':y_Elev_soft,'y_Azim_truth':y_Azim_truth,'y_Azim_soft':y_Azim_soft})


   