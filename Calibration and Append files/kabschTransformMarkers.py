# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:04:13 2019

@author: mfm160330
"""


import csv
import numpy as np
import pickle
import rmsd

XYZcurrent = np.array([[0.49042,-0.05, 0.204921], [0.584064, 0.306914, 0.108106], [0.850856, 0.022013, -0.0699648]])


pickle_in = open("BackCalib2019-6-20.pickle","rb")
labelIDsUni = pickle.load(pickle_in)
print(labelIDsUni)
XlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([4,8,13])-1].T, axis = 1)
YlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([4,8,13])-1].T, axis = 1)
ZlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([4,8,13])-1].T, axis =1)
numElemLabel =  pickle.load(pickle_in)

XYZref = np.hstack((XlabelUni,YlabelUni,ZlabelUni))


C_curr = rmsd.centroid (XYZcurrent)
C_ref = rmsd.centroid (XYZref)

XYZcurr_centered = XYZcurrent - C_curr  #XYZ current after centering 
XYZref_centered = XYZref - C_ref

R = rmsd.kabsch(XYZcurr_centered, XYZref_centered) # the optimal rotation matrix to rotate XYZcurrent to XYZref



#XYZToBeTransformed = np.array([0.7853, 0.5651, 0.0606])
XYZToBeTransformed = np.array([0.533068, -0.418022, -0.0693811])

XYZTransformed = np.matmul(XYZToBeTransformed - C_curr, R) + C_ref  
print(XYZTransformed)