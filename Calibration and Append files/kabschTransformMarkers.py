# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:04:13 2019

@author: mfm160330
"""


import csv
import numpy as np
import pickle
import rmsd

XYZcurrent = np.array([[0.5661,0.0327864, 0.243528], [0.617012, 0.388014, 0.0854373], [0.691728, -0.313644, 0.0731137], [0.787969, -0.321781, -0.00476555], [0.904381, -0.445258, -0.1777]])


pickle_in = open("../calibPickleFiles/MarkersAppended2019-6-20BackStandard.pickle","rb")
labelIDsUni = pickle.load(pickle_in)
print(labelIDsUni)
XlabelUni = pickle.load(pickle_in)[np.array([0,26,4,10,14])]#np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([0,26,4,10,14])-1].T, axis = 1)
YlabelUni = pickle.load(pickle_in)[np.array([0,26,4,10,14])]#np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([0,26,4,10,14])-1].T, axis = 1)
ZlabelUni = pickle.load(pickle_in)[np.array([0,26,4,10,14])]#np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([0,26,4,10,14])-1].T, axis =1)
numElemLabel =  pickle.load(pickle_in)

XYZref = np.vstack((XlabelUni,YlabelUni,ZlabelUni)).T


C_curr = rmsd.centroid (XYZcurrent)
C_ref = rmsd.centroid (XYZref)

XYZcurr_centered = XYZcurrent - C_curr  #XYZ current after centering 
XYZref_centered = XYZref - C_ref

R = rmsd.kabsch(XYZcurr_centered, XYZref_centered) # the optimal rotation matrix to rotate XYZcurrent to XYZref



#XYZToBeTransformed = np.array([0.7853, 0.5651, 0.0606])
XYZToBeTransformed = np.array([0.566071, 0.502468, 0.0572864])

XYZTransformed = np.matmul(XYZToBeTransformed - C_curr, R) + C_ref  
print(XYZTransformed)