# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:42:28 2019

This file is used to calibrate the cont. gaze data and transform them from back current to back reference

@author: mfm160330
"""
import csv
import pickle
import numpy as np
import pandas as pd
from math import pi, cos, sin, isnan

def parse_com(row):
    return (row["Xcom"], row["Ycom"], row["Zcom"])


def parse_target(row):
    return (row["Xtarget"]+row["Xcom"], row["Ytarget"]+row["Ycom"], row["Ztarget"]+row["Zcom"]) 

def parse_kabsch_transform_dump(dump_path):
    transform = open(dump_path, "rb")

    rot = pickle.load(transform)
    print('rotation = ',rot,'\n')
    c_curr = pickle.load(transform)
    print('c_curr = ',c_curr,'\n')
    c_goal = pickle.load(transform)
    print('c_goal = ',c_goal,'\n')


    return (rot, c_curr, c_goal)

def apply_kabsch_tfm(vec, transform):
    (rot, c_curr, c_goal) = transform

    return np.matmul(rot, vec - c_curr) + c_goal

def ConvertToSpherical_np(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[0]**2 + xyz[1]**2
    ptsnew[0] = np.sqrt(xy + xyz[2]**2)
    # for elevation angle defined from Z-axis down
    ptsnew[1] = np.arctan2(np.sqrt(xy), xyz[2])
    # ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[2] = np.arctan2(xyz[1], xyz[0])
    return ptsnew


IdDirectoryPath = "G:/ContGazeImages/FaceAndEyes/CFE2019-7-19"
TransformationMatrixPath = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/ReCalibrationOutputsCorrectCont/2019-7-19/cg/output/Back_Curr_Ref.pickle" #"C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D2019-7-19/ContGaze/KabaschRotTransCont.pickle" 

IDFileInput = IdDirectoryPath +"/"+ "AnglesIDfileCurrBack.csv"  #input ID file calibrated w.r.t current back
IDFileOutput = IdDirectoryPath +"/"+ "AnglesIDfile.csv" # output ID file calibrated w.r.t. reference back


#### load the rotation and translation matrices  #
back_back_tfm = parse_kabsch_transform_dump(TransformationMatrixPath)


# ========= Read the input ID file ==============#
IDs = pd.read_csv(IDFileInput, sep='\t')
numRowsInput = IDs.shape[1]


#=====
for i, row in IDs.iterrows():
    tgt = parse_target(row)
    com = parse_com(row)
    target_back_ref = apply_kabsch_tfm(tgt, back_back_tfm)
    com_back_ref = apply_kabsch_tfm(com, back_back_tfm)
    gaze_vector = np.subtract(target_back_ref, com_back_ref)
    gaze_angles = ConvertToSpherical_np(gaze_vector)
    
    (rho, elev, azim) = gaze_angles
    

    IDs.ix[i, "Rho"] = gaze_angles[0]
    IDs.ix[i, "Elev"] = gaze_angles[1]
    IDs.ix[i, "Azim"] = gaze_angles[2]

    IDs.ix[i, "Xcom"] = com_back_ref[0]
    IDs.ix[i, "Ycom"] = com_back_ref[1]
    IDs.ix[i, "Zcom"] = com_back_ref[2]

    IDs.ix[i, "Xtarget"] = target_back_ref[0]
    IDs.ix[i, "Ytarget"] = target_back_ref[1]
    IDs.ix[i, "Ztarget"] = target_back_ref[2]


#####=== Write to output ID file =======####
IDs.to_csv(IDFileOutput,sep='\t', index=False, na_rep = "nan")



