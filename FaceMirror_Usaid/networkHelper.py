# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:46:19 2020

@author: uxm170001
"""

import pandas as pd
import warnings
import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt

class NetworkHelper:
    
    EyeResize = 64

    #==== Dense classificiation Parameters ======#
    numElevClasses = 15 #number of Elevation Angles classes, 1) theta<=-45 2) -45<theta<=-43 3) -43<theta<=-41 .... 47) 45<theta
    numAzimClasses = 44 #number of Azimuth Angles classes, 1) phi<=-90 2) -90<phi<=-88 3) -43<theta<=-41 .... 92) 90<phi
    softLabels = 1 #transform the hard labels into soft ones to penalize errors differently 
    IsEyes = 1
    
#    def __init__(self, TrainIDs):
#        self.df = NetworkHelper.readIDfiles(TrainIDs)
#        
#    def __len__(self):
#        return self.df.shape[0]
#
#    def __getitem__(self, idx):
#        X_Face_REye_batch, X_Face_LEye_batch, X_Mirror_REye_batch, X_Mirror_LEye_batch, X_Face_Landmarks_batch, X_Mirror_Landmarks_batch, y_Elev, y_Azim = NetworkHelper.prepareData(self.df.loc[[idx]])
#        if NetworkHelper.softLabels == 1:
#            y_Elev_OH = NetworkHelper._softEncode(y_Elev, NetworkHelper.numElevClasses) # hard one hot encoding
#            y_Azim_OH = NetworkHelper._softEncode(y_Azim, NetworkHelper.numAzimClasses) 
#        else:
#            y_Elev_OH = NetworkHelper._oneHotEncode(y_Elev, NetworkHelper.numElevClasses) # soft one hot encoding
#            y_Azim_OH = NetworkHelper._oneHotEncode(y_Azim, NetworkHelper.numAzimClasses) 
#        
#        # return [X_Face_REye_batch, X_Face_LEye_batch, X_Mirror_REye_batch, X_Mirror_LEye_batch, X_Face_Landmarks_batch, X_Mirror_Landmarks_batch], [y_Elev_OH, y_Azim_OH]
#        X = np.concatenate([X_Face_Landmarks_batch, X_Mirror_Landmarks_batch])
#        y = np.concatenate([y_Elev, y_Azim])
#        return X, y
        
    def readIDFiles(TrainIDs):
        # Read all files into Pandas dataframe
        file_list = [pd.read_csv(i) for i in TrainIDs]
        df = pd.concat(file_list, ignore_index=True)
        
        # Drop all rows where the elev/azim is nan
        df.dropna(inplace=True)
        
        # Split the landmarks from X;Y into separate cells for X and Y
        for i in range(68):
            df[['Face' + str(i) + '_X', 'Face' + str(i) + '_Y']] = df['Face' + str(i)].str.split(';', expand=True)
            df.drop('Face' + str(i), axis=1, inplace=True)
            df = df.astype({'Face' + str(i) + '_X' : 'int16', 'Face' + str(i) + '_Y' : 'int16'})
        # Split the mirror/face landmark splitting to keep order of face/mirror columns
        for i in range(68):
            df[['Mirror' + str(i) + '_X', 'Mirror' + str(i) + '_Y']] = df['Mirror' + str(i)].str.split(';', expand=True)
            df.drop('Mirror' + str(i), axis=1, inplace=True)
            df = df.astype({'Mirror' + str(i) + '_Y' : 'int16', 'Mirror' + str(i) + '_Y' : 'int16'})
            
        # Put the elevation/azimuth angles into classes
        elev_classes = [-np.inf]
#        elev_classes.extend([-0.45 + 0.02 * i for i in range(46)]) # Add bins for -45, -43, .., 45
        elev_classes.extend([1 + 0.02 * i for i in range(46)]) # Add bins for -45, -43, .., 45
        elev_classes.append(np.inf)
        df['ElevClass'] = pd.cut(df.Elev, elev_classes, labels=False)
        azim_classes = [-np.inf]
        azim_classes.extend([-0.90 + 0.02 * i for i in range(91)]) # Add bins for -90, -88, ..., 90
        azim_classes.append(np.inf)
        df['AzimClass'] = pd.cut(df.Azim, azim_classes, labels=False)
        
#        hist = df.hist(column=['ElevClass', 'AzimClass', 'Azim', 'Elev'])
#        plt.show()
        # Shuffle the dataframe in-place and return
        df = df.sample(frac=1).reset_index(drop=True)
        return df
    
    def prepareData(train_df):
        X_Face_LEye, X_Face_REye, X_Mirror_LEye, X_Mirror_REye, X_Face_Landmarks, X_Mirror_Landmarks = [], [], [], [], [], []
        y_Elev, y_Azim = [], []
        
        for index, row in train_df.iterrows():
            # DatasetID is in form 12/1/2018 for some reason, need to change to 2018-12-1
            date_nums = row['DatasetID'].split('/')
            subject = date_nums[2] + '-' + date_nums[0] + '-' + date_nums[1]
            face_path = 'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping ' + subject + '/Face/' + row['Label']
            mirror_path = 'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping ' + subject + '/Mirror/' + row['Label']
            
            # Get eyes from corresponding directories
            face_right = cv2.imread(face_path + '/Reye/' + row['FaceImageID'])
            face_left = cv2.imread(face_path + '/Leye/' + row['FaceImageID'])
            # The mirror images have the face image IDs - this is a point of confusion that should be fixed
            mirror_right = cv2.imread(mirror_path + '/Reye/' + row['FaceImageID'])
            mirror_left = cv2.imread(mirror_path + '/Leye/' + row['FaceImageID'])
            
            # Check if eye images were not found, if so print a warning and skip
            if(face_right is None) or (face_left is None) or (mirror_right is None) or (mirror_left is None):
                warnings.warn('Missing image for subject {} with image {} or {} on line {} with path: {}'.format(subject, row['FaceImageID'], row['MirrorImageID'], index, face_path))
                continue
            
            # Normalize, resize eye images and add to lists
            X_Face_REye.append(cv2.resize(face_right, (NetworkHelper.EyeResize, NetworkHelper.EyeResize)) / 255)
            X_Face_LEye.append(cv2.resize(face_left, (NetworkHelper.EyeResize, NetworkHelper.EyeResize)) / 255)
            X_Mirror_REye.append(cv2.resize(mirror_right, (NetworkHelper.EyeResize, NetworkHelper.EyeResize)) / 255)
            X_Mirror_LEye.append(cv2.resize(mirror_left, (NetworkHelper.EyeResize, NetworkHelper.EyeResize)) / 255)
            
            # Add face, mirror landmarks
            X_Face_Landmarks.append(row.values[-274:-138].astype(np.int16))
            X_Mirror_Landmarks.append(row.values[-138:-2].astype(np.int16))
            
            # Add elev/azim classes to lists
            y_Elev.append(int(row['ElevClass']))  # Converting 0.45 to 45 angle
            y_Azim.append(int(row['AzimClass']))
            
        # Convert image lists to NumPY arrays
        X_Face_REye = np.array(X_Face_REye).reshape(-1, NetworkHelper.EyeResize, NetworkHelper.EyeResize, 3)
        X_Face_LEye = np.array(X_Face_LEye).reshape(-1, NetworkHelper.EyeResize, NetworkHelper.EyeResize, 3)
        X_Mirror_REye = np.array(X_Mirror_REye).reshape(-1, NetworkHelper.EyeResize, NetworkHelper.EyeResize, 3)
        X_Mirror_LEye = np.array(X_Mirror_LEye).reshape(-1, NetworkHelper.EyeResize, NetworkHelper.EyeResize, 3)
        X_Face_Landmarks = np.stack(X_Face_Landmarks)
        X_Mirror_Landmarks = np.stack(X_Mirror_Landmarks)
        
        return X_Face_REye, X_Face_LEye, X_Mirror_REye, X_Mirror_LEye, X_Face_Landmarks, X_Mirror_Landmarks, y_Elev, y_Azim
    
    def dataGenerator(train_df, batch_size, samples_per_epoch):
        counter = 0
        number_of_batches = samples_per_epoch / batch_size
        
        while True: # Generators for Keras need to be infinite
            batch = train_df[counter * batch_size : (counter + 1) * batch_size]
            X_Face_REye_batch, X_Face_LEye_batch, X_Mirror_REye_batch, X_Mirror_LEye_batch, X_Face_Landmarks_batch, X_Mirror_Landmarks_batch, y_Elev, y_Azim = NetworkHelper.prepareData(batch)
            counter += 1
            if NetworkHelper.softLabels == 1:
                y_Elev_OH = NetworkHelper._softEncode(y_Elev, NetworkHelper.numElevClasses) # hard one hot encoding
                y_Azim_OH = NetworkHelper._softEncode(y_Azim, NetworkHelper.numAzimClasses) 
            else:
                y_Elev_OH = NetworkHelper._oneHotEncode(y_Elev, NetworkHelper.numElevClasses) # soft one hot encoding
                y_Azim_OH = NetworkHelper._oneHotEncode(y_Azim, NetworkHelper.numAzimClasses) 
            
            X_Landmarks = np.concatenate([X_Face_Landmarks_batch, X_Mirror_Landmarks_batch], axis=1)
#            y = np.concatenate([y_Elev_OH, y_Azim_OH], axis=1)
#            yield [X_Face_REye_batch, X_Face_LEye_batch, X_Mirror_REye_batch, X_Mirror_LEye_batch, X_Face_Landmarks_batch, X_Mirror_Landmarks_batch], [y_Elev_OH, y_Azim_OH]
            yield [X_Face_REye_batch, X_Face_LEye_batch, X_Mirror_REye_batch, X_Mirror_LEye_batch, X_Landmarks], [y_Elev_OH, y_Azim_OH]
#            yield [X_Face_REye_batch, X_Face_LEye_batch, X_Mirror_REye_batch, X_Mirror_LEye_batch], [y_Elev_OH, y_Azim_OH]
            
            # restart counter to yeild data in the next epoch as well
            if counter >= number_of_batches:
                counter = 0
                train_df = train_df.sample(frac=1).reset_index(drop=True)
                
    # ========== Accuracy calculation function ====================== #
    def AccuracyCal(y_truth, y_pred): 
        count1 = 0
        for i1, j1 in zip(y_truth, y_pred):
            if i1 == j1:
                count1 = count1 + 1
            else:
                print(i1, j1)
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
        print(count3, y_truth)
        print(count3, y_pred_soft)
        Acc_2 = count3/len(y_truth)
        return Acc_2
                
    #======== A function that takes a list and maps it to one hot encoding =============#
    def _oneHotEncode(y,numClasses):
        y = list(map(float, y))
        y = np.asarray(y, dtype = int)
        y_OH = np.zeros((y.shape[0], numClasses)) #one hot encoded output   
        y_OH[np.arange(y.shape[0]), y] = 1
        return y_OH
    
    #====== A function that soft encodes true labels using Absoloute difference ==========#
    def _softEncode(y,numClasses):
        y = np.asarray(y, dtype = float)
        r_i = np.arange(numClasses)
        y_repeated = np.matlib.repmat(y, numClasses, 1).T
        r_i_repeated = np.matlib.repmat(r_i,y.shape[0],1)
    
        ### Square Difference
        #SquareDiff = -np.square(y_repeated - r_i_repeated)
        #y_soft = Mysoftmax(SquareDiff)
        
        ### Absoloute Difference
        AbsDiff = -np.absolute(y_repeated - r_i_repeated)
        y_soft = NetworkHelper._softmax(AbsDiff)
        
        ## Square log difference
        #sqLogDiff = -np.square(np.log2(y_repeated+1)-np.log2(r_i_repeated+1))  
        #y_soft = Mysoftmax(sqLogDiff)
        
        return y_soft
    
    def _softmax(x):
        e_x = np.exp(x)
        ex_sum = np.sum(e_x, axis = 1)
        ex_sum_repeated = numpy.matlib.repmat(ex_sum,x.shape[1],1).T
        return e_x / ex_sum_repeated
            