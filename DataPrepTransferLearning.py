# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:37:37 2019

@author: mfm160330
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:12:52 2019

@author: mfm160330
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from random import shuffle 
import pandas as pd
import csv
import cv2
import os

#train_data_dir = "data/train"
#validation_data_dir = "data/val"


# ===== A function that takes the batch IDs as inputs, extract images preprocess them and returns numpy arrays for face, left eye and right eye ===#
def MyPrepareData (batch_IDs):

    training_data =[]    
    for row in batch_IDs:#DataSet, ID, label in batch_IDs2:
        DataSet = row[0]
        ID = row[1]
        label = row[2]
        class_num = Categories.index(label) 
        DataDir = ReadLoc+'/'+DataSet+'/'+label
        Face_array = cv2.imread(os.path.join(DataDir,'Face','F'+ID) )  # convert to array                
        Left_array = cv2.imread(os.path.join(DataDir,'Leye','L'+ID) ) 
        Right_array = cv2.imread(os.path.join(DataDir,'Reye','R'+ID) ) 
        rFace_array = cv2.resize(Face_array, (FaceResize, FaceResize))/255  # resize to normalize data size and rescale it
        rLeft_array = cv2.resize(Left_array, (EyeResize, EyeResize))/255  
        rRight_array = cv2.resize(Right_array, (EyeResize, EyeResize))/255 
        
        training_data.append([rFace_array, rLeft_array, rRight_array, class_num])  # add this to our training_data
     
    XFace = []
    XLEye = []
    XREye = []
    y_flr = []
    for FaceFeatures, LeftFeatures, RightFeatures, class_num in training_data:
        XFace.append (FaceFeatures)
        XLEye.append (LeftFeatures)
        XREye.append (RightFeatures)
        y_flr.append (class_num)

    XFace = np.array(XFace).reshape(-1,FaceResize,FaceResize,3)
    XLEye = np.array(XLEye).reshape(-1,EyeResize,EyeResize,3)
    XREye = np.array(XREye).reshape(-1,EyeResize,EyeResize,3)
    return XFace,y_flr         


# ==========  data generator function: yields batches of trainning data  ========== #
def MydataGeneratorTest(PathIDs, batch_size, samples_per_epoch):
    
    counter = 0
    number_of_batches = samples_per_epoch/batch_size
    
    #while True: #generators for keras must be infinite
    batch_IDs = PathIDs[counter*batch_size : (counter+1)*batch_size ]
    X_batch, y_batch = MyPrepareData(batch_IDs)
    counter += 1
    return X_batch, y_batch
        
    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0

# ==========  data generator function: yields batches of trainning data  ========== #
def MydataGenerator(PathIDs, batch_size, samples_per_epoch):
    
    counter = 0
    number_of_batches = samples_per_epoch/batch_size
    
    while True: #generators for keras must be infinite
        batch_IDs = PathIDs[counter*batch_size : (counter+1)*batch_size ]
        X_batch, y_batch = MyPrepareData(batch_IDs)
        counter += 1
        yield X_batch, y_batch
        
        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0


#====================================================================================#
                  ######## Main Function #######################
#====================================================================================#

#============= Intilizations ==============#

#==== Data prep. Intializations ======#  
#Categories = ["l", "m"] 
Categories = ["a- 4", "b- 1", "c- 8", "d- 2", "e- 13", "f- 5", "g- 9", "h- 11", "i- 6", "j- 20", "k- 19", "l- 18", "m- 21", "n- 17", "o- 16", "p- 14", "q- 3", "r- 7", "s- 10", "t- 12" ,"u- 15" ] 
FaceResize = 224
EyeResize = 224

#===== Training Intializations =======#
Epochs = 10#300  
LayersToFreeze = 19
numTestSam = 416+19 #412#286#25 
MyBatchSize = 32 
ValSize = 96
lRate = 0.001
ReadLoc = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/FaceAndEyes"
DataSets = ["FE2018-12-1", "FE2018-10-14", "FE2018-12-3"]
idFileName = 'id.csv' #'id.csv' 
augmentFlag = 1

#====== read ID file, Shuffle it, create pathes for train and test data sets =========#
IDs = []
for IDpath in DataSets:
    idFile = ReadLoc+'/'+IDpath+'/'+idFileName
    with open(idFile, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if not ''.join(row).strip():
                continue # ignore the blank lines
            IDs.append(row)
        
shuffle(IDs)
#MydataGeneratorTest(IDs, 32, len(IDs))
samples_per_epoch = len(IDs) - numTestSam - ValSize# number of trainning samples
TrainIDs = IDs[:samples_per_epoch]
ValIDs = IDs[samples_per_epoch:samples_per_epoch+ValSize]
TestIDs = IDs[samples_per_epoch+ValSize:]
numTestSamples =len(TestIDs)
#====================================#

#============= Train and test data generators ========================# 
Test = MydataGeneratorTest(TrainIDs, MyBatchSize, samples_per_epoch)
train_datagen = MydataGenerator(TrainIDs, MyBatchSize, samples_per_epoch)
Val_generator = MydataGenerator(ValIDs, MyBatchSize, ValSize)
#x, y = next(train_datagen)  ## for testing purposes
#test_datagen = MydataGenerator(TestIDs, MyBatchSize, numTestSamples)



#============== Create the face Network ==============================#
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (FaceResize, FaceResize, 3))

# Freeze the layers which you don't want to train
for layer in model.layers[:LayersToFreeze]:
    layer.trainable = False

modelOut = model.output
modelOut = Flatten()(modelOut)
modelOut = Dense(1024, activation="relu")(modelOut)
modelOut = Dropout(0.5)(modelOut)
modelOut = Dense(1024, activation="relu")(modelOut)
predictions = Dense(21, activation="softmax")(modelOut)

model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizers.Adam(lr=lRate), metrics=["accuracy"])
print(model_final.summary())

#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
StepsPerEpoch = np.ceil(samples_per_epoch/MyBatchSize)
#model_final.fit_generator( train_datagen, steps_per_epoch = StepsPerEpoch, epochs = Epochs,  verbose=1, nb_val_samples = ValSize, callbacks = [checkpoint])
#model_final.fit_generator( train_datagen, steps_per_epoch = StepsPerEpoch, epochs = Epochs,  verbose=1, nb_val_samples = ValSize)
model_final.fit_generator( train_datagen, steps_per_epoch = StepsPerEpoch, epochs = Epochs,  verbose=1, validation_data = Val_generator, nb_val_samples = ValSize)

####train_generator = train_datagen.flow(Xtrain,Ytrain)
#===== To evaluate the accuracy on this data after each epoch =====#
#Xval = Xtest[:ValSize,:,:,:]
#Yval = Ytest[:ValSize]
#Val_generator = test_datagen.flow(Xval,Yval) 

Xtest, Ytest = MyPrepareData (TestIDs)
Ypred= model_final.predict(Xtest)
Ypred = np.argmax(Ypred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(Ytest, Ypred))
print('Classification Report')
print(classification_report(Ytest, Ypred, target_names=Categories))

count = 0
for i, j in zip(Ytest, Ypred):
    if i == j:
        count = count+1
Accuracy = count/len(Ypred)
print('Accuracy = ', Accuracy)