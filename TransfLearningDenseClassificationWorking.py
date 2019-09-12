# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:30:24 2019

@author: mfm160330
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.utils.vis_utils import plot_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,  AveragePooling2D
from keras import backend as K 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from random import shuffle 
import pandas as pd
import csv
import cv2
import os
import shelve

#train_data_dir = "data/train"
#validation_data_dir = "data/val"

def customLoss(yTrue,yPred):
    return K.sum(K.log(yTrue) - K.log(yPred))



# implementation of my modified MMSE loss function
def MyModMmseLoss (angleTrue, anglePred):
    ElevPredict = np.array([0.5, 0.6, 0.1,0.4])
    angleTrue = 1
    
    
    return 0

# ===== A function that takes the batch IDs as inputs, extract images preprocess them and returns a numpy array and their Labels ===#
def MyPrepareData (batch_IDs):

    X_Face, X_LEye, X_REye = [], [], [] 
    y = []
    y_Elev, y_Azim = [], [], 
    for DataSetID, ImagePath, ImageID, ElevClass, AzimClass in batch_IDs:
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
        y.append([ElevClass, AzimClass])
        
    X_Face = np.array(X_Face).reshape(-1,FaceResize,FaceResize,3)
    X_LEye = np.array(X_LEye).reshape(-1,EyeResize,EyeResize,3)
    X_REye = np.array(X_REye).reshape(-1,EyeResize,EyeResize,3)


    return X_Face, y_Elev, y_Azim        


# ==========  data generator function: yields batches of trainning data  ========== #
def MydataGeneratorTest(PathIDs, batch_size, samples_per_epoch):
    
    counter = 0
    number_of_batches = np.ceil(samples_per_epoch/batch_size)
    
    #while True: #generators for keras must be infinite
    batch_IDs = PathIDs[counter*batch_size : (counter+1)*batch_size ]
    X_batch, y_Elev, y_Azim = MyPrepareData(batch_IDs)
    counter += 1
    return X_batch, [y_Elev, y_Azim]
        
    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0

# ==========  data generator function: yields batches of trainning data  ========== #
def MydataGenerator(PathIDs, batch_size, samples_per_epoch):
    
    counter = 0
    number_of_batches = samples_per_epoch/batch_size
    
    while True: #generators for keras must be infinite
        batch_IDs = PathIDs[counter*batch_size : (counter+1)*batch_size ]
        X_batch, y_Elev, y_Azim = MyPrepareData(batch_IDs)
        counter += 1
        yield X_batch, [y_Elev, y_Azim]#{'ElevPredict': y_batch_Elev, 'AzimuthPredict': y_batch_Azim} #y_batch_Elev, y_batch_Azim
        
        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0


# =========== Shelving function: to save all variables ==================#
def MyShelf(ShelveFilename):
    my_shelf = shelve.open(ShelveFilename,'n') # 'n' for new
    for key in dir():
        try:
            my_shelf[key] = globals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
    my_shelf.close()





#====================================================================================#
                  ######## Main Function #######################
#====================================================================================#

#============= Intilizations ==============#


#==== Data prep. Intializations ======#  
#Categories = ["a- 4", "b- 1", "c- 8", "d- 2", "e- 13", "f- 5", "g- 9", "h- 11", "i- 6", "j- 20", "k- 19", "l- 18", "m- 21", "n- 17", "o- 16", "p- 14", "q- 3", "r- 7", "s- 10", "t- 12" ,"u- 15" ] 

#======= File Pathes intializations =======#
idFile = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseClassification.csv'
ShelveFilename = 'variables/run1.out'
###======================================###

FaceResize = 224
EyeResize = 224

#==== Dense classificiation Parameters ======#
numElevClasses = 20 #number of Elevation Angles classes, 1) theta<=-45 2) -45<theta<=-43 3) -43<theta<=-41 .... 47) 45<theta
numAzimClasses = 52 #number of Azimuth Angles classes, 1) phi<=-90 2) -90<phi<=-88 3) -43<theta<=-41 .... 92) 90<phi

#===== Training Intializations =======#
Epochs = 5#300  
LayersToFreeze = 19
numTestSam = 416+19#412#286#25 
MyBatchSize = 32 
ValSize = 96
lRate = 0.001
#====== read ID file, Shuffle it, create pathes for train and test data sets =========#
IDs = []

with open(idFile, "r") as csvfile:
   readCSV = csv.reader(csvfile, delimiter='\t')
   next(csvfile) #skip heading
   for row in readCSV:
       if not ''.join(row).strip():
           continue # ignore the blank lines
       IDs.append(row)
        
shuffle(IDs)
samples_per_epoch = len(IDs) - numTestSam - ValSize # number of trainning samples
TrainIDs = IDs[:samples_per_epoch]
ValIDs = IDs[samples_per_epoch:samples_per_epoch+ValSize]
TestIDs = IDs[samples_per_epoch+ValSize:]
numTestSamples = len(TestIDs)
#====================================#

#============= Train and test data generators ========================# 
X_batch_test, [y_Elev_test, y_Azim_test] = MydataGeneratorTest(TrainIDs, MyBatchSize, samples_per_epoch)
train_datagen = MydataGenerator(TrainIDs, MyBatchSize, samples_per_epoch)
Val_generator = MydataGenerator(ValIDs, MyBatchSize, ValSize)
#x, y = next(train_datagen)  ## for testing purposes



#============== Create the face Network ==============================#
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (FaceResize, FaceResize, 3))

# Freeze the layers which you don't want to train
for layer in model.layers[:LayersToFreeze]:
    layer.trainable = False

modelOut = model.output
modelOut = GlobalAveragePooling2D()(modelOut)

ElevBranch = Dense(1024, activation="relu")(modelOut) # Elevation Angles head
ElevPredict = Dense(numElevClasses, activation="softmax")(ElevBranch)

AzimBranch = Dense(1024, activation="relu")(modelOut) # Azimuth Angles head
AzimuthPredict = Dense(numAzimClasses, activation="softmax")(AzimBranch)

model_final = Model(input = model.input, output = [ElevPredict, AzimuthPredict])
plot_model(model_final, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# use "sparse_categorical_crossentropy" when your output is not one hot vectors 
model_final.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizers.Adam(lr=lRate), metrics=["accuracy"])
#model_final.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr=lRate), metrics=["accuracy"])

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

X_test, y_Elev_truth, y_Azim_truth = MyPrepareData (TestIDs) #test values
y_Elev_truth = list(map(float, y_Elev_truth))
y_Azim_truth = list(map(float, y_Azim_truth))
[y_Elev_soft, y_Azim_soft] = model_final.predict(X_test)
#print(Ypred)
y_Elev_pred = np.argmax(y_Elev_soft, axis=1)
y_Azim_pred = np.argmax(y_Azim_soft, axis=1)

print('Elevation Confusion Matrix')
print(confusion_matrix(y_Elev_truth, y_Elev_pred))

print('Azimuth Confusion Matrix')
print(confusion_matrix(y_Azim_truth, y_Azim_pred))

#print('Classification Report')
#print(classification_report(Ytest, Ypred, target_names=Categories))

## Elevation Accuracy (highest one)
count1 = 0
for i1, j1 in zip(y_Elev_truth, y_Elev_pred):
    if i1 == j1:
        count1 = count1 + 1
ElevAccuracy = count1/len(y_Elev_pred)
print('Elevation Accuracy = ', ElevAccuracy, "\n")

## Azimuth Accuracy
count2 = 0
for i2, j2 in zip(y_Azim_truth, y_Azim_pred):
    if i2 == j2:
        count2 = count2 + 1
AzimAccuracy = count2/len(y_Azim_pred)
print('Azimut Accuracy = ', AzimAccuracy, "\n")      


## Elevation Accuracy for double resolution
y_Elev_predSorted =  np.flip(np.argsort(y_Elev_soft, axis =1), axis=1)
Y_Elev_double_res = y_Elev_predSorted[:,:2]

count3 = 0
for i3, j3, k3 in zip(y_Elev_truth, Y_Elev_double_res[:,0], Y_Elev_double_res[:,1]):
    if (i3 == j3) or (i3 == k3):
        count3 = count3 + 1
Elev_acc_2 = count3/len(y_Elev_pred)
print("Elevation Accuracy double resolution = ", Elev_acc_2, "\n")
  

## Azimuth Accuracy for double resolution
y_Azim_predSorted =  np.flip(np.argsort(y_Azim_soft, axis =1), axis=1)
Y_Azim_double_res = y_Azim_predSorted[:,:2]

count4 = 0
for i4, j4, k4 in zip(y_Azim_truth, Y_Azim_double_res[:,0], Y_Azim_double_res[:,1]):
    if (i4 == j4) or (i4 == k4):
        count4 = count4 + 1
Azim_acc_2 = count4/len(y_Azim_pred)
print("Azimuth Accuracy double resolution = ", Azim_acc_2, "\n")



## ======= Shelving All variables ====== ####
MyShelf(ShelveFilename)




