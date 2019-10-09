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
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,  AveragePooling2D, Concatenate, concatenate
from keras import backend as K 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from random import shuffle 
import pandas as pd
import csv
import cv2
import os
#import shelve
import numpy.matlib

#train_data_dir = "data/train"
#validation_data_dir = "data/val"

def customLoss(yTrue,yPred):
    return K.sum(K.log(yTrue) - K.log(yPred))

#==== A function that compute softmax values for each sets of scores in x ======#
def Mysoftmax(x):
    e_x = np.exp(x)
    ex_sum = np.sum(e_x, axis = 1)
    ex_sum_repeated = numpy.matlib.repmat(ex_sum,x.shape[1],1).T
    return e_x / ex_sum_repeated 

#======== A function that takes a list and maps it to one hot encoding =============#
def MyOneHotEncode(y,numClasses):
    y = list(map(float, y))
    y = np.asarray(y, dtype = int)
    y_OH = np.zeros((y.shape[0], numClasses)) #one hot encoded output 
    y_OH[np.arange(y.shape[0]), y] = 1
    return y_OH

#====== A function that soft encodes true labels using square difference ==========#
def MySoftEncode(y,numClasses):
    y = np.asarray(y, dtype = float)
    r_i = np.arange(numClasses)
    y_repeated = numpy.matlib.repmat(y, numClasses, 1).T
    r_i_repeated = numpy.matlib.repmat(r_i,y.shape[0],1)

    ### Square Difference
    #SquareDiff = -np.square(y_repeated - r_i_repeated)
    #y_soft = Mysoftmax(SquareDiff)
    
    ### Absoloute Difference
    AbsDiff = -np.absolute(y_repeated - r_i_repeated)
    y_soft = Mysoftmax(AbsDiff)
    
    ## Square log difference
    #sqLogDiff = -np.square(np.log2(y_repeated+1)-np.log2(r_i_repeated+1))  
    #y_soft = Mysoftmax(sqLogDiff)
    
    return y_soft


# ===== A function that takes the batch IDs as inputs, extract images preprocess them and returns a numpy array and their Labels ===#
def MyPrepareData (batch_IDs):

    X_Face, X_LEye, X_REye = [], [], [] 
    y_Elev, y_Azim = [], [], 
    for DataSetID, ImagePath, ImageID, ElevClass, AzimClass,_,_ in batch_IDs:
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


# ==========  data generator function: yields batches of trainning data  ========== #
def MydataGeneratorTest(PathIDs, batch_size, samples_per_epoch):
    
    counter = 0
    number_of_batches = np.ceil(samples_per_epoch/batch_size)
    
    #while True: #generators for keras must be infinite
    batch_IDs = PathIDs[counter*batch_size : (counter+1)*batch_size ]
    X_F_batch, X_R_batch, X_L_batch, y_Elev, y_Azim = MyPrepareData(batch_IDs)
    counter += 1
    y_Elev_OH = MyOneHotEncode(y_Elev, numElevClasses)
    y_Azim_OH = MyOneHotEncode(y_Azim, numAzimClasses)
    
    if softLabels == 1:
        y_Elev_OH = MySoftEncode(y_Elev, numElevClasses) # hard one hot encoding
        y_Azim_OH = MySoftEncode(y_Azim, numAzimClasses) 
    else:
        y_Elev_OH = MyOneHotEncode(y_Elev, numElevClasses) # soft one hot encoding
        y_Azim_OH = MyOneHotEncode(y_Azim, numAzimClasses) 
        
    
    return [X_F_batch, X_R_batch, X_L_batch], [y_Elev_OH, y_Azim_OH]
        
    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0

# ==========  data generator function: yields batches of trainning data  ========== #
def MydataGenerator(PathIDs, batch_size, samples_per_epoch):
    
    counter = 0
    number_of_batches = samples_per_epoch/batch_size
    
    while True: #generators for keras must be infinite
        batch_IDs = PathIDs[counter*batch_size : (counter+1)*batch_size ]
        X_F_batch, X_R_batch, X_L_batch, y_Elev, y_Azim = MyPrepareData(batch_IDs)
        counter += 1
        if softLabels == 1:
            y_Elev_OH = MySoftEncode(y_Elev, numElevClasses) # hard one hot encoding
            y_Azim_OH = MySoftEncode(y_Azim, numAzimClasses) 
        else:
            y_Elev_OH = MyOneHotEncode(y_Elev, numElevClasses) # soft one hot encoding
            y_Azim_OH = MyOneHotEncode(y_Azim, numAzimClasses) 
        
        yield [X_F_batch, X_R_batch, X_L_batch], [y_Elev_OH, y_Azim_OH]#{'ElevPredict': y_batch_Elev, 'AzimuthPredict': y_batch_Azim} #y_batch_Elev, y_batch_Azim
        
        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0
            shuffle(TrainIDs)


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

# =============Read ID file ============================#
def readIDfile(IDfile):
    IDs = []
    with open(IDfile, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter='\t')
        next(csvfile) #skip heading
        for row in readCSV:
            if not ''.join(row).strip():
                continue # ignore the blank lines
            IDs.append(row)
    
    shuffle(IDs)
    return IDs
    


#====================================================================================#
                  ######## Main Function #######################
#====================================================================================#

#============= Intilizations ==============#


#==== Data prep. Intializations ======#  
#Categories = ["a- 4", "b- 1", "c- 8", "d- 2", "e- 13", "f- 5", "g- 9", "h- 11", "i- 6", "j- 20", "k- 19", "l- 18", "m- 21", "n- 17", "o- 16", "p- 14", "q- 3", "r- 7", "s- 10", "t- 12" ,"u- 15" ] 

#======= File Pathes intializations =======#
TrainIdFile = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/AugmentedNineDownFour.csv'
ValIdFile = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseValCont2019-7-10.csv'
TestIdFile = 'C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/DenseTest2019-7-10And11.csv'
ShelveFilename = 'variables/run12.out'
CheckpointFilePath = 'mySavedModels/run12.h5' 
checkPeriod = 1 #Period of saving weights
###======================================###

FaceResize = 224
EyeResize = 64#224

#==== Dense classificiation Parameters ======#
numElevClasses = 14 #number of Elevation Angles classes, 1) theta<=-45 2) -45<theta<=-43 3) -43<theta<=-41 .... 47) 45<theta
numAzimClasses = 33 #number of Azimuth Angles classes, 1) phi<=-90 2) -90<phi<=-88 3) -43<theta<=-41 .... 92) 90<phi
softLabels = 1 #transform the hard labels into soft ones to penalize errors differently 
IsEyes = 1
#===== Training Intializations =======#
Epochs = 25#300  
LayersToFreeze = 18
MyBatchSize = 32 
ValSize = 1000
lRate = 0.001
#====== read Train, Validataion and test ID file amd Shuffle them =========#
TrainIDs = readIDfile(TrainIdFile)
samples_per_epoch = len(TrainIDs) #- numTestSam - ValSize # number of trainning samples

ValIDs = readIDfile(ValIdFile)
ValIDs = ValIDs[0:ValSize]


TestIDs = readIDfile(TestIdFile)
numTestSam = len(TestIDs)
#====================================#

#============= Train and test data generators ========================# 
#[X_F_batch_test, X_R_batch_test, X_L_batch_test], [y_Elev_test, y_Azim_test] = MydataGeneratorTest(TrainIDs, MyBatchSize, samples_per_epoch)
train_datagen = MydataGenerator(TrainIDs, MyBatchSize, samples_per_epoch)
Val_generator = MydataGenerator(ValIDs, MyBatchSize, ValSize)
#x, y = next(train_datagen)  ## for testing purposes



#============== Create the face Network ==============================#
model_F = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (FaceResize, FaceResize, 3)) ##VGG network for face
model_L = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (EyeResize, EyeResize, 3)) ##VGG network for left Eye
model_R = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (EyeResize, EyeResize, 3)) ##VGG network for right Eye

# change the layers' names in left and right eyes network
for i, layer in enumerate(model_L.layers):
    layer.name = layer.name + '_l'
for i, layer in enumerate(model_R.layers):
    layer.name = layer.name + '_r'

# Freeze the layers which you don't want to train
for layer in model_F.layers[:LayersToFreeze]:
    layer.trainable = False
for layer in model_L.layers[:LayersToFreeze]:
    layer.trainable = False
for layer in model_R.layers[:LayersToFreeze]:
    layer.trainable = False

# Global Average pooling layer at output of three networks
modelOutF = model_F.output
modelOutL = model_L.output
modelOutR = model_R.output
modelOutF = GlobalAveragePooling2D()(modelOutF) 
modelOutL = GlobalAveragePooling2D()(modelOutL)
modelOutR = GlobalAveragePooling2D()(modelOutR)



# combine the output of the 3 branches
modelOut = concatenate([modelOutF, modelOutL, modelOutR])


ElevBranch = Dense(1024, activation="relu")(modelOut) # Elevation Angles head
ElevBranch = Dropout(0.4)(ElevBranch)
ElevPredict = Dense(numElevClasses, activation="softmax")(ElevBranch)

AzimBranch = Dense(1024, activation="relu")(modelOut) # Azimuth Angles head
AzimBranch = Dropout(0.4)(AzimBranch)
AzimuthPredict = Dense(numAzimClasses, activation="softmax")(AzimBranch)

#model_final = Model(inputs = [model_F.input, model_L.input] , outputs = [ElevPredict, AzimuthPredict])
model_final = Model(inputs = [model_F.input, model_L.input, model_R.input] , outputs = [ElevPredict, AzimuthPredict])
#model_final = Model(inputs = [xFace.input, xLEye.input, xREye.input] , outputs = [ElevPredict, AzimuthPredict])
plot_model(model_final, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)

# use "sparse_categorical_crossentropy" when you have a non-encoded output
model_final.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr=lRate), metrics=["accuracy"])

print(model_final.summary())

checkpoint = ModelCheckpoint(CheckpointFilePath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=checkPeriod)


StepsPerEpoch = np.ceil(samples_per_epoch/MyBatchSize)
model_final.fit_generator( train_datagen, steps_per_epoch = StepsPerEpoch, epochs = Epochs,  verbose=1, validation_data = Val_generator, nb_val_samples = ValSize, callbacks= [checkpoint])

####train_generator = train_datagen.flow(Xtrain,Ytrain)
#===== To evaluate the accuracy on this data after each epoch =====#
#Xval = Xtest[:ValSize,:,:,:]
#Yval = Ytest[:ValSize]
#Val_generator = test_datagen.flow(Xval,Yval) 


# ========== Test in batch ============#
num_t = int(np.floor(len(TestIDs)/MyBatchSize)) #number of test iterations
num_t_s = num_t*MyBatchSize #number of actual test samples 
y_Elev_truth, y_Azim_truth = [], []
y_Elev_soft = np.empty([num_t_s,numElevClasses])
y_Azim_soft = np.empty([num_t_s,numAzimClasses])
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

#print('Elevation Confusion Matrix')
#print(confusion_matrix(y_Elev_truth, y_Elev_pred))

#print('Azimuth Confusion Matrix')
#print(confusion_matrix(y_Azim_truth, y_Azim_pred))

#print('Classification Report')
#print(classification_report(Ytest, Ypred, target_names=Categories))

## Elevation And Azimuth Accuracy (highest one)
ElevAccuracy = AccuracyCal(y_Elev_truth, y_Elev_pred)
print('Elevation Accuracy = ', ElevAccuracy, "\n")

AzimAccuracy = AccuracyCal(y_Azim_truth, y_Azim_pred)
print('Azimuth Accuracy = ', AzimAccuracy, "\n")      


## Elevation and Azimuth Accuracy for double resolution
Elev_acc_2 = DoubleResAccuracy(y_Elev_truth, y_Elev_soft)
print("Elevation Accuracy double resolution = ", Elev_acc_2, "\n")
  
Azim_acc_2 = DoubleResAccuracy(y_Azim_truth, y_Azim_soft)
print("Azimuth Accuracy double resolution = ", Azim_acc_2, "\n")

#model_final.save(CheckpointFilePath)

## ======= Shelving All variables ====== ####
#MyShelf(ShelveFilename)




