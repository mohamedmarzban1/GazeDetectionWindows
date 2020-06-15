"""
Created on Sunday May 10 2020

@author: Usaid Malik
"""

from networkHelper import NetworkHelper
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, InputLayer, GlobalAveragePooling2D, AveragePooling2D, Concatenate, concatenate, Flatten
from keras import Sequential, Model, Input
from keras import optimizers
import numpy as np
from keras_vggface import VGGFace

#===== Training Intializations =======#
Epochs = 50
LayersToFreeze_F = 25   #N.B: VGGface without the top includes 31 layers
LayersToFreeze_E = 18  
MyBatchSize = 32 
ValSize = 1702
lRate = 0.001

# File Path initializations
TrainIDFiles = ['D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2018-12-1/Mapping 2018-12-1.csv']
ValidIDFiles = ['D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2018-12-1/Mapping 2018-12-1.csv']
TestIDFiles = ['D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2018-12-1/Mapping 2018-12-1.csv']

#TrainIDFiles = [
#        'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2018-12-1/Mapping 2018-12-1.csv',
#        'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2019-6-11/Mapping 2019-6-11.csv',
#        'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2019-6-14/Mapping 2019-6-14.csv',
#        'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2019-7-9/Mapping 2019-7-9.csv',
#        'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2019-7-10/Mapping 2019-7-10.csv',
#        'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2019-7-15/Mapping 2019-7-15.csv',
#        'D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2019-7-23/Mapping 2019-7-23.csv'
#        ]
#ValidIDFiles = ['D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2019-5-22/Mapping 2019-5-22.csv']
#TestIDFiles = ['D:/EyeMapping_WithAngles/DiscreteLocationMapping/Mapping 2019-5-30/Mapping 2019-5-30.csv']

TrainIDs = NetworkHelper.readIDFiles(TrainIDFiles)
samples_per_epoch = TrainIDs.shape[0]

ValidIDs = NetworkHelper.readIDFiles(ValidIDFiles)
ValidIDs = ValidIDs.head(ValSize)

TestIDs = NetworkHelper.readIDFiles(TestIDFiles)
numTestSum = TestIDs.shape[0]

# Train and test data generators
train_datagen = NetworkHelper.dataGenerator(TrainIDs, MyBatchSize, samples_per_epoch)
test_datagen = NetworkHelper.dataGenerator(ValidIDs, MyBatchSize, ValSize)

# For now, try training with just landmarks and elevation/azimuth to see what happens

model_M_L = VGGFace(include_top=False, input_shape=(NetworkHelper.EyeResize, NetworkHelper.EyeResize, 3))
model_M_R = VGGFace(include_top=False, input_shape=(NetworkHelper.EyeResize, NetworkHelper.EyeResize, 3))
model_F_L = VGGFace(include_top=False, input_shape=(NetworkHelper.EyeResize, NetworkHelper.EyeResize, 3))
model_F_R = VGGFace(include_top=False, input_shape=(NetworkHelper.EyeResize, NetworkHelper.EyeResize, 3))

# change the layers' names in mirror/face left and right eyes network
for i, layer in enumerate(model_M_L.layers):
    layer.name = layer.name + '_m_l'
for i, layer in enumerate(model_M_R.layers):
    layer.name = layer.name + '_m_r' 
for i, layer in enumerate(model_F_L.layers):
    layer.name = layer.name + '_f_l'
for i, layer in enumerate(model_F_R.layers):
    layer.name = layer.name + '_f_r' 

# Freeze layers that are not needed to train
for layer in model_M_L.layers[:LayersToFreeze_E]:
    layer.trainable = False
for layer in model_M_R.layers[:LayersToFreeze_E]:
    layer.trainable = False
for layer in model_F_L.layers[:LayersToFreeze_E]:
    layer.trainable = False
for layer in model_F_R.layers[:LayersToFreeze_E]:
    layer.trainable = False
    
last_layer_M_L = model_M_L.get_layer('conv5_3_m_l').output
last_layer_M_R = model_M_R.get_layer('conv5_3_m_r').output
last_layer_F_L = model_F_L.get_layer('conv5_3_f_l').output
last_layer_F_R = model_F_R.get_layer('conv5_3_f_r').output

# Global Average pooling layer at output of 4 networks
modelOutM_L = GlobalAveragePooling2D()(last_layer_M_L)
modelOutM_R = GlobalAveragePooling2D()(last_layer_M_R)
modelOutF_L = GlobalAveragePooling2D()(last_layer_F_L)
modelOutF_R = GlobalAveragePooling2D()(last_layer_F_R)

# Landmark layers
model_land_i = Input((272,))
model_land_hidden_1 = Dense(272)(model_land_i)
model_land_elev_pred = Dense(NetworkHelper.numElevClasses)(model_land_hidden_1)
model_land_azim_pred = Dense(NetworkHelper.numAzimClasses)(model_land_hidden_1)

modelOut = concatenate([modelOutM_L, modelOutM_R, modelOutF_L, modelOutF_R], axis=1)

eye_elev_pred = Dense(NetworkHelper.numElevClasses)(modelOut)
eye_azim_pred = Dense(NetworkHelper.numAzimClasses)(modelOut)
elev_pred = concatenate([eye_elev_pred, model_land_elev_pred], axis=1)
azim_pred = concatenate([eye_azim_pred, model_land_azim_pred], axis=1)
ElevPredict = Dense(NetworkHelper.numElevClasses, activation='softmax')(elev_pred)
AzimPredict = Dense(NetworkHelper.numAzimClasses, activation='softmax')(azim_pred)


model = Model(inputs=[model_M_L.input, model_M_R.input, model_F_L.input, model_F_R.input, model_land_i], outputs=[ElevPredict, AzimPredict])

#model = Sequential()
#model.add(InputLayer(272))
#model.add(Dense(272, activation='relu'))
#model.add(Dense(15, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr=lRate, decay=lRate/Epochs), metrics=["accuracy", "categorical_accuracy"])
print(model.summary())

StepsPerEpoch = np.floor(samples_per_epoch/MyBatchSize)
model.fit_generator(train_datagen, steps_per_epoch = StepsPerEpoch, epochs = Epochs,  verbose=1)

# ========== Test in batch ============#
num_t = int(np.floor(len(TestIDs)/MyBatchSize)) #number of test iterations
num_t_s = num_t*MyBatchSize #number of actual test samples 
y_Elev_truth, y_Azim_truth = [], []
y_Elev_soft = np.empty([num_t_s,NetworkHelper.numElevClasses])
y_Azim_soft = np.empty([num_t_s,NetworkHelper.numAzimClasses])
for i in range(num_t):
    TestIDsBatch = TestIDs[i*MyBatchSize:(i+1)*MyBatchSize]
    X_Face_REye, X_Face_LEye, X_Mirror_REye, X_Mirror_LEye, X_Face_Landmarks, X_Mirror_Landmarks, y_Elev_truth_b, y_Azim_truth_b = NetworkHelper.prepareData(TestIDsBatch) #test values
    y_Elev_truth = y_Elev_truth +  list(map(float, y_Elev_truth_b))
    y_Azim_truth = y_Azim_truth + list(map(float, y_Azim_truth_b))
    X_Landmarks = np.concatenate([X_Face_Landmarks, X_Mirror_Landmarks], axis=1)
    [y_Elev_soft_b, y_Azim_soft_b] = model.predict([X_Face_REye, X_Face_LEye, X_Mirror_REye, X_Mirror_LEye, X_Landmarks]) # predictions for Test data
    y_Elev_soft[i*MyBatchSize:(i+1)*MyBatchSize,:] = y_Elev_soft_b
    y_Azim_soft[i*MyBatchSize:(i+1)*MyBatchSize,:] = y_Azim_soft_b
    
y_Elev_pred = np.argmax(y_Elev_soft, axis=1)
y_Azim_pred = np.argmax(y_Azim_soft, axis=1)

## Elevation And Azimuth Accuracy (highest one)
ElevAccuracy = NetworkHelper.AccuracyCal(y_Elev_truth, y_Elev_pred)
print('Elevation Accuracy = ', ElevAccuracy, "\n")

AzimAccuracy = NetworkHelper.AccuracyCal(y_Azim_truth, y_Azim_pred)
print('Azimuth Accuracy = ', AzimAccuracy, "\n")      

## Elevation and Azimuth Accuracy for double resolution
Elev_acc_2 = NetworkHelper.DoubleResAccuracy(y_Elev_truth, y_Elev_soft)
print("Elevation Accuracy double resolution = ", Elev_acc_2, "\n")
  
Azim_acc_2 = NetworkHelper.DoubleResAccuracy(y_Azim_truth, y_Azim_soft)
print("Azimuth Accuracy double resolution = ", Azim_acc_2, "\n")