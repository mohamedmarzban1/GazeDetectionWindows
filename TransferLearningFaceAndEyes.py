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
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
#import dask.array as da

#train_data_dir = "data/train"
#validation_data_dir = "data/val"

Epochs = 300#20#6  
LayersToFreeze = 19
numTestSam = 459#286#25 
MyBatchSize = 32 
ValSize = 64
Categories = ["a- 4", "b- 1", "c- 8", "d- 2", "e- 13", "f- 5", "g- 9", "h- 11", "i- 6", "j- 20", "k- 19", "l- 18", "m- 21", "n- 17", "o- 16", "p- 14", "q- 3", "r- 7", "s- 10", "t- 12" ,"u- 15" ] 


#---- Load the data -----#
pickle_in = open("XFace.pickle","rb")
XFace = pickle.load(pickle_in)
tt = da.from_array(XFace)

pickle_in2 = open("XLEye.pickle","rb")
XLEye = pickle.load(pickle_in2)

pickle_in3 = open("XREye.pickle","rb")
XREye = pickle.load(pickle_in3)

pickle_in4 = open("y_flr.pickle","rb")
y_flr = pickle.load(pickle_in4)


Xtrain = X[numTestSam:,:,:]
Ytrain = y[numTestSam:]
numTrainSam = len(Ytrain)

Xtest = X[0:numTestSam,:,:]
Ytest = y[0:numTestSam]

#---------------------#
img_width, img_height = X.shape[1], X.shape[2]
#x1 = 
#x2 = 
#----------------------#

model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:LayersToFreeze]:
    layer.trainable = False

modelOut = model.output
modelOut = Flatten()(modelOut)
modelOut = Dense(1024, activation="relu")(modelOut)
modelOut = Dropout(0.5)(modelOut)
modelOut = Dense(1024, activation="relu")(modelOut)
predictions = Dense(21, activation="softmax")(modelOut)
#predictions = Dense(2, activation="softmax")(modelOut)


model_final = Model(input = model.input, output = predictions)

#model_final.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizers.Adam(lr=0.001), metrics=["accuracy"])

#keras.optimizer


print(model_final.summary())

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow(Xtrain,Ytrain)



# To evaluate the accuracy on this data after each epoch
Xval = Xtest[:ValSize,:,:,:]
Yval = Ytest[:ValSize]
Val_generator = test_datagen.flow(Xval,Yval) 

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model 
StepsPerEpoch = np.ceil(numTrainSam/MyBatchSize)
model_final.fit_generator( train_generator, steps_per_epoch = StepsPerEpoch, epochs = Epochs,  verbose=1, validation_data = Val_generator, nb_val_samples = ValSize, callbacks = [checkpoint])

#model_final.fit(Xtrain,Ytrain,epochs = Epochs)

#test_loss, test_acc = model_final.evaluate(Xtest, Ytest)
#print('Test accuracy:', test_acc)

Ypred= model_final.predict(Xtest)
Ypred = np.argmax(Ypred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(Ytest, Ypred))
print('Classification Report')
print(classification_report(Ytest, Ypred, target_names=Categories))




