# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 22:13:42 2019

@author: mfm160330
"""
import pickle
import numpy as np


pickle_in = open("../calibPickleFiles/MarkersAppended2019-6-20BackStandard.pickle","rb")
OutputFileName = "../calibPickleFiles/MarkersAppended2019-6-20BackStandardv2.pickle"
labelIDsUni = pickle.load(pickle_in)
XlabelUni = pickle.load(pickle_in)#np.expand_dims(np.ravel(pickle.load(pickle_in)), axis = 1)
YlabelUni = pickle.load(pickle_in)#np.expand_dims(np.ravel(pickle.load(pickle_in)), axis = 1)
ZlabelUni = pickle.load(pickle_in)#np.expand_dims(np.ravel(pickle.load(pickle_in)), axis =1)
numElemLabel =  pickle.load(pickle_in)#pickle.load(pickle_in)

# delete the labels with zero values
#labelIDsUni = np.delete(labelIDsUni,[4,10,26]) 
#XlabelUni = np.delete(XlabelUni,[4,10,26])
#YlabelUni = np.delete(YlabelUni,[4,10,26])
#ZlabelUni = np.delete(ZlabelUni,[4,10,26])


newLabelIDs = [206]
Xnew = [0.4268444]
Ynew = [0.6287]
Znew = [-0.20702]

#Stack saved and new data point 
labelIDsUniUpdated = np.hstack((labelIDsUni,newLabelIDs))
XlabelUniUpdated = np.hstack((XlabelUni,Xnew))
YlabelUniUpdated = np.hstack((YlabelUni,Ynew))
ZlabelUniUpdated = np.hstack((ZlabelUni,Znew))

# sort the points ascendingly based on their labels
IndxSorted = np.argsort(labelIDsUniUpdated)
labelIDsUniSorted = labelIDsUniUpdated[IndxSorted]
XlabelUniSorted = XlabelUniUpdated[IndxSorted]
YlabelUniSorted = YlabelUniUpdated[IndxSorted]
ZlabelUniSorted = ZlabelUniUpdated[IndxSorted]

numElemLabelUpdated = np.ones(31)*10

pickle_out = open(OutputFileName,"wb")
pickle.dump(labelIDsUniSorted, pickle_out)
pickle.dump(XlabelUniSorted, pickle_out)
pickle.dump(YlabelUniSorted, pickle_out)
pickle.dump(ZlabelUniSorted, pickle_out)
pickle.dump(numElemLabelUpdated, pickle_out)

pickle_out.close()


print("detected IDs = ",labelIDsUniSorted)
print("x = ",XlabelUniSorted)
print("y = ",YlabelUniSorted)
print("z= ",ZlabelUniSorted)

