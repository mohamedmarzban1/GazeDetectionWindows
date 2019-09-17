# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 22:13:42 2019

@author: mfm160330
"""
import pickle
import numpy as np


pickle_in = open("../calibPickleFiles/BackCalibAll2019-6-20.pickle","rb")
OutputFileName = "MarkersAppended2019-6-20.pickle"
labelIDsUni = pickle.load(pickle_in)
XlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in)), axis = 1)
YlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in)), axis = 1)
ZlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in)), axis =1)
numElemLabel =  pickle.load(pickle_in)

# delete the labels with zero values
labelIDsUni = np.delete(labelIDsUni,[4,10,26]) 
XlabelUni = np.delete(XlabelUni,[4,10,26])
YlabelUni = np.delete(YlabelUni,[4,10,26])
ZlabelUni = np.delete(ZlabelUni,[4,10,26])


newLabelIDs = [314, 316, 317, 321]
Xnew = [1.15287, 1.19547, .5376698, 1.0777576]
Ynew = [.6344, -1.39106, .614429, -.3633154]
Znew = [-.36089, -.26, -.1447289, -.95447796]

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


pickle_out = open(OutputFileName,"wb")
pickle.dump(labelIDsUniSorted, pickle_out)
pickle.dump(XlabelUniSorted, pickle_out)
pickle.dump(YlabelUniSorted, pickle_out)
pickle.dump(ZlabelUniSorted, pickle_out)
pickle_out.close()


print("detected IDs = ",labelIDsUniSorted)
print("x = ",XlabelUniSorted)
print("y = ",YlabelUniSorted)
print("z= ",ZlabelUniSorted)
