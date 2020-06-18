# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 22:13:42 2019

@author: mfm160330
"""
import pickle
import numpy as np


pickle_in = open("../calibPickleFiles/BackReference0.032Appended.pickle","rb")
OutputFileName = "../calibPickleFiles/BackReference0.032Appendedv2.pickle"
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

[ 0.51101412 -1.15941086 -0.17225367]
newLabelIDs = [318]#[206, 316, 317, 321]
    
Xnew = [0.51101412] #[0.34526502, 1.0121309, 0.40895749, 0.85777994]
Ynew = [-1.15941086] #[0.48491659, -1.06747411, 0.68667606,  -0.28736424]
Znew = [-0.17225367] #[-0.17995367, -0.26459174, 0.09926209, -0.76973065]

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

