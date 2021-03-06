# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:48:41 2019

Labelling script
Inputs:
    1- ID file of face/eyes detection algorithm
    2- standardizeVisualize Output from fixed gaze points data
    3- BackCalibration.pickle file having the locations of the target markers
    4- Translation and rotation matricies of to get target markers loc in current tag coordinates
Outputs:
    1-AnglesIDfile.csv    
    
Methodology:
    1- Loads the loacation of the target markers in ReF. back coordinates
    2- Read the labeled data ID file and map labelled image to the XYZ gaze location
    3- extract COMs from standardizeVisualize file that maps to the wanted columns only
    4- Transforms the COMs from current back coordinates to reference back coordinates
    5- translate the origin to the center of mass of the driver's head
    6- Calculates the angles between the COMs and the target label correponding to each image.
 
@author: mfm160330
"""
import csv
import pickle
import numpy as np


#### A function that transforms cartesian coordinates to spherical coordinates ###########
def ConvertToSpherical_np(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    #ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = -np.arctan2(xyz[:,1], xyz[:,0]) # added a negative sign to make angles easily understandable
    return ptsnew


def getVideoOffsets(OffsetsTxtFile):
    with open(OffsetsTxtFile,"r") as file:
        for i in range(27):
            data = file.readline()
            if data.find('fixed face frame number start')>=0:
                face_offset = int(data[data.find('=') + 1::]) #in frame numbers
            if data.find('fixed back frame number start')>=0:
                back_offset = int(data[data.find('=') + 1::]) # in frame numbers
    return face_offset, back_offset


#================ Intialize Values ====================#
dataSets = ["2019-12-06-001"]#["2019-11-20-001", "2019-11-22-001", "2019-11-25-001", "2019-12-06-001", "2020-01-18-001", "2020-01-24-001", "2020-01-25-001", "2020-01-27-001", "2020-01-28-001", "2020-02-01-001", "2020-02-07-001", "2020-02-08-001", "2020-02-10-001", "2020-02-14-001", "2020-02-22-001", "2020-03-12-001"] # "2019-11-14-001", "2019-11-19-002", 
RefpickleFile = "calibPickleFiles/BackReference0.032Corners_sorted.pickle"
firstMarkerIndx = 13 # Indx of the first marker of the target markers
AnglesIdFileName = 'AnglesIDfileV4.csv'
Categories = ["a- 4", "b- 1", "c- 8", "d- 2", "e- 13", "f- 5", "g- 9", "h- 11", "i- 6", "j- 20", "k- 19", "l- 18", "m- 21", "n- 17", "o- 16", "p- 14", "q- 3", "r- 7", "s- 10", "t- 12" ,"u- 15" ] 
rotTransPickleFile = "KabaschRotTrans0.032.pickle"
FrameIndexNumber = 3 #   In image name, after which dash is the frame number located


##=======load the saved locations of the target markers ==========#
pickle_in = open(RefpickleFile, "rb")
labelIDsUni = pickle.load(pickle_in)[firstMarkerIndx:]-300
XlabelUni = pickle.load(pickle_in)[firstMarkerIndx:]
YlabelUni = pickle.load(pickle_in)[firstMarkerIndx:]
ZlabelUni = pickle.load(pickle_in)[firstMarkerIndx:]
    #numElemLabel =  pickle.load(pickle_in)

for dataset in dataSets:
    print("Processing subject", dataset)
    FaceIDFileAndWriteLoc = "D:/FixedGazeImages/FaceAndEyes/FE"+dataset
    OutputFilesReadLoc = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D"+dataset+"/FixedGaze/"
    OffsetsTxtFile = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/OutputFiles/D"+dataset+"/StartAndEndFrameNums.txt"
    
    # get face and back Fixed start
    #faceFixedStart = 141510 
    #backFixedStart = 141840 
    faceFixedStart, backFixedStart =  getVideoOffsets(OffsetsTxtFile)


    #IgnoreLabels = [17,18,20]
    backToFaceOffset = faceFixedStart - backFixedStart
    IDFileLoc = FaceIDFileAndWriteLoc+"/id.csv"
    VisualizeFile = OutputFilesReadLoc+"/visualize_framesT2.csv"
    #======== load the saved labels locations ===========#


    #====== Read ID file, Get all manually labeled file names =========#
    IDs, DataSets, labels = [], [] ,[]
    Face_X1, Face_Y1, Face_X2, Face_Y2 = [], [], [], []
    LEye_X1, LEye_Y1, LEye_X2, LEye_Y2 = [], [], [], []
    REye_X1, REye_Y1, REye_X2, REye_Y2 = [], [], [], []
    with open(IDFileLoc, "r") as csvfile:
        next(csvfile) #skip heading
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if not ''.join(row).strip():
                continue # ignore the blank lines
            DataSets.append(row[0])
            IDs.append(row[1])
            labels.append(row[2])
            Face_X1.append(row[3]), Face_Y1.append(row[4]), Face_X2.append(row[5]), Face_Y2.append(row[6])
            LEye_X1.append(row[7]), LEye_Y1.append(row[8]), LEye_X2.append(row[9]), LEye_Y2.append(row[10])
            REye_X1.append(row[11]), REye_Y1.append(row[12]), REye_X2.append(row[13]), REye_Y2.append(row[14])

    # =======  Get Frame Numbers from file name  ========== #
    frameNums = []
    for ID in IDs:
        SplittedID = ID.split("-")
        frameNum = SplittedID[FrameIndexNumber]
        frameNums.append(int(frameNum))    
    # ====== map each face image to its label location  ============#
    labelIDs = []
    Xlabel = []
    Ylabel = []
    Zlabel = []
    labelError = []
    for label in labels:
        SplittedLabel = label.split("-")
        labelID = int(SplittedLabel[1])
        labelIDs.append(labelID)
    
        #indx = labelIDsUni.index(labelID)
        try:
            indx = np.where(labelIDsUni == labelID)[0][0]
        except:
            #If Error occured, then I don't have an XYZ location for this label
            labelError.append(labelID)
            Xlabel.append('nan')
            Ylabel.append('nan')
            Zlabel.append('nan')      
            continue 
        Xlabel.append(XlabelUni[indx])
        Ylabel.append(YlabelUni[indx])
        Zlabel.append(ZlabelUni[indx])

    Xlabel = np.expand_dims(np.asarray(Xlabel, dtype=np.float32), axis=1) 
    Ylabel = np.expand_dims(np.asarray(Ylabel, dtype=np.float32), axis=1)
    Zlabel = np.expand_dims(np.asarray(Zlabel, dtype=np.float32), axis=1)
    XYZlabel = np.hstack((Xlabel,Ylabel,Zlabel)) 

# structured data now contain DataSets, IDs, labels, frameNums, labelNums, Xlabel, Ylabel, Zlabel

#================================================================#
    
    #========= Read Visluaize file  ============#
    FrameNumsV = [] #V stands for visualize
    XsV = [] 
    YsV = []
    ZsV = []
    with open(VisualizeFile, "r") as csvfile:
        next(csvfile) #skip heading
        readCSV = csv.reader(csvfile, delimiter='\t')
        for FrameNumVisual, Xvisual, Yvisual, Zvisual, a, bi, cj, dk in readCSV:
            FrameNumsV.append(int(FrameNumVisual)) 
            XsV.append(float(Xvisual))   ##XsV.append(Xvisual)
            YsV.append(float(Yvisual))   ##YsV.append(Yvisual)
            ZsV.append(float(Zvisual))   ##ZsV.append(Zvisual)        

    #====== Extract the COMs from Visualize file corresponding to the required frames Only ======#
    Xcom = []
    Ycom = []
    Zcom = []
    for frameNum in frameNums: 
        try:
            # here we added backToFaceOffset to map face frames to back ones
            BackframeNum = int(frameNum)-backToFaceOffset
            FrameNumindx = FrameNumsV.index(BackframeNum)
            Xcom.append(float(XsV[FrameNumindx])) 
            Ycom.append(float(YsV[FrameNumindx]))
            Zcom.append(float(ZsV[FrameNumindx]))
        except:
            #If Error occured, then I don't have a COM location for this frameNumber
            print("WARNING: No COM detected for frameNum ", frameNum, "\n")
            Xcom.append('nan')
            Ycom.append('nan')
            Zcom.append('nan')
        
    Xcom = np.expand_dims(np.asarray(Xcom, dtype=np.float32), axis=1) 
    Ycom = np.expand_dims(np.asarray(Ycom, dtype=np.float32), axis=1)
    Zcom = np.expand_dims(np.asarray(Zcom, dtype=np.float32), axis=1)
    XYZcom = np.hstack((Xcom,Ycom,Zcom)) 

    #==== transforms the COMs to refBack coordinates ============#

    # load the rotation and translation matrices  #
    pickle_in2 = open(OutputFilesReadLoc + rotTransPickleFile, "rb")
    R = pickle.load(pickle_in2)
    C_curr = pickle.load(pickle_in2)
    C_ref = pickle.load(pickle_in2)
    XYZcomCalib = np.matmul(XYZcom - C_curr, R) + C_ref  
    #test2 = np.dot(XYZcom - C_curr, R)+C_ref
    # structured data now contain DataSets, IDs, labels, frameNums, labelNums, Xlabel, Ylabel, Zlabel, Xcom, Ycom, Zcom
    
    #======== Calculate the polar angles based on label(Xlabel,Ylabel,Zlabel) and COM (Xcom,Ycom,Zcom) ==========#
    ## Move origin to COM
    XYZlc = XYZlabel - XYZcomCalib
    # structured data now contain DataSets, IDs, labels, frameNums, labelNums, XYZlabel, XYZcomCalib, XYZlc


    ### Transform cartesian coordinates to sperical coordinates to get angles
    AngleLabels = ConvertToSpherical_np(XYZlc) #Angle labels contain X,Y,X,Rho,Theta,Phi
    #np.hstack((DataSets,IDs,AngleLabels))

    # write Datasets, IDs, and angles to CSV
    #csv_header = ["DataSetID", "ImageID", "Rho", "Elev", "Azim", "LabelNum"]
    #csv_header = "DataSetID,ImageID,Rho,Elev,Azim,LabelNum"
    #csv_header = "DataSetID \t ImageID \t Rho \t Elev \t Azim \t LabelNum \t Xcom \t Ycom \t Zcom"
    csv_header = "DataSetID\tImageID\tRho\tElev\tAzim\tXcom\tYcom\tZcom\tXtarget\tYtarget\tZtarget\tFace_X1\tFace_Y1\tFace_X2\tFace_Y2\tLEye_X1\tLEye_Y1\tLEye_X2\tLEye_Y2\tREye_X1\tREye_Y1\tREye_X2\tREye_Y2\tlabels\n"
    csvfile = open (FaceIDFileAndWriteLoc+'/'+AnglesIdFileName, 'w+', newline="") 
    csvfile.write(csv_header)

    for i in range(XYZlc.shape[0]):
        with open(FaceIDFileAndWriteLoc+'/'+AnglesIdFileName, 'a+', newline="") as csvfile:
            filewriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)            
            #filewriter.writerow([DataSets[i], IDs[i], AngleLabels[i][0], AngleLabels[i][1], AngleLabels[i][2], labels[i], XYZcomCalib[i][0], XYZcomCalib[i][1], XYZcomCalib[i][2]])
            filewriter.writerow([DataSets[i], IDs[i], AngleLabels[i][0], AngleLabels[i][1], AngleLabels[i][2], XYZcomCalib[i][0], XYZcomCalib[i][1], XYZcomCalib[i][2], Xlabel[i][0], Ylabel[i][0], Zlabel[i][0], Face_X1[i], Face_Y1[i], Face_X2[i], Face_Y2[i], LEye_X1[i], LEye_Y1[i], LEye_X2[i], LEye_Y2[i], REye_X1[i], REye_Y1[i], REye_X2[i], REye_Y2[i], labels[i]])
    
    csvfile.close()
    print("finished subject", dataset)
print("Finished labelling all subjects")