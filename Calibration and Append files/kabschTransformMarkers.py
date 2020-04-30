# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:04:13 2019
Transforming from XYZ_current coordinate to ref coordinate

@author: mfm160330
"""


import csv
import numpy as np
import pickle
import rmsd

XYZcurrent = np.array([[0.5661,0.0327864, 0.243528], [0.617012, 0.388014, 0.0854373], [0.691728, -0.313644, 0.0731137], [0.787969, -0.321781, -0.00476555], [0.904381, -0.445258, -0.1777]])

### --- CONSTANTS --- ###

FACE_CURR = "../calib_files/FaceCurr2019-6-14.pickle"
FACE_REF = "../calib_files/FaceCalib2019-6-20.pickle"
BACK_REF = "../calib_files/BackCalibAll2019-6-20.pickle"
FACE_TRANSFORM_PATH = "../calib_files/FaceCurrToRef.pickle"
BACK_TO_FACE_TRANSFORM_PATH = "../calib_files/BackToFaceRefCalib.pickle"

#pickle_in = open("../calibPickleFiles/MarkersAppended2019-6-20BackStandard.pickle","rb")
#labelIDsUni = pickle.load(pickle_in)
#print(labelIDsUni)
#XlabelUni = pickle.load(pickle_in)[np.array([0,26,4,10,14])]#np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([0,26,4,10,14])-1].T, axis = 1)
#YlabelUni = pickle.load(pickle_in)[np.array([0,26,4,10,14])]#np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([0,26,4,10,14])-1].T, axis = 1)
#ZlabelUni = pickle.load(pickle_in)[np.array([0,26,4,10,14])]#np.expand_dims(np.ravel(pickle.load(pickle_in))[np.array([0,26,4,10,14])-1].T, axis =1)
#numElemLabel =  pickle.load(pickle_in)

#XYZref = np.vstack((XlabelUni,YlabelUni,ZlabelUni)).T

### --- FACE REFERENCE TO FACE CURRENT TRANSFORMATION --- ###
"""
CURR_FRAME_PATH = FACE_REF
GOAL_FRAME_PATH = FACE_CURR
OUTPUT_PATH = FACE_TRANSFORM_PATH
"""
### --- END --- ###

### --- BACK REFERENCE TO FACE REFERNCE TRANSFORMATION --- ####
CURR_FRAME_PATH = BACK_REF
GOAL_FRAME_PATH = FACE_REF
OUTPUT_PATH = BACK_TO_FACE_TRANSFORM_PATH
### --- END --- ###

### --- END CONSTANTS --- ###


def find_common_markers(marker_list_1, marker_list_2):
    return list(set(marker_list_1) & set(marker_list_2))


def parse_pickle(path, marker_indexes):
    print("loading pickle from", path)
    pickle_in = open(path, "rb")
    pickle.load(pickle_in)
    XlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in))[
                               np.array(marker_indexes)-1].T, axis=1)
    YlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in))[
                               np.array(marker_indexes)-1].T, axis=1)
    ZlabelUni = np.expand_dims(np.ravel(pickle.load(pickle_in))[
                               np.array(marker_indexes)-1].T, axis=1)
                               
    return np.hstack((XlabelUni, YlabelUni, ZlabelUni))


def read_markers(path):
    return pickle.load(open(path, "rb"))


"""
    Returns a tuple consisting of (R, C_curr, C_ref)
        specifiying the rotation, current centroid, and reference centroid
    Uses Kabsch algorithm to minimize rmsd through a matrix transformation
        between pointsCurrent to pointsRef
"""


def kabsch(pointsCurrent, pointsRef):

    C_curr = rmsd.centroid(pointsCurrent)
    C_ref = rmsd.centroid(pointsRef)

    points_curr_norm = pointsCurrent - C_curr  # XYZ current after centering
    points_ref_norm = pointsRef - C_ref

    # the optimal rotation matrix to rotate XYZcurrent to XYZref
    R = rmsd.kabsch(points_curr_norm, points_ref_norm)

    return (R, C_curr, C_ref)


def find_transform(back_path, face_path):
    back_markers = read_markers(back_path)
    face_markers = read_markers(face_path)
    common_markers = find_common_markers(back_markers, face_markers)

    if len(common_markers) > 3:
        print("Found common markers {}.".format(common_markers))
        common_markers = common_markers[:3]
        print("Using {} for Kabsch transformation.".format(common_markers))
    else:
        print("Can't run Kabsch algorithm. Only found {} common markers.".format(
            common_markers))
        exit()

    back_indexes = [(np.where(back_markers == cm)[0][0])
                    for cm in common_markers]
    face_indexes = [(np.where(face_markers == cm)[0][0])
                    for cm in common_markers]

    print('Shared markers correspond to BACK indices {}'.format(list(back_indexes)))
    print('Shared markers correspond to FACE indices {}'.format(list(face_indexes)))

    back_calib_matrix = parse_pickle(back_path, back_indexes)
    face_calib_matrix = parse_pickle(face_path, face_indexes)

    return kabsch(back_calib_matrix, face_calib_matrix)


"""
    Rotation process from P to Q:
        (R, c_cur, c_ref) = kabsh(P, Q)

    To rotate a point V:
        np.matmul(V - c_cur, R) + c_ref
"""


def main():
    (R, c_back, c_face) = find_transform(CURR_FRAME_PATH, GOAL_FRAME_PATH)

    calib_once_output = open(OUTPUT_PATH, "wb")

    print("Rotation\n", R)
    print("Centroid of curr reference frame", c_back)
    print("Centroid of goal reference frame", c_face)

    print("Dumping rotation and centroids to pickle file \"{}\"".format(OUTPUT_PATH))

    pickle.dump(R, calib_once_output)
    pickle.dump(c_back, calib_once_output)
    pickle.dump(c_face, calib_once_output)

#XYZToBeTransformed = np.array([0.566071, 0.502468, 0.0572864])

if __name__ == '__main__':
    main()
