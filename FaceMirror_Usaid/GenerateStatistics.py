# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:47:45 2020

@author: uxm170001

Creates statistics for undetected face/mirror for frames in each subject
for TxACE.
"""
import pandas as pd
import face_alignment
import cv2
import collections
import matplotlib.pyplot as plt
import os
import numpy as np
import csv

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda:0')

def get_eyes_from_file(image_path: str) -> (np.ndarray, np.ndarray, list):
    """
    Gets the right and left eye of the driver's face given face or mirror image and returns images with face landmarks
    :param image_path:  Direct filepath to image to read from
    :return:            left eye image, right eye image, landmarks of face used to create eye images
    """
    image = cv2.imread(image_path)
    return get_eyes(image)


def get_eyes(image: np.ndarray) -> (np.ndarray, np.ndarray, list):
    """
    Gets the right and left eye of driver from image and returns eyes with landmarks
    :param image: Image to extract eyes from
    :return: left eye, right eye, landmarks
    """
    landmarks = fa.get_landmarks(image)
    if landmarks is None:
        # No faces detected
        return None, None, None
    image_left = None
    image_right = None
    rightmost_bound = 0
    important_landmark = None

    # Pixels margin between boundaries of eye and cropped image
    eye_margin_horiz = 15
    eye_margin_vert = 15

    area = 0    # A combination of how to the right and large the face is
    for landmark in landmarks:
        landmark_np = np.asarray(landmark, dtype=np.int32)
        bounds = (np.amin(np.abs(landmark_np), axis=0), np.amax(np.abs(landmark_np), axis=0))
        if (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1]) < area:
            continue  # It detected something smaller than the previous faces current face
        else:
            area = (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])
        landmark_left = np.asarray(landmark[36:42, :], dtype=np.int32)    # Slice landmark array for left eye locations
        landmark_right = np.asarray(landmark[42:48, :], dtype=np.int32)   # Slice landmark array for right eye locations

        # Bounding box for left/right eye. ((x_min, y_min), (x_max, y_max))
        bounds_left = (np.amin(np.abs(landmark_left), axis=0),
                       np.amax(landmark_left, axis=0))  # Some landmarks are offscreen/hidden, so negative
        bounds_right = (np.amin(np.abs(landmark_right), axis=0), np.amax(landmark_right, axis=0))

        if rightmost_bound < bounds_left[1][0]:
            # Make sure larger face detected is as right as possible (to avoid passenger being used)
            rightmost_bound = bounds_left[1][0]
            image_bound_left = (max(bounds_left[0][1] - eye_margin_horiz, 0),
                                max(bounds_left[0][0] - eye_margin_vert, 0),
                                min(bounds_left[1][1] + eye_margin_horiz, image.shape[0]),
                                min(bounds_left[1][0] + eye_margin_horiz, image.shape[1]))
            image_bound_right = (max(bounds_right[0][1] - eye_margin_horiz, 0),
                                 max(bounds_right[0][0] - eye_margin_vert, 0),
                                 min(bounds_right[1][1] + eye_margin_horiz, image.shape[0]),
                                 min(bounds_right[1][0] + eye_margin_horiz, image.shape[1]))
            image_left = image[image_bound_left[0]:image_bound_left[2], image_bound_left[1]:image_bound_left[3]]
            image_right = image[image_bound_right[0]:image_bound_right[2], image_bound_right[1]:image_bound_right[3]]
            important_landmark = landmark

    return image_left, image_right, important_landmark

def save_csv(subject: str, image_id: str, mirror_id: str, label: str, landmark_face: list,
             landmark_mirror: list, angles: list = None, prefix: str = ''):
    """
    Saves row to CSV given parameters (all these optional settings are for different CSV requirements.
    TODO: Merge optional settings into single setting for code (don't have different uses for same save_csv)
    :param subject:         Date of subject (20XX-XX-XX).
    :param image_id:        Name of face image file.
    :param mirror_id:       Name of mirror image file.
    :param label:           Folder label (for discrete locations for driver to look at)
    :param landmark_face:   All face landmarks
    :param landmark_mirror: All mirror landmarks
    :param angles:          Angles as read from the AnglesID.csv
    :param prefix:          Prefix to where to save the CSV (or add start to filename, such as FAILED_)
    :return:                None
    """
    row = [
        subject,
        image_id,
        mirror_id,
        ]
    if label is not None:
        row.append(label)
    else:
        row.append('')
    # Add blank cells if landmark does not exist for face or mirror
    if landmark_face is not None:
        row.extend([str(int(x)) + ';' + str(int(y)) for x, y in landmark_face])
    else:
        row.extend(['' for i in range(68)])
    if landmark_mirror is not None:
        row.extend([str(int(x)) + ';' + str(int(y)) for x, y in landmark_mirror])
    else:
        row.extend(['' for i in range(68)])
    # Add in angles
    if angles is not None:
        row.extend(angles)

    # Add to CSV for failures to detect face/mirror
    if landmark_face is None or landmark_mirror is None:
        prefix += 'FAILED_'

    # Write to CSV - appends to the file so need to make sure duplicates do not happen by deleting before reruns
    with open(prefix + 'Mapping ' + subject + '.csv', 'a+', newline='') as file:
        filewriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\r')
        filewriter.writerow(row)

PATH = 'D:/EyeMapping_WithAngles/DiscreteLocationMapping/'
# List of subjects that code uses
SUBJECT_LIST = ['2018-12-1', '2019-5-22', '2019-5-30', '2019-6-11',
                '2019-6-14',  '2019-7-9', '2019-7-10', '2019-7-11',
                '2019-7-15', '2019-7-23']

CONT_SUBJECT_LIST = ['2019-5-22', '2019-5-30', '2019-6-11', '2019-6-14', 
                     '2019-6-21', '2019-7-9', '2019-7-10', '2019-7-11', 
                     '2019-7-15', '2019-7-23', '2019-8-27', '2019-10-30', 
                     '2019-10-31']
# Labels of subdirectories (labels of discrete locations driver looked at for discrete part of dataset)
folder_labels = ['a- 4', 'b- 1', 'c- 8', 'd- 2', 'e- 13', 'f- 5', 'g- 9', 'h- 11', 'i- 6', 'j- 20', 'k- 19',
                 'l- 18', 'm- 21', 'n- 17', 'o- 16', 'p- 14', 'q- 3', 'r- 7', 's- 10', 't- 12', 'u- 15']

with open('tables.csv', 'w+') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\r')
    header = ['SUBJECT', 'TOTAL', 'TOTAL_FACE', 'TOTAL_MIRROR', 'MISSING_FACE']
    header.extend(['MISSING_FACE ({})'.format(i) for i in folder_labels])
    header.append('MISSING_MIRROR')
    header.extend(['MISSING_MIRROR ({})'.format(i) for i in folder_labels])
    writer.writerow(['DISCRETE'])
    writer.writerow(header)
    
    for subject in SUBJECT_LIST:
        successful_df = pd.read_csv(PATH + 'Mapping ' + subject + '/Mapping ' + subject + '.csv')
        merged_df = None
        df_list = [successful_df]
        failed_df = None
        processed_df = None
        try:
            failed_df = pd.read_csv(PATH + 'Mapping ' + subject + '/FAILED_Mapping ' + subject + '.csv')
            df_list.append(failed_df)
        except OSError as e:
            print("No FAILED csv for " + subject)
        try:
            processed_df = pd.read_csv(PATH + 'Mapping ' + subject + '/PROCESSED_FAILED_Mapping ' + subject + '.csv')
            df_list.append(processed_df)
        except OSError as e:
            print("No PROCSSED csv for " + subject)
        
        merged_df = pd.concat(df_list, ignore_index=True)
        missing_face = merged_df[merged_df['Face0'].isna()]
        missing_mirror = merged_df[merged_df['Mirror0'].isna()]
        total_missing = missing_face.shape[0] + missing_mirror.shape[0]
        total_missing_perc = float(total_missing) / merged_df.shape[0]
        missing_face_perc = float(missing_face.shape[0]) / merged_df.shape[0]
        missing_mirror_perc = float(missing_mirror.shape[0]) / merged_df.shape[0]
        
        missing_mirror_b1 = missing_mirror.loc[missing_mirror['Label'] == 'b- 1'].shape[0]
        row = [subject, merged_df.shape[0], merged_df.shape[0] - missing_face.shape[0], merged_df.shape[0] - missing_mirror.shape[0], missing_face.shape[0]]
        row.extend([missing_face.loc[missing_face['Label'] == i].shape[0] for i in folder_labels])
        row.append(missing_mirror.shape[0])
        row.extend([missing_mirror.loc[missing_mirror['Label'] == i].shape[0] for i in folder_labels])
        writer.writerow(row)
        print('SUBJECT: {} - MISSING FACE: {} ({}%), MISSING MIRROR: {} ({}%), TOTAL MISSING: {} ({}%), TOTAL: {}, B-1: {}'.format(subject, missing_face.shape[0], missing_face_perc, missing_mirror.shape[0], missing_mirror_perc, total_missing, total_missing_perc, merged_df.shape[0], missing_mirror_b1))

#for subject_date in SUBJECT_LIST:
#    try:
#        with open('D:/FixedGazeProcessed/FE' + subject_date + '/FaceNotDetected.txt', 'r') as file:
#            file_root = ''
#            file_list = []
#            for line in file:
#                label = line.split('      ')[0].split(' ')[3] + ' ' + line.split('      ')[0].split(' ')[4]
#                image_id = line.split('      ')[1].split(' ')[0]
#                print(label, image_id)
#                
#                mirror_path = None
#                timecode = '-'.join(image_id.split('-')[-3:])[:-4]
#        
#                if file_root != 'D:/Mapping/Mapping ' + subject_date + '/Mirror/' + label:
#                    print(label)
#                    print("Getting list of files")
#                    for root, dirs, files in os.walk('D:/Mapping/Mapping ' + subject_date + '/Mirror/' + label):
#                        if len(files) == 0:  # Empty directories
#                            continue
#                        file_list = files  # There is only one directory, so only one set of files[]
#                        file_root = root
#        
#                mirror_id = None
#                for file in file_list:
#                    if timecode + '.' in file:
#                        mirror_path = file_root + '/' + file
#                        mirror_id = file
#                        break
#                print(image_id, mirror_id, timecode, mirror_path)
#                
#                face_path = 'D:/LF/LF ' + subject_date + '/' + label + '/' + image_id
#                print(face_path, mirror_path)
#        
#                # Get right/left eye of face
#                face_left, face_right, face_landmarks = get_eyes_from_file(face_path)
#                mirror_left, mirror_right, mirror_landmarks = get_eyes_from_file(mirror_path)
#                
#                save_csv(subject_date, image_id, mirror_id, label, face_landmarks, mirror_landmarks,                 
#                         prefix=PATH + 'Mapping ' + subject_date + '/PROCESSED_')
#
#                # Do not save the images since we don't have the angles for them
##                if face_landmarks is not None:
##                    cv2.imwrite('EyeMapping_WithAngles/Mapping ' + subject_date + '/Face/' + label + '/Leye/' + image_id, face_left)
##                    cv2.imwrite('EyeMapping_WithAngles/Mapping ' + subject_date + '/Face/' + label + '/Reye/' + image_id, face_right)
##                if mirror_landmarks is not None:
##                    cv2.imwrite('EyeMapping_WithAngles/Mapping ' + subject_date + '/Mirror/' + label + '/Leye/' + image_id, mirror_left)
##                    cv2.imwrite('EyeMapping_WithAngles/Mapping ' + subject_date + '/Mirror/' + label + '/Reye/' + image_id, mirror_right)
#    except OSError as e:
#        print("Could not find file for subject: " + subject_date)