"""
Author: Usaid Malik (uxm170001)
Date: 3/6/2020
Project: Generating dataset where mirror/face right/left eyes are created and CSV with head angles and image filename
         created
To whomever is reading this, enjoy the PEP8 - files were taking 1 hour to copy, so had the time to do this :)
"""
import face_alignment
import cv2
import collections
import matplotlib.pyplot as plt
import os
import numpy as np
import csv

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda:0')

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
        ]
    if label is not None:
        row.append(label)
    else:
        row.append('')
    # Add blank cells if landmark does not exist for face or mirror
    if landmark_face is not None:
        row.extend([str(int(x)) + ';' + str(int(y)) for x, y in landmark_face])
    else:
        row.extend(['' for i in range(69)])
    if landmark_mirror is not None:
        row.extend([str(int(x)) + ';' + str(int(y)) for x, y in landmark_mirror])
    else:
        row.extend(['' for i in range(69)])
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


def create_folders(prefix: str, discrete=True):
    """
    Makes directories to populate with images.
    :param prefix Prefix of file directory to create mapping directories in
    :param discrete Whether or not need to create folder labels for the discrete location part of dataset
    :return: None
    """
    temp_category = ['/Reye', '/Leye']
    LIST = None
    if discrete:
        LIST = SUBJECT_LIST
    else:
        LIST = CONT_SUBJECT_LIST
    for subject_date in LIST:
        for camera_category in ['/Mirror/', '/Face/']:
            for category in temp_category:
                path = prefix + '/Mapping ' + subject_date  # 'EyeMapping_WithAngles/Mapping ' + subject_date
                images_path = os.getcwd() + '/Mapping ' + subject_date + camera_category
                if discrete:
                    for folder in folder_labels:
                        try:
                            os.makedirs(path + camera_category + '/' + folder + category)
                        except OSError:
                            print("Failed to create folder - " + path + camera_category + '/' + folder + category)
                else:
                    try:
                        os.makedirs(path + camera_category + category)
                    except OSError:
                        print("Failed to create folder - " + path + camera_category + category)


def display_landmarks(image: str, landmarks: list):
    """
    Displays an image with the provided landmarks highlighted so it is easy to identify which face is being used.
    Intended to assist debugging when incorrect face detected.
    :param image: Image to put landmarks on
    :param landmarks: Landmarks of face (assumed to be in image)
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    plot_style = dict(marker='o',
                      markersize=2,
                      linestyle='-',
                      lw=2)
    pred_types = {
        'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
        'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
        'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
        'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
        'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
        'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
        'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
        'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
        'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
        }
    ax.imshow(image)
    for pred_type in pred_types.values():
        ax.plot(landmarks[pred_type.slice, 0],
                landmarks[pred_type.slice, 1],
                color='magenta', **plot_style)
    plt.show()


def create_discrete_mapping():
    """
    Creates face-mirror eye mapping for right/left eye for the discrete images (the ones where gaze was not
    continuous). Uses this pseudocode (was created for easy writing, now here for easy understanding because of
    laziness, sadly).
    For every row of AnglesID.csv for subject in FixedGazeProcessed:
        Get subject name from row
        Get folder label from row
        Get cropped face image from LF
        Get to corresponding mirror folder with subject name + label
        Find corresponding image based on minute/second
        Get eyes from face image
            If no face
                Create CSV with subject, face id, mirror id, face landmarks
        Get eyes from mirror image
            If no face
                Create CSV with subject, face id, mirror id, mirror_landmarks
        Save eyes for face and mirror if they are available
        Create CSV with subject, image id from cropped face, mirror id,
            folder, landmarks, angles
    :return: None
    """
    for subject in SUBJECT_LIST:
        with open('D:/FixedGazeProcessed/FE' + subject + '/AnglesIDfile.csv', 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            print(next(reader))  # Get rid of header row
            file_list = []  # Stores list of files in mirror directory
            file_root = ''  # Stores root of corresponding mirror directory
            for row in reader:  # Parse CSV row by row
                if len(row) == 0:
                    continue
                subject_date = row[0][2:]  # Stores subject date
                label = row[-1]  # Stores discrete location label
                if label == 'b-1':  # Single typo in foldername for subject 6-11
                    label = 'b- 1'
                # 2019-6-11 are .PNG
                image_id = row[1].split('U')[0] + ('.jpg' if subject_date != '2019-6-11' else '.png')
                mirror_path = None
                timecode = '-'.join(image_id.split('-')[-3:])[:-4]

                if file_root != 'D:/Mapping/Mapping ' + subject_date + '/Mirror/' + label:
                    print(label)
                    print("Getting list of files")
                    for root, dirs, files in os.walk('D:/Mapping/Mapping ' + subject_date + '/Mirror/' + label):
                        if len(files) == 0:  # Empty directories
                            continue
                        file_list = files  # There is only one directory, so only one set of files[]
                        file_root = root

                mirror_id = None
                for file in file_list:
                    if timecode + '.' in file:
                        mirror_path = file_root + '/' + file
                        mirror_id = file
                        break

                if mirror_path is None:
                    print(file_list)
                    print(file_root)
                    print('D:/Mapping/Mapping ' + subject_date + '/Mirror/' + label)

                face_path = 'D:/LF/LF ' + subject_date + '/' + label + '/' + image_id
                print(face_path, mirror_path)

                # Get right/left eye of face
                face_left, face_right, face_landmarks = get_eyes_from_file(face_path)
                mirror_left, mirror_right, mirror_landmarks = get_eyes_from_file(mirror_path)

                save_csv(subject_date, image_id, mirror_id, label, face_landmarks, mirror_landmarks, row[2:11],
                         prefix='EyeMapping_WithAngles/Mapping ' + subject_date + '/')

                if face_landmarks is not None:
                    cv2.imwrite('EyeMapping_WithAngles/Mapping ' + subject_date + '/Face/' + label + '/Leye/' + image_id, face_left)
                    cv2.imwrite('EyeMapping_WithAngles/Mapping ' + subject_date + '/Face/' + label + '/Reye/' + image_id, face_right)
                if mirror_landmarks is not None:
                    cv2.imwrite('EyeMapping_WithAngles/Mapping ' + subject_date + '/Mirror/' + label + '/Leye/' + image_id, mirror_left)
                    cv2.imwrite('EyeMapping_WithAngles/Mapping ' + subject_date + '/Mirror/' + label + '/Reye/' + image_id, mirror_right)


def create_continuous_mapping():
    """
    Creates continuous mapping of images. 
    Gets image filename from CSV for subject.
    Uses the FaceAndEyes cropped-face dataset to generate eye mappings 
    (duplicating the eye mappings done in FaceAndEyes/) then getting 
    corresponding mirror frame.
    :return: None
    """
    offsets = {
        # filename frame offset + [0] = actual frame num in face video (+ [1] = frame num in mirror video)
        # [2] is the video filename of the mirror video
        '2019-5-22': [9282, 0, 'GH010116.MP4'],
        '2019-5-30': [13247, 489, 'GH010103.MP4'],
        '2019-6-11': [6186, 28345, 'GH010104.MP4'],
        '2019-6-14': [9930, 760, 'GH010106.MP4'],
        '2019-6-21': [21750, 344, 'GH010126.MP4'],
        '2019-7-9': [5256, 348, 'GH010310.MP4'],
        '2019-7-10': [16162, 418, 'GH010128.MP4'],
        '2019-7-11': [3145, 394, 'GH010130.MP4'],
        '2019-7-15': [20585, 400, 'GH010313.MP4'],
        '2019-7-23': [21643, 332, 'GH010133.MP4'],  # This is different from the 21685 offset Marzban wrote, I found 42-frame offset
        '2019-8-27': [16115, -29, 'GH010118.MP4'],  # Added 1 on [0] since I found a small offset. Not sure if it is incorrect mirror offset though
        '2019-10-30': [15749, 265, 'GH010205.MP4'],
        '2019-10-31': [13109, 260, 'GH010146.MP4']
    }
    # This list is because there is specific filepaths for 2019-7-23 and 2018-12-1 doesn't exist
    # TODO: Retry 7-23, there is a mistake with the frame numbers it seems. Gotta recheck the mirror offset
    # if it is correct then there is a problem with the crop number stated (subtracted 42 already)
    for subject in ['2019-8-27']:
        with open('D:/ContGazeImages/FaceAndEyes/CFE' + subject + '/AnglesIDfile.csv', 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            print(next(reader))  # Ignore header row
            for row in reader:
                if len(row) == 0:  # Every other row is blank, ignore them
                    continue
                print(row[1])
                frame_num_read = int(row[1].split("f")[-1].split(".")[0])
                # Calculate mirror frame number = file # + crop offset + mirror offset
                frame_num = frame_num_read + offsets[subject][0] + offsets[subject][1]
                face_img = cv2.imread('D:/ContGazeImages/FaceAndEyes/CFE' + subject + '/Face/F' + row[1])
                mirror_video = cv2.VideoCapture('D:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive ' + subject + '/Mirror/' + offsets[subject][2])
                mirror_video.set(1, frame_num)
                ret, mirror_img = mirror_video.read()
                face_left, face_right, face_landmarks = get_eyes(face_img)
                mirror_left, mirror_right, mirror_landmarks = get_eyes(mirror_img)

                prefix = 'D:/EyeMapping_WithAngles/ContinuousLocationMapping/Mapping ' + subject
                image_id = row[1]
                save_csv(subject, image_id, image_id, None, face_landmarks, mirror_landmarks, row[2:11],
                         prefix=prefix + '/')

                if face_landmarks is not None:
                    cv2.imwrite(prefix + '/Face/Leye/' + image_id,
                                face_left)
                    cv2.imwrite(prefix + '/Face/Reye/' + image_id,
                                face_right)
                if mirror_landmarks is not None:
                    cv2.imwrite(prefix + '/Mirror/Leye/' + image_id,
                                mirror_left)
                    cv2.imwrite(prefix + '/Mirror/Reye/' + image_id,
                                mirror_right)


# create_eye_pics()
# create_folders('D:/EyeMapping_WithAngles/ContinuousLocationMapping', False)
# create_continuous_mapping()
# cv2.destroyAllWindows()
# create_discrete_mapping()
get_eyes_from_file('D:/LF/LF 2018-12-1/m- 21/D1Dec18Face-13619-9-46-59.jpg')