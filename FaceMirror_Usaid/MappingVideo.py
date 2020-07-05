import glob
import os
import cv2

"""
F1 - shutter at 2102, total number of frames=57719
F2 - total number of frames=57719
Mirror - shutter at 2591, total number of frames=57719
Offset - 2591 - 2102 = 489

F2=a-4  - 27785 start - 27803 end (27632 difference from recorded frame number)
F3=a-4  - 11402 start - 11444 end (1978 difference from recorded frame number)
"""

#Stores the parameters for the various videos to make it easy to switch between subjects
params = {
    #Mr. Merzban himself
    '2018-12-1': [
        #Due to the huge offset, there is some custom code and first/second mirror videos are swapped
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/TestDrive2018-12-1/Face/F3.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/TestDrive2018-12-1/Mirror/GH030070.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/TestDrive2018-12-1/Face/IGNORE.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/TestDrive2018-12-1/Mirror/GH020070.MP4',
        21578,
        0,
        -36746,	#37211 F1 shutter (second one, first one mirror not on), 465 Mirror1 shutter, 465 - 37211 = -36746 offset!
        2,
        False,
        'IGNORE'
    ], #Could not find mirror video for subject 2018-12-3
    #White young male, short black curly hair w/ thin beard
    '2019-5-22': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-5-22/Face/GH020175.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-5-22/Mirror/GH020116.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-5-22/Face/F3.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-5-22/Mirror/GH030104.MP4',
        30570,
        0,
        0,	#3366 F1 shutter, 3366 Mirror1 shutter, 3366 - 3366 = 0 offset (wow!)
        4,
        False,
        'IGNORE'
    ],
    #Asian male, blue polo with black glasses
    '2019-5-30': [
        'F2_NoSound.MP4',		#Path to audio-removed video file for facecam (video 1)
        'MirrorF2_NoSound.MP4',	#Path to audio-removed video file for mirror cam (video 1)
        'F3_NoSound.MP4',		#Path to audio-removed video file for facecam (video 2)
        'MirrorF3_NoSound.MP4',	#Path to audio-removed video file for mirror cam (video 2)
        27632,					#Offset of cropped image frame start to actual video frame start (video 1)
        1978,					#Offset of cropped image frame start to actual video frame start (video 2)
        489,					#Offset between facecam and mirror (mirror = facecam + offset)
        4,						#Index of frame number to extract from filename split by '-' + 1
        True,					#Whether there is a single video or not (if True, there is a second video)
        'FacesecVid'			#The label for the video 2
    ],
    #Lot of frames in mirror were blocked by hand in this one
    '2019-6-11': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-11/Face/F2.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-11/Mirror/GH020104.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-11/Face/F3.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-11/Mirror/GH030104.MP4',
        3600,
        0,
        28345,	#1203 F1 shutter, 29547 Mirror1 shutter, 29547 - 1203 = 28344 offset (wow!)
        4,
        False,
        'IGNORE'
    ],
    #Asian young male, short black hair w/ thin glasses
    '2019-6-14': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-14/Face/GH030178.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-14/Mirror/GH030106.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-14/Face/F3.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-14/Mirror/GH030104.MP4',
        -40, #This one was weird, none of the timestamps nor frame numbers were accurate. Spent 30 mins for this offset
        0,
        760,	#995 F1 shutter, 1755 Mirror1 shutter, 1755 - 995 = 800 offset. However, reduce by 40 since the face offset is negative
        4,
        False,
        'IGNORE'
    ],
    #White young man with UTD t-shirt and brown beard, glasses. Closed his eyes every time he was moving his head
    #to gaze a new location, so several closed-eye frames
    '2019-7-9': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-9/Face/GH030182.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-9/Mirror/GH030310.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-9/Face/F3.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-9/Mirror/GH030104.MP4',
        0,
        0,
        348,	#486 F1 shutter, 834 Mirror1 shutter, 834 - 486 = 348 offset (tiny!)
        4,
        False,
        'IGNORE'
    ],
    #White lady, black curly hair, blue dress
    '2019-7-10': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-10/Face/GH030184.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-10/Mirror/GH030128.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-10/Face/F3.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-10/Mirror/GH030104.MP4',
        0,
        0,
        418,	#1413 F1 shutter, 1831 Mirror1 shutter, 1831 - 1413 = 418 offset
        4,
        False,
        'IGNORE'
    ],
    #White young male, blue/white striped shirt w/glasses
    '2019-7-11': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-11/Face/GH030186.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-11/Mirror/GH030130.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-11/Face/F3.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-11/Mirror/GH030104.MP4',
        0,
        0,
        394,	#612 F1 shutter, 1006 Mirror1 shutter, 1006 - 612 = 394 offset. On this one, it seemed that the offset was off. I double-checked using blink light from back camera
        4,
        False,
        'IGNORE'
    ],
    #Indian subcontinent middle-aged male, black beard w/glasses
    '2019-7-15': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-15/Face/GH030188.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-15/Mirror/GH030313.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-15/Face/F3.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-15/Mirror/GH040313.MP4',
        22118, #This one had a frame difference between the frame calculated from the timestamp and the actual frame num
        0,
        399,	#1036 F1 shutter, 1436 Mirror1 shutter, 1436 - 1036 = 400 offset. 399 checking using back GoPro blinks
        4,
        False,
        'IGNORE'
    ],
    #Dr. Naofal himself
    '2019-7-23': [
        'D:/Face/GH030193.MP4',
        'D:/Mirror/GH030133.MP4',
        '',
        '',
        0,  # This one had a frame difference between the frame calculated from the timestamp and the actual frame num
        0,
        332,  # 500 F1 shutter, 832 Mirror1 shutter, 832 - 500 = 332 offset.
        4,
        False,
        'IGNORE'
    ],

    # Start of unique continuous drives
    # The 3rd and 4th lines are irrelevant since the continuous parts are in the first video only
    # This is mostly here just for record-keeping rather than for usage in this file (bad practice, I know)
    # Asian male student, black shirt, no glasses. Was he a RA before? He appears in the back seat quite a few times.
    '2019-6-21': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-21/Face/GH010181.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-6-21/Mirror/GH010126.MP4',
        '',  # Irrelevant
        '',  # Irrelevant
        None,  # Irrelevant
        None,  # Irrelevant
        334,  # 809 F1 shutter, 1143 Mirror1 shutter, 1143 - 809 = 334 offset.
        None,  # Irrelevant
        None,  # Irrelevant
        'IGNORE'
    ],
    # Middle Eastern lady with blue headscarf, no glasses
    '2019-7-19': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-19/Face/GH010190.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-7-19/Mirror/GH010116.MP4',
        '',
        '',
        None,
        None,
        666,  # 485 F1 shutter, 1151 Mirror1 shutter, 1151 - 485 = 666 offset.
        None,
        None,
        'IGNORE'
    ],
    # Asian male with thin black glasses, brownish-green shirt
    '2019-8-27': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-8-27/Face/F1.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-8-27/Mirror/GH010118.MP4',
        '',
        '',
        None,
        None,
        -29,  # 886 F1 shutter, 857 Mirror1 shutter, 857 - 886 = -29 offset.
        None,
        None,
        'IGNORE'
    ],
    # Asian lady with maroon glasses, blue jacket (unzipped) with blue shirt
    '2019-10-30': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-10-30/Face/GH010330.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-10-30/Mirror/GH010205.MP4',
        '',
        '',
        None,
        None,
        265,  # 848 F1 shutter, 1113 Mirror1 shutter, 1113 - 848 = 265 offset.
        None,
        None,
        'IGNORE'
    ],
    # White young-adult woman with no glasses, gray sweater(?)
    '2019-10-31': [
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-10-31/Face/F1.MP4',
        'F:/backup2019-7-19/Multi-sensors gaze Data Collection/Drive 2019-10-30/Mirror/GH010146.MP4',
        '',
        '',
        None,
        None,
        260,	#977 F1 shutter, 1237 Mirror1 shutter, 1237 - 977 = 260 offset.
        None,
        None,
        'IGNORE'
    ]
}

SUBJECT = '2019-7-23'
FACE_VIDEO1 = params[SUBJECT][0]
FACE_VIDEO2 = params[SUBJECT][2]
MIRROR_VIDEO1 = params[SUBJECT][1]
MIRROR_VIDEO2 = params[SUBJECT][3]
CROP_OFFSET1 = params[SUBJECT][4]
CROP_OFFSET2 = params[SUBJECT][5]
MIRROR_CROP_OFFSET = params[SUBJECT][6]
LABEL_VIDEO2 = params[SUBJECT][9]
FRAME_NUM_INDEX = params[SUBJECT][7]

cap = cv2.VideoCapture(FACE_VIDEO1)
cap_mirror = cv2.VideoCapture(MIRROR_VIDEO1)
cap2 = cv2.VideoCapture(FACE_VIDEO2)
cap2_mirror = cv2.VideoCapture(MIRROR_VIDEO2)

#Make directories to populate
temp_category = ['\\Face\\', '\\Mirror\\']

folder_labels = ['a- 4', 'b- 1', 'c- 8', 'd- 2', 'e- 13', 'f- 5', 'g- 9', 'h- 11', 'i- 6', 'j- 20', 'k- 19',
                 'l- 18', 'm- 21', 'n- 17', 'o- 16', 'p- 14', 'q- 3', 'r- 7', 's- 10', 't- 12', 'u- 15']
for category in temp_category:
    path = 'Mapping ' + SUBJECT + category
    for folder in folder_labels:
        try:
            os.makedirs(path + '\\' + folder)
        except OSError:
            print("Failed to create folders!")

def convert_timestep_framenum(timestep):
    minutes_seconds_pps = [int(i) for i in timestep.split('-')]
    return minutes_seconds_pps[0] * 60 * 60 + minutes_seconds_pps[1] * 60 + minutes_seconds_pps[2]

for path, subdirs, files in os.walk("F:/LF/LF " + SUBJECT):
    for file in files:
        #Ignore not-image files
        if ".jpg" not in file and ".png" not in file:
            continue

        #Get image path relative to the root, extract frame number
        full_path = os.path.join(path, file).split("\\")
        relative_path = full_path[1] + "/" + full_path[2]
        relative_path_split = relative_path.split("-")
        read_frame_num = int(relative_path_split[FRAME_NUM_INDEX])
        face_frame_num = 0

        if(CROP_OFFSET1 is 0): #Timestamp is valid, use it instead of custom add to frame num
            face_frame_num = convert_timestep_framenum((relative_path_split[5] + '-' + relative_path_split[6] + '-' + relative_path_split[7])[:-4])
        else:
            #Determine face-cam frame number based on whether it is the first or second video
            if LABEL_VIDEO2 not in file:
                face_frame_num = read_frame_num + CROP_OFFSET1
            else:
                face_frame_num = read_frame_num + CROP_OFFSET2#Offset due to cropping previously

        #Use the offset for the mirror-cam relative to the face-cam
        mirror_frame_num = face_frame_num + MIRROR_CROP_OFFSET#Offset due to camera start

        use_second_video_face = False
        use_second_video_mirror = False
        if face_frame_num > 57719:
            print("ERROR - too big face number for: ", full_path, face_frame_num)
            use_second_video_face = True
            face_frame_num -= 57719
        if mirror_frame_num > 57719:
            print("ERROR - too big mirror number for: ", full_path, mirror_frame_num)
            use_second_video_mirror = True
            mirror_frame_num -= 57719
            #mirror_frame_num -= 2 #Just for SUBJECT 7-15, remove for others (I don't know why, but 2 frames skipped!)
        #Just for SUBJECT 12-1, the below if statement. This is because the offset between face/mirror was -36k
        if mirror_frame_num < 0 and MIRROR_CROP_OFFSET < -1000:
            print("ERROR - negative mirror number for: ", full_path, mirror_frame_num)
            use_second_video_mirror = True
            mirror_frame_num += 57720
        if LABEL_VIDEO2 in file:
            use_second_video_face = True
            use_second_video_mirror = True
        face_frame = None
        mirror_frame = None

        #Get corresponding face/mirror images from correct videos
        if not use_second_video_face:
            cap.set(1, face_frame_num)
            ret, face_frame = cap.read()
        else:
            cap2.set(1, face_frame_num)
            ret, face_frame = cap2.read()
        if not use_second_video_mirror:
            cap_mirror.set(1, mirror_frame_num)
            ret, mirror_frame = cap_mirror.read()
        else:
            cap2_mirror.set(1, mirror_frame_num)
            ret, mirror_frame = cap2_mirror.read()

        #Reconstruct pics with proper frame number
        relative_path_split[FRAME_NUM_INDEX] = str(face_frame_num)

        print('Mapping ' + SUBJECT + '\\Face\\' + ''.join([piece + '-' for piece in relative_path_split])[:-1])
        try:
            cv2.imwrite('Mapping ' + SUBJECT + '\\Face\\' + ''.join([piece + '-' for piece in relative_path_split])[:-1], face_frame)
        except:
            capNoSound = cv2.VideoCapture('C:/Users/uxm170001/Documents/Adobe/Premiere Pro/14.0/GH030106.AVI')
            capNoSound.set(1, face_frame_num)
            ret, face_frame = capNoSound.read()
            cv2.imwrite('Mapping ' + SUBJECT + '\\Face\\' + ''.join([piece + '-' for piece in relative_path_split])[:-1], face_frame)
        cv2.imwrite('Mapping ' + SUBJECT + '\\Mirror\\' + ''.join([piece + '-' for piece in relative_path_split])[:-1], mirror_frame)
