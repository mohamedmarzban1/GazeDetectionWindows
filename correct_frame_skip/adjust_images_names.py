import os
from shutil import copyfile

# --- CONSTANTS --- #

LF_FOLDER = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas/ADAS data/Labeled data Face/LF 2019-7-23"
G_FOLDER = "G:/FixedGazeImages/Gaze points Data/G2019-7-23"
FE_FOLDER = "C:/Users/mfm160330/OneDrive - The University of Texas at Dallas\ADAS data/FaceAndEyes/FE2019-7-23"

OUTPUT_DIR = "G:/CorrectedOutput"
G_NEW_Folder = "G2019-7-23"
LF_NEW_FOLDER = "LF 2019-7-23"

FE_NEW_Folder = "FE2019-7-23"
LAST_FRAME_PRE_NONE = 52  # last value before none
FIRST_FRAME_POST_NONE = 120  # first value after none sequence

# --- END CONSTANTS -- #


def frame_id(file_path):
    return int(deconstruct_id(file_path)[3])


def deconstruct_id(file_path):
    return file_path.split("-")


def construct_id(prefix, month, day, frameid, suffix=''):
    minute = frameid // 3600
    second = (frameid // 60) % 60
    frame = frameid % 60

    return '{}-{}-{}-{}-{}-{}-{}.jpg'.format(prefix, month, day, frameid, minute, second, str(frame) + suffix)


def correct_files(path, output_dir, from_frame, gap_end_frame, hasU=False):
    if os.path.isdir(path) is False:
        print(path, " is not a valid directory.")
        return

    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)
    frame_offset = gap_end_frame - from_frame - 1

    for file_name in os.listdir(path):
        if file_name.endswith(".jpg") is False:
            continue

        suffix = ''

        if hasU is True:
            suffix = file_name[-8:-4]

        id_parts = deconstruct_id(file_name)
        fid = int(id_parts[3])

        if fid > from_frame:
            print("Shifting file", file_name)
            new_frame_id = fid - frame_offset

            new_file_name = construct_id(
                id_parts[0], id_parts[1], id_parts[2], new_frame_id, suffix=suffix)

            copyfile(path + "/" + file_name, output_dir + "/" + new_file_name)
        else:
            print("Copying file", file_name)
            copyfile(path + "/" + file_name, output_dir + "/" + file_name)


def main():
    print("Processing G Folder")
    correct_files(G_FOLDER, OUTPUT_DIR + '/'+ G_NEW_Folder, LAST_FRAME_PRE_NONE, FIRST_FRAME_POST_NONE)

    print("Processing LF Folder")
    for tag in os.listdir(LF_FOLDER):
        print("LF/", tag)
        correct_files(LF_FOLDER + '/' + tag, OUTPUT_DIR + '/'+ LF_NEW_FOLDER +'/' + tag, LAST_FRAME_PRE_NONE, FIRST_FRAME_POST_NONE)

    print("Processing FE folder")
    for tag in os.listdir(FE_FOLDER):
        print("FE/", tag)
        if os.path.isdir(FE_FOLDER + '/' + tag) is False:
            continue
        for category in os.listdir(FE_FOLDER + '/' + tag):
            print("FE/", tag, "/", category)
            correct_files(FE_FOLDER + '/' + tag + '/' + category,
                            OUTPUT_DIR + '/' + FE_NEW_Folder + '/' + tag + '/' + category, LAST_FRAME_PRE_NONE, FIRST_FRAME_POST_NONE, True)


main()
