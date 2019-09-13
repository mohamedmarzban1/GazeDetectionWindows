import os
from shutil import copyfile
import pandas as pd

# --- CONSTANTS --- #

ID_FILE = "C:/Users/aps170830/Desktop/naofal-lab/data/2019-6-14 - Copy/FE2019-6-14/id.csv"
ANGLES_ID = "C:/Users/aps170830/Desktop/naofal-lab/data/2019-6-14 - Copy/FE2019-6-14/AnglesIDFile.csv"

LAST_FRAME_PRE_NONE = 52  # last value before none
FIRST_FRAME_POST_NONE = 101  # first value after none sequence

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


def correct_id_file(path, output_path, from_frame, gap_end_frame):
    if os.path.isfile(path) is False:
        print(path, " is not a valid directory.")
        return

    df = pd.read_csv(path, sep=',')

    offset = gap_end_frame - from_frame - 1

    for i in range(0, len(df)):
        id = df[' ImageID'][i]

        parts = deconstruct_id(id)

        frameid = int(parts[3])

        if frameid > gap_end_frame:

            suffix = id[-8:-4]
            print('old', id)

            new_id = construct_id(
                parts[0], parts[1], parts[2], frameid - offset, suffix)

            print(new_id)
            df[' ImageID'][i] = new_id
        else:
            print(frameid)

    df.to_csv(output_path, line_terminator='\n\n', index=False)


def correct_angles_file(path, output_path, from_frame, gap_end_frame):
    if os.path.isfile(path) is False:
        print(path, " is not a valid directory.")
        return

    df = pd.read_csv(path, sep='\t')

    offset = gap_end_frame - from_frame - 1

    for i in range(0, len(df)):
        id = df['ImageID'][i]

        parts = deconstruct_id(id)

        frameid = int(parts[3])

        if frameid > gap_end_frame:

            suffix = id[-8:-4]
            new_id = construct_id(
                parts[0], parts[1], parts[2], frameid - offset, suffix)

            print(new_id)
            df['ImageID'][i] = new_id
        else:
            print(frameid)

    df.to_csv(output_path, line_terminator='\n\n',
              sep="\t", na_rep='nan', index=False)


def main():
    correct_id_file(ID_FILE, "C:/Users/aps170830/Desktop/naofal-lab/data/2019-6-14 - Copy/FE2019-6-14/id - Copy.csv",
                    LAST_FRAME_PRE_NONE, FIRST_FRAME_POST_NONE)
    correct_angles_file(ANGLES_ID, "C:/Users/aps170830/Desktop/naofal-lab/data/2019-6-14 - Copy/FE2019-6-14/AnglesIDFile - Copy.csv",
                        LAST_FRAME_PRE_NONE, FIRST_FRAME_POST_NONE)


main()
