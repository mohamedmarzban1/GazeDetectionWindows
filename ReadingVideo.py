"""
Created on Thu Nov  8 19:14:19 2018

@author: mfm160330
"""
import numpy as np
import cv2
import os
import argparse

'''
    Parsing 3 arguments from the command line
        >   video_path  str     path to the video file to split into images
        >   write_path  str     path of the output folder to spill images into
        >   frame_start int     starting frame in the video to begin outputting images
'''


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        'video_path', help='Video to split into images', type=str)
    args.add_argument(
        'write_path', help='EXISTING folder to write the output images to', type=str)
    args.add_argument(
        'image_prefix', help='Image prefix for output images', type=str
    )
    args.add_argument(
        'frame_start', help='Start frame to begin outputting images from', type=int, default=0)
    return args.parse_args()


def GetFrameIds(frame):
    return (np.floor(frame/3600), np.floor((frame % 3600) / 60), frame % 60)


def ReadingVideo(video_name, write_path, image_prefix, frame_start):
    VideoName = video_name
    WriteLocation = write_path
    ImagesName = image_prefix

    #==== Intializing the crop values in HD===================#
    cropFlag = 1
    # 180 #(face14Oct18: 50, face1Dec: 100   mirror:150 ) #Vertical Crop Start
    vCropS = 80
    vCropE = 1000  # (face14Oct18: 1000,  mirror:950) #Vertical crop end
    hCropS = 500  # (face14Oct18: 500,  mirror:800)
    hCropE = 1550  # (face14Oct18: 1600, mirror:1650)

    ##========Read Video =======##
    cap = cv2.VideoCapture(VideoName)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1)

    if not cap.isOpened():
        print("Unable to open video %s" % (video_name))
        exit()

    print(write_path)

    video_size = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    count = 0
    frame_well_read = True
    while(cap.isOpened() & frame_well_read & count <= video_size):
        frame_well_read, frame = cap.read()

        if frame is None:
            print("%d is a null frame" % (count))
            continue

        frameCropped = frame[vCropS:vCropE, hCropS:hCropE]

        if count < frame_start:
            count = count + 1
            continue

        # cv2.imshow('frame',frame) # PlZ comment me before running (To display the image)
        # cv2.imshow('frameCropped',frameCropped) # PlZ comment me before running (To display the image)
        # cv2.waitKey(0) # PlZ comment me before running
        framesSinceStart = count - frame_start
        (minName, secName, secFrameName) = GetFrameIds(framesSinceStart)

        name = WriteLocation+"/"+ImagesName + \
            "-%d-%d-%d-%d.jpg" % (framesSinceStart,
                                  minName, secName, secFrameName)

        print(name)
        cv2.imwrite(name, frameCropped)
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Finished without Errors")


if __name__ == "__main__":
    prg_args = parse_args()
    ReadingVideo(prg_args.video_path, prg_args.write_path,
                 prg_args.image_prefix, prg_args.frame_start)
