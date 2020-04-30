# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:59:21 2019
Concatenate Videos
@author: mfm160330
"""
#import ffmpy
import os


os.system("ffmpeg -safe 0 -f concat -i 20190530001\road_list.txt -c copy 20190530001\road.mp4")

