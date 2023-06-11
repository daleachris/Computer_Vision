# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:54:01 2023

@author: u55530
"""

import cv2
# %% Extract Frames of Video: videoHD1.avi
vidcap = cv2.VideoCapture('videoHD1.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("./Frames_videoHD1_calibration/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame #:',count, 'successfull?: ', success)
  count += 1

# %% Extract Frames of Video: videoWW1_calibration.avi
vidcap = cv2.VideoCapture('videoWW1_calibration.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("./Frames_videoWW1_calibration/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame #:',count, 'successfull?: ', success)
  count += 1
