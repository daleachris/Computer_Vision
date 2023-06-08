# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:15:00 2023

@author: u55530
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# %% Varibels
i=0
smallest_deviation=10
good_enough=False
target_distance=4.7
target_width=.3795
is_ok=False
camera_matrix_array=[]
distance1_array=[]
distance2_array=[]
distance3_array=[]

# %% CHECKERBOARD finding and Camera calibration
# CHECKERBOARD size
CHECKERBOARD = (7,7)
# termination criteria
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# search images inside of folder
images = glob.glob('*.jpg')
_img_shape = None

# Start with the evaluation of each image within the folder Single frame calibration!
print('####################################_START_#############################')
for fname in images:
    print('#######################################################################')  
    i=i+1;
    #Display loop process in %
    print('Read a new frame #:',fname)
    process= i/len(images)*100
    print('process in %:',process )
    #Check image
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('Corner points found')
        #objpoints.append(objp) not needed here because of single frame calibration!
        objpoints=[objp]
        corners2 =cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        #imgpoints.append(corners2) not needed here because of single frame calibration!
        imgpoints=[corners2]
        
        # Draw and display the corners uncomment below if needed
        #cv2.drawChessboardCorners(img, (7 ,7), corners2, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        
        #Higlight Cornerpoint within the image uncomment if needed
        # corner_point = (710, 576)
        # # Draw a circle or marker at the corner point
        # radius = 5  # Adjust the size of the circle/marker as desired
        # color = (0, 255, 0)  # Set the color (in BGR format) of the circle/marker
        # thickness = 2  # Adjust the thickness of the circle/marker as desired
        # cv2.circle(img, corner_point, radius, color, thickness)   
        # Display the image with highlighted corner point
        # cv2.imshow('Highlighted Corner Point', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    else:
        print('!!!!!!!!!!!!__Corner points not found__!!!!!!!!!!!!')
        
    #Calculate camera_matrix --> single frame calibration 
    ret, camerea_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #Save for each frame the camera_matrix 
    camera_matrix_array.append(camerea_matrix)
    
# %% Distance calculation for each frame
    Corner1_cord=corners2[0]
    Corner1_cord=np.append(Corner1_cord,1)
    Corner2_cord=corners2[6]
    Corner2_cord=np.append(Corner2_cord,1)
    Kinv=np.linalg.inv(camerea_matrix)
    z=target_width/np.linalg.norm((Kinv.dot(Corner1_cord-Corner2_cord)))
    print('z is',z)
    L1=z*Kinv.dot(Corner1_cord)
    L2=z*Kinv.dot(Corner2_cord)
    distance1= np.sqrt(np.square(np.linalg.norm(L1)-np.square(target_width/2)))
    distance2= np.sqrt(np.square(np.linalg.norm(L2)-np.square(target_width/2)))
    distance3= (distance1+distance2)/2
    print('distance1 is',distance1)
    print('distance2 is',distance2)
    print('distance3 is',distance3)
    distance1_array.append(distance1)
    distance2_array.append(distance2)
    distance3_array.append(distance3)

    deviation=abs(distance3-target_distance)
    print('deviation in[cm]:',100*deviation)   
# %% Check for the smallest_deviation
    if deviation < smallest_deviation:
        smallest_deviation=deviation
        best_frame=fname
        array_index=i-1
        print('new_smallest_deviation_found in[cm]:',100*smallest_deviation)   
    
# %% Print best candidate

print('#############################_BEST_DIST_MIDDLE_########################')
print('final_smallest_deviation in[cm]:',100*smallest_deviation)   
print('best_frame',best_frame)
print('nearest_distance1 in[m] @ frame#:' ,best_frame,'is',distance1_array[array_index])
print('nearest_distance2 in[m] @ frame#:' ,best_frame,'is',distance2_array[array_index])
print('nearest_distance3 in[m] @ frame#:' ,best_frame,'is',distance3_array[array_index])
print('#############################_BEST_DIST_FAR_###########################')
camerea_matrix=camera_matrix_array[array_index]
img = cv2.imread(best_frame)
#cv2.drawChessboardCorners(img, (7 ,7), corners2, ret)
cv2.imshow(best_frame, img)
cv2.waitKey()
cv2.destroyAllWindows()     

    
# %% Image Far away   

PATH1="C:/Users/U55530/Desktop/TMP/Master/Frames_videoHD1_calibration/Three_distances/frame1853.jpg"
img = cv2.imread(PATH1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
# If found, add object points, image points (after refining them)
if ret == True:
    print('Corner points found')
    objpoints.append(objp)
    corners2 =cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
    imgpoints.append(corners2)
    
    
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (7,7), corners2, ret)
    cv2.imshow('frame1853.jpg', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


Corner1_cord=corners2[0]
Corner1_cord=np.append(Corner1_cord,1)
Corner2_cord=corners2[6]
Corner2_cord=np.append(Corner2_cord,1)
Kinv=np.linalg.inv(camerea_matrix)
z=target_width/np.linalg.norm((Kinv.dot(Corner1_cord-Corner2_cord)))
print('z is',z)
L1=z*Kinv.dot(Corner1_cord)
L2=z*Kinv.dot(Corner2_cord)
distance1= np.sqrt(np.square(np.linalg.norm(L1)-np.square(target_width/2)))
distance2= np.sqrt(np.square(np.linalg.norm(L2)-np.square(target_width/2)))
distance3= (distance1+distance2)/2
print('distance1 in[m] @ frame#: frame1853.jpg is',distance1)
print('distance2 in[m] @ frame#: frame1853.jpg is',distance2)
print('distance3 in[m] @ frame#: frame1853.jpg is',distance3)


# %% Image Near 
print('#############################_BEST_DIST_NEAR_##########################')
PATH1="C:/Users/U55530/Desktop/TMP/Master/Frames_videoHD1_calibration/Three_distances/frame893.jpg"
img = cv2.imread(PATH1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
# If found, add object points, image points (after refining them)
if ret == True:
    print('Corner points found')
    objpoints.append(objp)
    corners2 =cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
    imgpoints.append(corners2)
    
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (7,7), corners2, ret)
    cv2.imshow('frame893.jpg', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


Corner1_cord=corners2[0]
Corner1_cord=np.append(Corner1_cord,1)
Corner2_cord=corners2[6]
Corner2_cord=np.append(Corner2_cord,1)
Kinv=np.linalg.inv(camerea_matrix)
z=target_width/np.linalg.norm((Kinv.dot(Corner1_cord-Corner2_cord)))
print('z is',z)
L1=z*Kinv.dot(Corner1_cord)
L2=z*Kinv.dot(Corner2_cord)
distance1= np.sqrt(np.square(np.linalg.norm(L1)-np.square(target_width/2)))
distance2= np.sqrt(np.square(np.linalg.norm(L2)-np.square(target_width/2)))
distance3= (distance1+distance2)/2
print('distance1 in[m] @ frame#: frame893.jpg is',distance1)
print('distance2 in[m] @ frame#: frame893.jpg is',distance2)
print('distance3 in[m] @ frame#: frame893.jpg is',distance3)
print('#######################################################################')
print('################################_END_##################################')


# img = cv2.imread("C:/Users/U55530/Desktop/TMP/Master/Frames_videoHD1_calibration/Three_distances/frame893.jpg")
# h,  w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camerea_matrix, dist_coef, (w,h), -1, (w,h))

#  # undistort
# dst = cv2.undistort(img, camerea_matrix, dist_coef, None, newcameramtx)
#  # crop the image
# #x, y, w, h = roi
# #dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png', dst)






 