#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import time
import numpy as np

import objrecogn 

def draw_str(dst,(x,y),s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),lineType=cv2.LINE_AA)

if __name__ == '__main__':

    def nothing(*arg):
        pass
    cv2.namedWindow('Features')
    cv2.namedWindow('ImageDetector')
    cv2.moveWindow('ImageDetector',1200,5)
    cv2.createTrackbar('projer', 'Features', 5, 10, nothing)
    cv2.createTrackbar('inliers', 'Features', 20, 50, nothing)
    cv2.createTrackbar('drawKP', 'Features', 0, 1, nothing)

    cap = cv2.VideoCapture(0)
    paused = False
    methodstr = 'None'

    dataBaseDictionary = objrecogn.loadModel()
    
    while True:
        if not paused:
            _,frame = cap.read()
            frame = cv2.resize(frame, (400, 300))
        if frame is None:
            print('End of video input')
            break

        methodstr = 'SIFT'
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=250)
        #detector = cv2.ORB_create()
                
        imgin = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        imgout = frame.copy()
        t1 = time.time()
        kp, desc = detector.detectAndCompute(imgin, None)
        #desc = np.float32(desc)
        selectedDataBase = dataBaseDictionary[methodstr]
        if len(selectedDataBase) > 0:
            imgsMatchingMutuos = objrecogn.findMatching(selectedDataBase, desc, kp)    
            minInliers = int(cv2.getTrackbarPos('inliers', 'Features'))
            projer = float(cv2.getTrackbarPos('projer', 'Features'))
            bestImage, inliersWebCam, inliersDataBase =  objrecogn.calculateBestImage(selectedDataBase, projer, minInliers)            
            if not bestImage is None:
               objrecogn.calculateAffinityMatrix(bestImage, inliersDataBase, inliersWebCam, imgout)
               
        t1 = 1000 * (time.time() - t1)  
        if desc is not None:
            if len(desc) > 0:
                dim = len(desc[0])
            else:
                dim = -1
       
        if (int(cv2.getTrackbarPos('drawKP', 'Features')) > 0):
            cv2.drawKeypoints(imgout, kp, imgout,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        draw_str(imgout, (20, 20),"Method {0}, {1} features found, desc. dim. = {2} ".format(methodstr, len(kp), dim))
        draw_str(imgout, (20, 40), "Time (ms): {0}".format(str(t1)))
        cv2.imshow('Features', imgout)

        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:  
            break

    cv2.destroyAllWindows()
