# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

class ImageFeature(object):
    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        self.nameFile = nameFile
        self.shape = shape
        self.imageBinary = imageBinary
        self.kp = kp
        self.desc = desc
        self.matchingWebcam = []
        self.matchingDatabase = []

    def clearMatchingMutuos(self):
        self.matchingWebcam = []
        self.matchingDatabase = []

def loadModel():
    dataBase = dict([('SIFT', [])])
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
    #sift = cv2.xfeatures2d.SURF_create(400)

    for imageFile in os.listdir("train-image"):
        colorImage = cv2.imread("train-image/" + str(imageFile))
        currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)

        kp, desc = sift.detectAndCompute(currentImage, None)
        dataBase["SIFT"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
    return dataBase
    
def findMatching(selectedDataBase, desc, kp):
    for img in selectedDataBase:
        img.clearMatchingMutuos()
        for i in range(len(desc)):
             distanceListFromWebCam = np.linalg.norm(desc[i] - img.desc, axis=-1)
             candidatoDataBase = distanceListFromWebCam.argmin() 
             distanceListFromDataBase = np.linalg.norm(img.desc[candidatoDataBase] - desc,axis=-1)
             candidatoWebCam = distanceListFromDataBase.argmin()
             if (i == candidatoWebCam):
                img.matchingWebcam.append(kp[i].pt)
                img.matchingDatabase.append(img.kp[candidatoDataBase].pt)
        img.matchingWebcam = np.array(img.matchingWebcam)
        img.matchingDatabase = np.array(img.matchingDatabase)
    return selectedDataBase

def calculateBestImage(selectedDataBase, projer, minInliers):
    if minInliers < 15:
        minInliers = 15
    bestIndex = None
    bestMask = None
    numInliers = 0
    for index, imgWithMatching in enumerate(selectedDataBase):
        _, mask = cv2.findHomography(imgWithMatching.matchingDatabase, imgWithMatching.matchingWebcam, cv2.RANSAC, projer)
        if not mask is None:
            countNonZero = np.count_nonzero(mask)
            if (countNonZero >= minInliers and countNonZero > numInliers):
                numInliers = countNonZero
                bestIndex = index
                bestMask = (mask >= 1).reshape(-1)
    if not bestIndex is None:
        bestImage = selectedDataBase[bestIndex]
        inliersWebCam = bestImage.matchingWebcam[bestMask]
        inliersDataBase = bestImage.matchingDatabase[bestMask]
        return bestImage, inliersWebCam, inliersDataBase
    return None, None, None
                
def calculateAffinityMatrix(bestImage, inliersDataBase, inliersWebCam, imgout):
    A = cv2.estimateRigidTransform(inliersDataBase, inliersWebCam, fullAffine=True)
    A = np.vstack((A, [0, 0, 1]))
    
    a = np.array([0, 0, 1], np.float)
    b = np.array([bestImage.shape[1], 0, 1], np.float)
    c = np.array([bestImage.shape[1], bestImage.shape[0], 1], np.float)
    d = np.array([0, bestImage.shape[0], 1], np.float)
    centro = np.array([float(bestImage.shape[0])/2, float(bestImage.shape[1])/2, 1], np.float)
       
    a = np.dot(A, a)
    b = np.dot(A, b)
    c = np.dot(A, c)
    d = np.dot(A, d)
    centro = np.dot(A, centro)
    
    areal = (int(a[0]/a[2]), int(a[1]/b[2]))
    breal = (int(b[0]/b[2]), int(b[1]/b[2]))
    creal = (int(c[0]/c[2]), int(c[1]/c[2]))
    dreal = (int(d[0]/d[2]), int(d[1]/d[2]))
    centroreal = (int(centro[0]/centro[2]), int(centro[1]/centro[2]))
    
    points = np.array([areal, breal, creal, dreal], np.int32)
    cv2.polylines(imgout, np.int32([points]),1, (255,255,255), thickness=2)
    draw_str(imgout, centroreal, bestImage.nameFile.upper())
    cv2.imshow('ImageDetector', bestImage.imageBinary)

def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),lineType=cv2.LINE_AA)
