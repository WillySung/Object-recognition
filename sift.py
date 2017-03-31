import cv2
import numpy as np
import os

def loadModel():
    dataBase = dict([('SIFT', [])])
    #sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
    sift = cv2.xfeatures2d.SURF_create()

    for imageFile in os.listdir("modelos"):
        trainImg = cv2.imread("modelos/" + str(imageFile),0)

        trainKP,trainDesc = detector.detectAndCompute(trainImg,None)
        trainDesc = np.float32(trainDesc)
        dataBase["SIFT"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
    return dataBase

if __name__ == '__main__':
    MIN_MATCH_COUNT = 30

    #detector = cv2.ORB_create()
    detector = cv2.xfeatures2d.SURF_create()
    FLANN_INDEX_KDTREE = 0
    flannParam = dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
    flann = cv2.FlannBasedMatcher(flannParam,{})

    cam = cv2.VideoCapture(0)

    while (1):
        ret,QueryImgBGR = cam.read()
        QueryImg = cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
        queryKP,queryDesc = detector.detectAndCompute(QueryImg,None)
        queryDesc = np.float32(queryDesc)
        matches = flann.knnMatch(queryDesc,trainDesc,k=2)

        goodMatch = []
        for m,n in matches:
            if(m.distance<0.75*n.distance):
                goodMatch.append(m)

        if(len(goodMatch)>MIN_MATCH_COUNT):
            tp = []
            qp = []
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp,qp = np.float32((tp,qp))
            H,status = cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
            h,w = trainImg.shape
            trainBorder = np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryBorder = cv2.perspectiveTransform(trainBorder,H)
            cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),3)
            cv2.putText(QueryImgBGR,'cup',(10,30),cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0),thickness=2)

        #else:
        #   print "Not enough matches - %d%d"%(len(goodMatch),MIN_MATCH_COUNT)

        cv2.imshow('result',QueryImgBGR)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()