import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
pen_cascade = cv2.CascadeClassifier('classifier/pen_2.xml')
handdrill_cascade = cv2.CascadeClassifier('classifier/handDrill.xml')
strawberry_cascade = cv2.CascadeClassifier('classifier/strawberry_classifier.xml')
apple_cascade = cv2.CascadeClassifier('classifier/apple.xml')


cap = cv2.VideoCapture(0)

while (True):
    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,
                        scaleFactor=1.3,
                        minNeighbors=5,
                        minSize=(50, 50))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img, 'face', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))
    
    pens = pen_cascade.detectMultiScale(gray,
                        scaleFactor=1.2,
                        minNeighbors=5,
                        minSize=(50, 50))
    for (x,y,w,h) in pens:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img, 'pen', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))

    handdrill = handdrill_cascade.detectMultiScale(gray,
                        scaleFactor=5,
                        minNeighbors=10,
                        minSize=(50, 50))
    for (x,y,w,h) in handdrill:
        if (w<300):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img, 'hand drill', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))

    '''apple = apple_cascade.detectMultiScale(gray,
                        scaleFactor=5,
                        minNeighbors=10,
                        minSize=(50, 50))
    for (x,y,w,h) in apple:
        if (w<300):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img, 'apple', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))'''

    cv2.imshow('img',img)
    ch = cv2.waitKey(5) & 0xFF
    if ch == 27:  
        break
cv2.destroyAllWindows()
