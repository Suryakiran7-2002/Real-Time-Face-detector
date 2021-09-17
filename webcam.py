import cv2 as cv
import numpy as np

vid = cv.VideoCapture(0)

while True:

    t,img = vid.read()


    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    haar_cas = cv.CascadeClassifier('face_cascade.xml')

    faces = haar_cas.detectMultiScale(grey,1.1,11)


    for x,y,w,h in faces:

        cv.rectangle(img,(x,y),(x+w,y+w),(0,255,0),2)

    cv.imshow('webcam',img)

    if cv.waitKey(1) & 0xFF == 27:
        break