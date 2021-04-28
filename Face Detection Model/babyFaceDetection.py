import numpy as np
import cv2 
import matplotlib.pyplot as plt

faceCascade = cv2.CascadeClassifier('./assets/classifier/haarcascade_frontalface_default.xml')

for i in range(1, 21):
    img = cv2.imread('./assets/images/'+str(i)+'.jpg')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        roiGray = imgGray[fy:fy + fh, fx:fx + fw]
        roiColor = img[fy:fy + fh, fx:fx + fw]
    
    cv2.imwrite('./assets/resultImages/'+str(i)+'.jpg', img)