import cv2

faceCascade = cv2.CascadeClassifier('./assets/classifier/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('./assets/classifier/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('./assets/classifier/haarcascade_smile.xml')

capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        roiGray = imgGray[fy:fy + fh, fx:fx + fw]
        roiColor = img[fy:fy + fh, fx:fx + fw]

        eyes = eyeCascade.detectMultiScale(roiGray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smiles = smileCascade.detectMultiScale(roiGray, 1.7, 22)
        for sx, sy, sw, sh in smiles:
            cv2.rectangle(roiColor, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.imshow("Face, Eyes and Smile Detection", img)
    key = cv2.waitKey(30) & 0xff

    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()