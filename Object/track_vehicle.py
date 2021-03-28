import numpy as np
import cv2

videosrc = 'slow_traffic_small.mp4'
captura1 = cv2.VideoCapture(videosrc)
font = cv2.FONT_HERSHEY_SIMPLEX

shiTo_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,4))

ret, firstFrame = captura1.read()
firstGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
blurG = cv2.GaussianBlur(firstGray, (21, 21), 0)
while (1):
    cornersFrame = cv2.goodFeaturesToTrack(blurG, mask=None, **shiTo_params)
    ret, frame = captura1.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p_new = np.zeros_like(cornersFrame)
    p_new, status, error = cv2.calcOpticalFlowPyrLK(blurG,grayFrame, cornersFrame, p_new, **lk_params )

    good_new = p_new[status==1]
    good_old = cornersFrame[status==1]

    mask = np.zeros_like(frame)

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        cv2.line(mask, (a,b),(c,d),color[i].tolist(),2)
        cv2.circle(frame,(int(a),int (b)),4,(255,0,0),1)        

    lastFrame = cv2.add(frame,mask)    
    cv2.imshow('frame', lastFrame)
    blurG = grayFrame.copy()
    if cv2.waitKey(1) & 0xFF == ord('q') : 
        print ("PROGRAM CLOSED SUCCESSFULLY")
        break

captura1.release()
cv2.destroyAllWindows()