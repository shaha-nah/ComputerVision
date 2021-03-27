import cv2
 
def funcCan(thresh1=0):
    thresh1 = cv2.getTrackbarPos('thresh1', 'canny')
    thresh2 = cv2.getTrackbarPos('thresh2', 'canny')
    edge = cv2.Canny(img, thresh1, thresh2)
    cv2.imshow('canny', edge)
 
if __name__== '__main__':
 
    original=cv2.imread("./fish/F1.jpg",1)
    img=original.copy()
    img=cv2.GaussianBlur(img,(11,11),0)
 
    cv2.namedWindow('canny')
 
 
    thresh1=100
    thresh2=1
    cv2.createTrackbar('thresh1','canny',thresh1,255,funcCan)
    cv2.createTrackbar('thresh2','canny',thresh2,255,funcCan)
    funcCan(0)
    cv2.imshow('Frame',original)
 
 
    cv2.waitKey(0)
 
 
cv2.destroyAllWindows()