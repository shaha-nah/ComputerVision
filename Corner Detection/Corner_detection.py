import cv2
import numpy as np

img_file = 'furniture.jpg'
img = cv2.imread(img_file, cv2.IMREAD_COLOR)

imgDim = img.shape
dimA = imgDim[0]
dimB = imgDim[1]

# RGB to Gray scale conversion
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# Noise removal with iterative bilateral filter(removes noise while preserving edges)
noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
# Thresholding the image
ret,thresh_image = cv2.threshold(noise_removal,220,255,cv2.THRESH_OTSU)
th = cv2.adaptiveThreshold(noise_removal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Applying Canny Edge detection
canny_image = cv2.Canny(th,250,255)
canny_image = cv2.convertScaleAbs(canny_image)

# dilation to strengthen the edges
kernel = np.ones((3,3), np.uint8)
# Creating the kernel for dilation
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
#np.set_printoptions(threshold=np.nan)

contours, h = cv2.findContours(dilated_image, 1, 2)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:1]


corners    = cv2.goodFeaturesToTrack(thresh_image,6,0.06,25)
corners    = np.float32(corners)

for item in corners:
    x,y    = item[0]
    cv2.circle(img,(x,y),10,255,-1)
cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
cv2.imshow("Corners",img)
cv2.waitKey()