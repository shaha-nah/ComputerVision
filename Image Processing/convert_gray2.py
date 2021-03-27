import cv2
  
image = cv2.imread('../InputImages/fruits.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("../InputImages/modfruits.jpg", gray)
cv2.imshow('../InputImages/Original image',image)
cv2.imshow('../InputImages/Gray image', gray)
  
cv2.waitKey(0)
cv2.destroyAllWindows()