import cv2
import numpy as np

# Load the image
# specify image name or path
img = cv2.imread('../InputImages/uom.jpg')

# Check the datatype of the image
print(img.dtype)

# Subtract the img from max value(calculated from dtype)
img_neg = 255 - img

# Show the image
cv2.imshow('negative',img_neg)
cv2.waitKey(0)