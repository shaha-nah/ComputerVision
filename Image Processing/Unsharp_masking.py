import cv2
import numpy as np
from matplotlib import pyplot as plt
 
# Load the image
image = cv2.imread("../InputImages/food.jpeg")

# Blur the image
gauss = cv2.GaussianBlur(image, (7,7), 0)

# Apply Unsharp masking
unsharp_image = cv2.addWeighted(image, 2, gauss, -1, 0)

plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(unsharp_image),plt.title('Sharpened')
plt.xticks([]), plt.yticks([])
plt.show()