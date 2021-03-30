import cv2 as cv

# Load the classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier('./assets/classifier/haarcascade_frontalface_alt.xml')

for i in range(1,51):
    # Read image from your local file system
    # original_image = cv.imread('./assets/images/PicturesOriginal/'+str(i)+'.jpg')
    original_image = cv.imread('./assets/images/PicturesEdited/'+str(i)+'.jpg')

    # Convert color image to grayscale for Viola-11Jones
    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(grayscale_image)

    for (column, row, width, height) in detected_faces:
        cv.rectangle(
            original_image,
            (column, row),
            (column + width, row + height),
            (0, 255, 0),
            2
        )
    
    # cv.imwrite('./assets/images/ResultsOriginal/'+str(i)+'.png', original_image)
    cv.imwrite('./assets/images/ResultsEdited/'+str(i)+'.png', original_image)

# cv.imshow('Image', original_image)
# cv.waitKey(0)
# cv.destroyAllWindows()