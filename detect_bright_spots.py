#Detect bright spot in the images
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

image = cv2.imread("diff_image_inv.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# smoothing (blurring) to reduce high frequency noise
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

# Closing is Dilation followed by Erosion. 
# It is useful in closing small holes inside the foreground objects, or small black points on the object.
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None, iterations=50)

cv2.imshow("Blurred", blurred)
cv2.imshow("Threshed", thresh)
cv2.imshow("Closed", closed)
cv2.waitKey(0)