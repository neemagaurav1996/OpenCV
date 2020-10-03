import cv2
import imutils
from skimage.metrics import structural_similarity
import numpy as np
from skimage import measure

# load images
imageA = cv2.imread("imageA.jpg")
imageB = cv2.imread("imageB.jpg")

# convert to gray
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = structural_similarity(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

print ("Thresholding...")
# threshold the image - Inverted binary followed by OTSU
thresh1 = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY_INV)[1]
thresh1 = cv2.threshold(thresh1, 0, 255, cv2.THRESH_OTSU)[1]

print ("Smoothning...")
# smoothing (blurring) to reduce high frequency noise
blurred = cv2.GaussianBlur(thresh1, (11, 11), 0)

print ("Thresholding...")
# threshold the image to reveal light regions in the blurred image
thresh2 = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]


print ("Closing...")
h,w = thresh2.shape
#Kernel to only dilate in horizontal direction
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2*w,1))
# Closing is Dilation followed by Erosion. 
# It is useful in closing small holes inside the foreground objects, or small black points on the object.
closed = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

# Find contours
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
	# compute the bounding box of the contour 
	# Then draw the bounding box on both input images to represent where the two images differ
	(x, y, w, h) = cv2.boundingRect(c)
	padding = 20
	# Capture only horizontal rectangles
	if(w>h):
		cv2.rectangle(imageA, (x, y-padding), (x+w, y+h+padding), (0,0,255), 2)
		cv2.rectangle(imageB, (x, y-padding), (x+w, y+h+padding), (0,0,255), 2)

cv2.imwrite("imageA_out.jpg",imageA)
cv2.imwrite("imageB_out.jpg",imageB)