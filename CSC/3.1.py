import cv2
import numpy

img = cv2.imread("puppy.jpg")
img_copy = cv2.imread("puppy.jpg")
fil = numpy.zeros((5, 5, 1))
fil[3, 3, 0] = 1
fill = cv2.GaussianBlur(fil, (5, 5), 0)

img_copy = cv2.GaussianBlur(img_copy, (5, 5), 0)
cv2.imshow("Original", img)
cv2.imshow("Filter", fill)
cv2.imshow("Filtered Image", img_copy)

cv2.imwrite("3.1_original.jpg", img)
cv2.imwrite("3.1-filtered.jpg", img_copy)
cv2.imwrite("3.1_filter.jpg", fill)
cv2.waitKey()
