import cv2
import numpy

img_puppy = cv2.imread("puppy_noisy.jpg")
#img_

# 3.2.1
img_puppy_copy1 = cv2.GaussianBlur(img_puppy, (3, 3), 0)
img_puppy_copy2 = cv2.GaussianBlur(img_puppy, (9, 9), 0)
img_puppy_copy3 = cv2.GaussianBlur(img_puppy, (27, 27), 0)

cv2.imshow("Original Noisy Puppy", img_puppy)
cv2.imshow("3,3", img_puppy_copy1)
cv2.imshow("9, 9", img_puppy_copy2)
cv2.imshow("27, 27", img_puppy_copy3)

cv2.imwrite("puppy_noise.jpeg", img_puppy)
cv2.imwrite("puppy_3.jpeg", img_puppy_copy1)
cv2.imwrite("puppy_9.jpeg", img_puppy_copy2)
cv2.imwrite("puppy_27.jpeg", img_puppy_copy3)



# 3.2.2

img_puppy_noise = img_puppy
img_puppy = cv2.imread("puppy.jpg")
img_field = cv2.imread("field.jpg")

img_puppy_edge = cv2.Canny(img_puppy, 50, 50)
img_puppy_noise_edge = cv2.Canny(img_puppy_noise, 50, 50)
img_field_edge = cv2.Canny(img_field, 50, 50)
cv2.imshow("Original Puppy Canny", img_puppy_edge)
cv2.imshow("Noisy Puppy Canny", img_puppy_noise_edge)
cv2.imshow("Field Original", img_field)
cv2.imshow("Field Edge", img_field_edge)

cv2.imwrite("puppy_noise_edge.jpg", img_puppy_noise_edge)
cv2.imwrite("puppy_edge.jpg", img_puppy_edge)
cv2.imwrite("field_edge.jpg", img_field_edge)

cv2.waitKey()