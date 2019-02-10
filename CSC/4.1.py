import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D

# image setup
pup = cv2.imread("puppy.jpg")
pup_small = cv2.resize(pup, (0, 0), fx=0.125, fy=0.125)
graySmall = cv2.cvtColor(pup_small, cv2.COLOR_BGR2GRAY) # convert to greyscale

# coord array
xx, yy = np.mgrid[0:graySmall.shape[0], 0:graySmall.shape[1]]

# fig
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, graySmall, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)

# 2d dft
f_graySmall = np.fft.fft2(graySmall)
fig = plt.figure()
#ax = fig.gca(projection='3d')
Y = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
X = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
X, Y = np.meshgrid(X, Y)
#ax.plot_surface(X, Y, np.fft.fftshift(np.abs(f_graySmall)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.plot(X,Y,'b')
plt.savefig('4.1_fig2d.jpg')

# show/ plot the magnitude and the log(magnitude +1)
magnitudeImage = np.fft.fftshift(np.abs(f_graySmall))
magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
magnitudeImage = ski.img_as_ubyte(magnitudeImage)
cv2.imwrite("magnitude.jpg", magnitudeImage)
cv2.imshow('mag plot', magnitudeImage)
logMagnitudeImage = np.fft.fftshift(np.log(np.abs(f_graySmall)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
cv2.imwrite("magnitude_plus.jpg", logMagnitudeImage)
cv2.imshow('log mag plot', logMagnitudeImage)
cv2.waitKey(0)

# save output

