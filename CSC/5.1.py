import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D


img_puppy = cv2.imread("puppy.jpg")
small_puppy = cv2.resize(img_puppy, (0, 0), fx=0.25, fy=0.25)
graySmall = cv2.cvtColor(small_puppy, cv2.COLOR_BGR2GRAY)


F2_graySmall = np.fft.fft2(graySmall.astype(float))
Y = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
X = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
X, Y = np.meshgrid(X, Y)

plt.plot(X, Y)
plt.savefig("5.1origianl.jpg")

U = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
V = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(X*X + Y*Y)
# create x-points for plotting
xval = np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.25 * D.max()

# The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# and all D(u,v) where D(u,v) > 0 equal to 0
idealLowPass = D <= D0

print(graySmall.dtype)
FTgraySmall = np.fft.fft2(graySmall.astype(float))
FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealLowPass)
graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
cv2.imwrite("idealLowPass.jpg", idealLowPass)
cv2.imwrite("grayImageIdealLowpassFiltered.jpg", graySmallFiltered)

plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    # Create Butterworth filter of order n
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    # Apply the filter to the grayscaled image
    FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
    graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
    graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
    cv2.imwrite("grayImageButterworth-n" + str(n) + ".jpg", graySmallFiltered)
    # cv2.imshow('H', H)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite("butter-n" + str(n) + ".jpg", H)
    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')
    plt.savefig("buttermag-n" + str(n) + ".jpg")

plt.show()