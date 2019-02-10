import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


img_puppy = cv2.imread("puppy.jpg")
small_puppy = cv2.resize(img_puppy, (0, 0), fx=0.25, fy=0.25)
graySmall_puppy = cv2.cvtColor(small_puppy, cv2.COLOR_BGR2GRAY)

img_puppy_noisy = cv2.imread("puppy_noisy.jpg")
small_puppy_noisy = cv2.resize(img_puppy, (0, 0), fx=0.25, fy=0.25)
graySmall_puppy_noisy = cv2.cvtColor(small_puppy, cv2.COLOR_BGR2GRAY)

img_field = cv2.imread("field.jpg")
small_field = cv2.resize(img_field, (0, 0), fx=0.25, fy=0.25)
graySmall_field = cv2.cvtColor(small_field, cv2.COLOR_BGR2GRAY)


F2_graySmall_puppy = np.fft.fft2(graySmall_puppy.astype(float))
fig = plt.figure()
ax = fig.gca(projection='3d')
Y = (np.linspace(-int(graySmall_puppy.shape[0]/2), int(graySmall_puppy.shape[0]/2)-1, graySmall_puppy.shape[0]))
X = (np.linspace(-int(graySmall_puppy.shape[1]/2), int(graySmall_puppy.shape[1]/2)-1, graySmall_puppy.shape[1]))
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, np.fft.fftshift(np.abs(F2_graySmall_puppy)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("4.2_puppyfig.jpg")


fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.plot_surface(X, Y, np.fft.fftshift(np.log(np.abs(F2_graySmall_puppy)+1)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("4.2_puppyfigplus.jpg")

# fig4 = plt.figure()
col = int(small_puppy.shape[1]/2)
colData = small_puppy[0:small_puppy.shape[0], col, 0]
F_colData = np.fft.fft2(colData)

for k in range(0, int(len(colData)/2)):
    # Truncate frequencies and then plot the resulting function in real space
    Trun_F_colData = F_colData.copy()
    Trun_F_colData[k+1:len(Trun_F_colData)-k] = 0
    trun_colData = np.fft.ifft(Trun_F_colData)
    # Plot
    xvalues = np.linspace(0, len(trun_colData) - 1, len(trun_colData))
    plt.plot(xvalues, colData, 'b')
    plt.plot(xvalues, trun_colData, 'r')
    plt.title('k = 0 : ' + str(k))

plt.savefig("inv.jpg")

F2_graySmall_field = np.fft.fft2(graySmall_field.astype(float))
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
YY = (np.linspace(-int(graySmall_field.shape[0]/2), int(graySmall_field.shape[0]/2)-1, graySmall_field.shape[0]))
XX = (np.linspace(-int(graySmall_field.shape[1]/2), int(graySmall_field.shape[1]/2)-1, graySmall_field.shape[1]))
XX, YY = np.meshgrid(XX, YY)
ax2.plot_surface(XX, YY, np.fft.fftshift(np.abs(F2_graySmall_field)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("4.2_feildfig.jpg")

fig6 = plt.figure()
ax6 = fig6.gca(projection='3d')
ax6.plot_surface(XX, YY, np.fft.fftshift(np.log(np.abs(F2_graySmall_field)+1)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("4.2_feildfigplus.jpg")

graySmall_puppy_noisy
F2_graySmall_puppy_noisy = np.fft.fft2(graySmall_puppy_noisy.astype(float))
fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')
YYY = (np.linspace(-int(graySmall_puppy_noisy.shape[0]/2), int(graySmall_puppy_noisy.shape[0]/2)-1, graySmall_puppy_noisy.shape[0]))
XXX = (np.linspace(-int(graySmall_puppy_noisy.shape[1]/2), int(graySmall_puppy_noisy.shape[1]/2)-1, graySmall_puppy_noisy.shape[1]))
XXX, YYY = np.meshgrid(XXX, YYY)
ax3.plot_surface(XXX, YYY, np.fft.fftshift(np.abs(F2_graySmall_puppy_noisy)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("4.2_puppynoisyfig.jpg")

fig7 = plt.figure()
ax7 = fig7.gca(projection='3d')
ax7.plot_surface(XXX, YYY, np.fft.fftshift(np.log(np.abs(F2_graySmall_puppy_noisy)+1)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("4.2_puppynoisyfigplus.jpg")