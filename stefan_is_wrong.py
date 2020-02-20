import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(0, 2 * np.pi, 100)
x, y = np.meshgrid(x, y)

fig = plt.figure()

fig.add_subplot(2, 3, 1)
z = np.sin(np.sqrt(np.multiply(x, x) + np.multiply(y, y)))
z = z - np.mean(z)
plt.imshow(z)


fig.add_subplot(2, 3, 2)
f = np.abs(np.fft.fft2(z))
plt.imshow(f)


fig.add_subplot(2, 3, 3)
fs = np.fft.fftshift(f)
plt.imshow(fs)



fig.add_subplot(2, 3, 4)
z = np.sin(np.sqrt(np.multiply(x, x) + np.multiply(y, y)) * 20)
z = z - np.mean(z)
plt.imshow(z)


fig.add_subplot(2, 3, 5)
f = np.abs(np.fft.fft2(z))
plt.imshow(f)


fig.add_subplot(2, 3, 6)
fs = np.fft.fftshift(f)
plt.imshow(fs)

plt.show()




