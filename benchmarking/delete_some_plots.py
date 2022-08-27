import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
core = np.load("/home/simon/Documents/delete_aligned_pattern/core.np.npy")

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(core[:, 0], core[:, 1], linestyle='', marker='.', markersize=1)

for i in range(10):
    other = np.load(f"/home/simon/Documents/delete_aligned_pattern/other{i}.np.npy")
    ax.plot(other[:, 0], other[:, 1], linestyle='', marker='.', markersize=1.5)

plt.show()
