import matplotlib.pyplot as plt
import numpy as np
from os.path import expanduser

np.save(expanduser("~/Desktop/test.npy"), np.array([1, 2]))
plt.plot([1, 2])
plt.show()