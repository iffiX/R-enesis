import numpy as np
import matplotlib.pyplot as plt

x = np.log(np.linspace(1, 1e-10, 3000))
plt.plot(np.linspace(0, 3000, 3000), np.exp(x))
plt.show()
