import numpy as np
from renesis.utils.shape_decomposition import ShapeDecomposition
from renesis.utils.virtual_shape import generate_cross
import matplotlib.pyplot as plt

cross = generate_cross(50, length_ratio=0.3)
sd = ShapeDecomposition(cross, iterations=2, kernel_size=10)
segments = sd.get_segments()
voxel = np.zeros_like(cross)
colors = np.empty(cross.shape, dtype=object)
for i in range(len(segments)):
    voxel[segments[i]] = i + 1
    colors[segments[i]] = [
        "red",
        "green",
        "blue",
        "yellow",
        "pink",
        "black",
        "grey",
        "purple",
    ][i]

fig, axs = plt.subplots(1, 1, subplot_kw={"projection": "3d"})


axs.voxels(voxel, facecolors=colors)
plt.show()
