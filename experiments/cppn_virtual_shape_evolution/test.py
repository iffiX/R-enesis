import time
import pyvista as pv
import numpy as np
from PIL import Image
from renesis.utils.plotter import Plotter
from experiments.cppn_virtual_shape_evolution.utils import generate_3d_shape

if __name__ == "__main__":
    plotter = Plotter(interactive=True)
    # shape = generate_3d_shape(10, 100)
    # origins, materials = plotter.voxel_array_to_origins_and_materials(shape)
    # plotter.plot_voxels(
    #     origins,
    #     materials,
    #     distance=20,
    #     # palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    # )

    shape = generate_3d_shape(10, 100)
    approximate_shape = np.copy(shape)
    random = np.random.rand(*approximate_shape.shape)
    wrong_change = np.logical_and(random < 0.5, shape > 1)
    approximate_shape[wrong_change] = 1
    missing_change = np.logical_and(
        np.logical_and(random > 0.5, random < 0.7), shape > 1
    )
    approximate_shape[missing_change] = 0
    excessive_change = np.logical_and(random < 0.02, shape == 0)
    approximate_shape[excessive_change] = 2

    pv.global_theme.window_size = [1536, 768]
    plotter.plot_voxel_error(
        shape,
        approximate_shape,
        distance=20,
        # palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    )

    # plotter = Plotter()
    # for i in range(6):
    #     begin = time.time()
    #     shape = generate_3d_shape(10, 100)
    #     end = time.time()
    #     print(f"Generation {end - begin:3f} sec")
    #
    #     begin = time.time()
    #     origins, materials = plotter.voxel_array_to_origins_and_materials(shape)
    #     img = plotter.plot_voxels(
    #         origins,
    #         materials,
    #         distance=20,
    #         # palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    #     )
    #     end = time.time()
    #     print(f"Render {end - begin:3f} sec")
    #
    #     p_img = Image.fromarray(img, mode="RGB")
    #     p_img.save(f"{i}.png")
    #     print(i)
