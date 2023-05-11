import os
import pickle
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from renesis.utils.plotter import Plotter

pv.global_theme.full_screen = True
np.set_printoptions(precision=4, threshold=10000, suppress=True)

if __name__ == "__main__":
    path = "/home/mlw0504/ray_results/CustomPPO_2023-05-10_18-03-52/CustomPPO_VoxcraftSingleRewardTestPatchEnvironment_eeca8_00000_0_2023-05-10_18-03-53/data"
    # path = input("Enter path to ray experiment data directory: ")
    data_files = {}
    for file in os.listdir(path):
        if file.startswith("data"):
            data_files[int(file.split("_")[2])] = file
    while len(data_files) > 0:
        print(f"Available iterations: {sorted(list(data_files.keys()))}")
        iter_num = int(input("Enter iteration number: "))
        if iter_num not in data_files:
            print(f"Invalid iteration number: {iter_num}")
            continue
        with open(os.path.join(path, data_files[iter_num]), "rb") as file:
            data = pickle.load(file)
            data = sorted(data, key=lambda d: d["reward"], reverse=True)[:8]
            row_size = int(np.ceil(np.sqrt(len(data) / 2)))
            col_size = row_size * 2
            # pv_plotter = pv.Plotter(shape=(row_size, col_size))
            # voxel_plotter = Plotter(interactive=True)
            fig, axs = plt.subplots(row_size, col_size, subplot_kw={"projection": "3d"})
            print([data[idx]["reward"] for idx in range(len(data))])
            for row in range(row_size):
                for col in range(col_size):
                    idx = row * col_size + col

                    # pv_plotter.subplot(row, col)
                    # if idx < len(data):
                    #     if idx < 3:
                    #         print(np.stack(data[idx]["step_dists"]))
                    #     if data[idx]["reward"] > 0:
                    #         pv_plotter.set_background(
                    #             color=[0.8, 0.8, 0.8], all_renderers=False
                    #         )
                    #     pv_plotter.add_text(
                    #         str(data[idx]["reward"]), color="red", font_size=20
                    #     )
                    #     voxel_plotter.plot_voxel(
                    #         data[idx]["voxels"].astype(np.float32),
                    #         # palette=["blue", "green", "red"],
                    #         plotter=pv_plotter,
                    #         distance=30,
                    #     )

                    if idx < len(data):
                        if idx < 3:
                            print(np.stack(data[idx]["step_dists"]))
                            print(np.stack(data[idx]["patches"]))
                        if data[idx]["reward"] <= 0:
                            axs[row][col].set_facecolor([0.2, 0.2, 0.2])
                        axs[row][col].set_title(str(data[idx]["reward"]))
                        axs[row][col].set_xticks([])
                        axs[row][col].set_yticks([])
                        axs[row][col].set_zticks([])
                        filled = data[idx]["voxels"] != 0
                        colors = np.empty(data[idx]["voxels"].shape, dtype=object)
                        colors[data[idx]["voxels"] == 1] = "blue"
                        colors[data[idx]["voxels"] == 2] = "green"
                        colors[data[idx]["voxels"] == 3] = "red"
                        axs[row][col].voxels(filled, facecolors=colors)

            # pv_plotter.show()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            fig.show()
