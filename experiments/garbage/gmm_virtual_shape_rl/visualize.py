import os
import pickle
import numpy as np
import pyvista as pv
from PIL import Image
from renesis.utils.plotter import Plotter
from sklearn.metrics import precision_score, recall_score, f1_score

PATH = "/home/iffi/data"

if __name__ == "__main__":
    plotter = Plotter()
    pv.global_theme.window_size = [1536, 768]
    for file in os.listdir(PATH):
        if file.startswith("generated_data"):
            print(f"Rendering {file}")
            with open(os.path.join(PATH, file), "rb") as handle:
                ref, vox, dim, gen_path, error_path = pickle.load(handle)
            # ref[ref == 0] = 2
            # ref[ref == 1] = 0
            # vox[vox == 0] = 2
            # vox[vox == 1] = 0

            # occurences = [np.sum(ref == mat) for mat in (0, 1, 2, 3)]
            # weights = np.array(
            #     [1 / occurrence if occurrence > 0 else 0 for occurrence in occurences]
            # )
            # scores = f1_score(
            #     ref.reshape(-1),
            #     vox.reshape(-1),
            #     labels=(0, 1, 2, 3),
            #     average=None,
            #     zero_division=0,
            # )
            # print(10 * np.average(scores, weights=weights))
            img = plotter.plot_voxel(vox, distance=3 * dim)
            Image.fromarray(img, mode="RGB").save(
                os.path.join(PATH, os.path.basename(gen_path))
            )
            img = plotter.plot_voxel_error(ref, vox, distance=3 * dim,)
            Image.fromarray(img, mode="RGB").save(
                os.path.join(PATH, os.path.basename(error_path))
            )
