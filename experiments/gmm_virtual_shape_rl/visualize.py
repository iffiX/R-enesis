import os
import pickle
from PIL import Image
from renesis.utils.plotter import Plotter

PATH = "/home/iffi/data"

if __name__ == "__main__":
    plotter = Plotter()
    for file in os.listdir(PATH):
        if file.startswith("generated_data"):
            print(f"Rendering {file}")
            with open(os.path.join(PATH, file), "rb") as handle:
                ref, vox, dim, gen_path, error_path = pickle.load(handle)
            img = plotter.plot_voxel(vox, distance=3 * dim)
            Image.fromarray(img, mode="RGB").save(
                os.path.join(PATH, os.path.basename(gen_path))
            )
            img = plotter.plot_voxel_error(ref, vox, distance=3 * dim,)
            Image.fromarray(img, mode="RGB").save(
                os.path.join(PATH, os.path.basename(error_path))
            )
