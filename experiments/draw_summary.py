import os
import matplotlib.pyplot as plt

image_dir = "/home/mlw0504/Projects/R-enesis/experiments/summary_data"
if __name__ == "__main__":
    images = []
    for file in os.listdir(image_dir):
        images.append(
            (
                float(file.strip(".png")),
                plt.imread(os.path.join(image_dir, file)),
            )
        )
    images = sorted(images, key=lambda x: x[0])
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(25, 10))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(images):
                reward_value, image = images[index]
                axes[i, j].imshow(image)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("generated_data/summary.pdf", bbox_inches="tight", pad_inches=0.5)
    plt.show()
