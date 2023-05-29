import os
import matplotlib.pyplot as plt

image_dir = "/home/mlw0504/Projects/R-enesis/experiments/visualize_data/images"
if __name__ == "__main__":
    images = []
    for file in os.listdir(image_dir):
        images.append(
            (
                float(file.split("_")[6].strip(".png")),
                plt.imread(os.path.join(image_dir, file)),
            )
        )
    images = sorted(images, key=lambda x: x[0])
    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(images):
                reward_value, image = images[index]
                axes[i, j].imshow(image)
                axes[i, j].text(
                    0.05,
                    0.05,
                    f"Reward: {reward_value:.2f}",
                    transform=axes[i, j].transAxes,
                    color="black",
                    fontsize=10,
                    verticalalignment="bottom",
                )
                axes[i, j].text(
                    0.05,
                    0.95,
                    chr(index + ord("A")),
                    transform=axes[i, j].transAxes,
                    color="black",
                    fontsize=10,
                    verticalalignment="top",
                )
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()
