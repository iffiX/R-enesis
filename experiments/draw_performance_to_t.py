import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

files = [
    "/home/mlw0504/Projects/R-enesis/experiments/generated_data/robot_voxcraft_performance_relative_to_t_0.data",
    "/home/mlw0504/Projects/R-enesis/experiments/generated_data/robot_voxcraft_performance_relative_to_t_1.data",
    "/home/mlw0504/Projects/R-enesis/experiments/generated_data/robot_voxcraft_performance_relative_to_t_2.data",
    "/home/mlw0504/Projects/R-enesis/experiments/generated_data/robot_voxcraft_performance_relative_to_t_3.data",
    "/home/mlw0504/Projects/R-enesis/experiments/generated_data/robot_voxcraft_performance_relative_to_t_4.data",
]

data = []
for idx in range(len(files)):
    with open(files[idx], "rb") as file:
        data.append(pickle.load(file))
data = np.array(data)

# mean = np.mean(data, axis=2)
# std = np.std(data, axis=2)
# action_counts = [idx for idx in range(100)]
# for i in range(len(files)):
#     confidence_interval = std[i] * 2.576 / np.sqrt(128)
#     plt.plot(action_counts, mean[i], color=f"C{i}", label=f"trial: {i}")
#     plt.fill_between(
#         action_counts,
#         mean[i] - confidence_interval,
#         mean[i] + confidence_interval,
#         color=f"C{i}",
#         alpha=0.5,
#     )
#     plt.xlabel("Step count")
#     plt.ylabel("Average reward")
# plt.legend()
# plt.show()

mean = np.mean(data, axis=(0, 2))
std = np.std(data, axis=(0, 2))
action_counts = [idx for idx in range(100)]
confidence_interval = std * 2.576 / np.sqrt(data.shape[0] * data.shape[2])
plt.figure(figsize=(3, 3))
plt.plot(action_counts, mean, color=f"grey")
plt.fill_between(
    action_counts,
    mean - confidence_interval,
    mean + confidence_interval,
    color=f"lightgrey",
)
plt.xlabel("Action count", fontsize=14)
plt.ylabel("Displacement\n(voxel length)", fontsize=14)
plt.savefig(
    f"generated_data/performance_to_t.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()
