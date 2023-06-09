# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
#
# np.set_printoptions(precision=4, threshold=10000, suppress=True)
#
#
# def get_colormap():
#     N = 256
#     vals = np.ones((N, 4))
#     vals[:, 0] = np.linspace(1, 0 / 256, N)
#     vals[:, 1] = np.linspace(1, 91 / 256, N)
#     vals[:, 2] = np.linspace(1, 150 / 256, N)
#     return ListedColormap(vals)
#
#
# if __name__ == "__main__":
#     path = "/home/mlw0504/Projects/R-enesis_results/task_voxcraft/T=40_V=10x10x10_b=3x3x3-sphere_elastic_mod=1e6_std_bias=2voxels_freq=5/CustomPPO_VoxcraftSingleRewardPatchSphereEnvironment_20366_00000_0_2023-05-14_13-49-58/data"
#     max_epochs_to_show = 3000
#     bins_per_epoch = 32
#
#     data_files = {}
#     for file in os.listdir(path):
#         if file.startswith("data"):
#             data_files[int(file.split("_")[2])] = file
#
#     epochs = np.array(sorted([e for e in data_files.keys() if e <= max_epochs_to_show]))
#     data_of_all_epochs = []
#     for epoch in epochs:
#         with open(os.path.join(path, data_files[epoch]), "rb") as file:
#             data_of_all_epochs.append(pickle.load(file))
#
#     # shape: [epoch, batch_size]
#     rewards = np.array(
#         [
#             [data[idx]["reward"] for idx in range(len(data))]
#             for data in data_of_all_epochs
#         ]
#     )
#     robot_voxels = [
#         [data[idx]["voxels"] for idx in range(len(data))] for data in data_of_all_epochs
#     ]
#     robots = [
#         [data[idx]["robot"] for idx in range(len(data))] for data in data_of_all_epochs
#     ]
#
#     # Plot component figure 1
#     # draw the min & max reward curve
#     # plot robots as heatmap in between the curve
#     min_rewards = np.min(rewards, axis=1)
#     max_rewards = np.max(rewards, axis=1)
#     # plt.plot(epochs, min_rewards, color="skyblue")
#
#     hmap_x, hmap_y = np.meshgrid(
#         np.linspace(0, np.max(epochs), np.max(epochs) + 1),
#         np.linspace(0, np.max(rewards), bins_per_epoch + 1),
#     )
#     heatmap_bins = np.linspace(0, np.max(rewards), bins_per_epoch + 1)
#     heatmap_bins[-1] += 1e-3
#     heatmap_values = (
#         np.array(
#             [
#                 np.histogram(epoch_rewards, bins=heatmap_bins)[0]
#                 for epoch_rewards in rewards
#             ]
#         )
#         / 128
#     )
#     im = plt.pcolormesh(
#         hmap_x,
#         hmap_y,
#         heatmap_values.transpose(1, 0),
#         cmap=get_colormap(),
#         vmin=1e-3,
#         vmax=np.max(heatmap_values),
#     )
#     plt.colorbar(im, label="Fraction of robots")
#     plt.xlabel("Epochs")
#     # plt.ylabel("Volume")
#     plt.ylabel("Travel distance in voxels")
#     plt.xlim(-30, np.max(epochs) + 30)
#     plt.ylim(0, 20)
#     plt.axvline(x=0, color="lightcoral", linestyle="dashed")
#     plt.axvline(x=np.argmax(max_rewards), color="lightgreen", linestyle="dashed")
#     plt.axvline(x=np.max(epochs), color="lightblue", linestyle="dashed")
#     cummax_rewards = np.maximum.accumulate(max_rewards)
#     plt.plot(
#         epochs,
#         cummax_rewards,
#         color="grey",
#     )
