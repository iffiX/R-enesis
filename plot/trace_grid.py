import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from moviepy.editor import VideoFileClip
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

video_paths = [
    "data/visualize_video_data/non_com/2500_0.mkv",
    "data/visualize_video_data/non_com/2500_6.mkv",
    "data/visualize_video_data/non_com/2500_7.mkv",
    "data/visualize_video_data/non_com/2500_15.mkv",
    "data/visualize_video_data/com/1000_0.mkv",
    "data/visualize_video_data/com/1000_1.mkv",
    "data/visualize_video_data/com/1000_3.mkv",
    "data/visualize_video_data/com/1000_4.mkv",
]

history_record_paths = [
    "data/visualize_video_data/non_com/tmp_it_2500_robot_0_rew_15.227669403096456.history",
    "data/visualize_video_data/non_com/tmp_it_2500_robot_6_rew_11.279425758876204.history",
    "data/visualize_video_data/non_com/tmp_it_2500_robot_7_rew_10.550575196642125.history",
    "data/visualize_video_data/non_com/tmp_it_2500_robot_15_rew_8.185663015907753.history",
    "data/visualize_video_data/com/tmp_it_1000_robot_0_rew_14.033060232132664.history",
    "data/visualize_video_data/com/tmp_it_1000_robot_1_rew_13.633829788544421.history",
    "data/visualize_video_data/com/tmp_it_1000_robot_3_rew_12.681339058219965.history",
    "data/visualize_video_data/com/tmp_it_1000_robot_4_rew_12.097341734228257.history",
]


def read_frames(history_record_path):
    with open(history_record_path, "r") as file:
        rescale = float(
            file.readline().strip().strip("{{{setting}}}<rescale>").strip("</rescale>")
        )
        frames = []
        for line in file.readlines():
            if line.startswith("<<<Step"):
                line = line.strip().strip("<<<>>>").split(">>>")[1]
                voxels = line.split(";")[:-1]
                frame = []
                for voxel in voxels:
                    values = [float(v) for v in voxel.split(",")[:-1]]
                    frame.append([values[i] * rescale for i in range(3)])
                frames.append(frame)
    # shape [frame num, 2]
    frames = np.mean(np.array(frames), axis=1)[:, :2]

    # use start as origin
    frames = frames - frames[0:1]

    # rotate clock wise so the vector from start to end aligns with +X
    sin_theta = frames[-1, 1] / np.sqrt(np.sum(frames[-1] ** 2))
    cos_theta = frames[-1, 0] / np.sqrt(np.sum(frames[-1] ** 2))
    return np.matmul(
        np.array(
            [
                [cos_theta, sin_theta],
                [-sin_theta, cos_theta],
            ]
        ),
        frames.transpose(1, 0),
    ).transpose(1, 0)


def get_robot_snapshot(video_path):
    video = VideoFileClip(video_path)
    return video.get_frame(0)


def get_line_collection(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="jet")
    lc.set_array(np.linspace(0, 1, len(x)))
    lc.set_linewidth(5)
    return lc


fig, axes = plt.subplots(4, 2, figsize=(4.4, 8))

# Loop through the video paths and extract and display the first frame for each
for i, (video_path, history_record_path) in enumerate(
    zip(video_paths, history_record_paths)
):
    ax = axes[i % 4][0 if i < 4 else 1]
    ax.imshow(get_robot_snapshot(video_path))
    ax.text(
        0.05,
        0.95,
        chr(int(i) + ord("A")),
        transform=ax.transAxes,
        color="black",
        fontsize=15,
        verticalalignment="top",
    )
    if i == 3:
        ax.set_title("Max Voxel\nDisplacement", y=-0.35, fontsize=15)
    elif i == 7:
        ax.set_title("Center of Mass\nDisplacement", y=-0.35, fontsize=15)
    ax.axis("off")

    axins = inset_axes(
        ax,
        width="50%",
        height="30%",
        loc="lower right",
        bbox_to_anchor=(0.1, 0, 0.9, 0.8),
        bbox_transform=ax.transAxes,
    )
    frames = read_frames(history_record_path)
    rainbow_cmap = plt.get_cmap("rainbow")
    # axins.plot(frames[:, 0], frames[:, 1], linestyle="-")
    axins.add_collection(get_line_collection(frames[:, 0], frames[:, 1]))
    axins.set_xlim(0, 0.1)
    axins.set_ylim(-0.05, 0.05)
    axins.axis("off")

rect_left = plt.Rectangle(
    # (lower-left corner), width, height
    (0.05, 0.0),
    0.44,
    0.99,
    fill=False,
    color="k",
    lw=2,
    zorder=1000,
    transform=fig.transFigure,
    figure=fig,
)
rect_right = plt.Rectangle(
    # (lower-left corner), width, height
    (0.51, 0.0),
    0.44,
    0.99,
    fill=False,
    color="k",
    lw=2,
    zorder=1000,
    transform=fig.transFigure,
    figure=fig,
)

fig.patches.extend([rect_left, rect_right])

# Adjust spacing and display the grid
plt.tight_layout()
plt.savefig(
    "data/generated_data/trace.pdf",
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()
