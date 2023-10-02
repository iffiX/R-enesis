from moviepy.editor import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    clips_array,
)
import moviepy.video.fx.all as vfx

bad_video_paths = [
    (
        "data/visualize_video_data/non_com/2500_125_bad.mkv",
        ("00:00:01.07", "00:00:15.07"),
    ),
    (
        "data/visualize_video_data/com/1000_bad.mkv",
        ("00:00:01.02", "00:00:14,16"),
    ),
]
good_video_paths = [
    ("data/visualize_video_data/non_com/2500_0.mkv", ("00:00:00.29", "00:00:14.11")),
    ("data/visualize_video_data/non_com/2500_6.mkv", ("00:00:01.09", "00:00:14.19")),
    ("data/visualize_video_data/non_com/2500_7.mkv", ("00:00:01.07", "00:00:15.20")),
    ("data/visualize_video_data/non_com/2500_15.mkv", ("00:00:00.28", "00:00:14.07")),
    ("data/visualize_video_data/com/1000_0.mkv", ("00:00:00.00", "00:00:12.11")),
    ("data/visualize_video_data/com/1000_1.mkv", ("00:00:00.29", "00:00:14.09")),
    ("data/visualize_video_data/com/1000_3.mkv", ("00:00:00.26", "00:00:13.27")),
    ("data/visualize_video_data/com/1000_4.mkv", ("00:00:01.00", "00:00:14.09")),
]

# width, height
clip_size = (512, 512)
margin_size = 10

bad_video_clips = [
    VideoFileClip(video_path[0])
    .without_audio()
    .resize(clip_size)
    .margin(margin_size)
    .subclip(*video_path[1])
    .fx(vfx.loop, n=3)
    for video_path in bad_video_paths
]
good_video_clips = [
    VideoFileClip(video_path[0])
    .without_audio()
    .resize(clip_size)
    .margin(margin_size)
    .subclip(*video_path[1])
    .fx(vfx.loop, n=3)
    for video_path in good_video_paths
]

bad_video_labels = [
    TextClip(
        chr(i + ord("A")),
        color="red",
        fontsize=50,
        bg_color="black",
    )
    for i in [0, 5]
]

good_video_labels = [
    TextClip(
        chr(i + ord("A")),
        color="white",
        fontsize=50,
        bg_color="black",
    )
    for i in [1, 2, 3, 4, 6, 7, 8, 9]
]

# Create a list of clips for the first column (2 videos), bad examples
column1_clips = clips_array(
    [
        [CompositeVideoClip([bad_video_clips[0], bad_video_labels[0]])],
        [CompositeVideoClip([bad_video_clips[1], bad_video_labels[1]])],
    ]
)


# Create a list of clips for the second column (8 videos), good examples
column2_clips = clips_array(
    [
        [
            CompositeVideoClip([good_video_clips[i], good_video_labels[i]])
            for i in range(4)
        ],
        [
            CompositeVideoClip([good_video_clips[i], good_video_labels[i]])
            for i in range(4, 8)
        ],
    ]
)

# # Combine the clips into a grid with two columns
grid = clips_array([[column1_clips, column2_clips]])
output_path = "output_grid_with_labels.mp4"
grid = grid.set_duration(30)
grid.write_videofile(output_path, codec="libx264", fps=24)
