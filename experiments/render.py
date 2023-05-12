import os
from renesis.utils.media import create_video_subproc
from renesis.sim import Voxcraft, VXHistoryRenderer


def render(history):
    try:
        renderer = VXHistoryRenderer(history=history, width=640, height=480)
        renderer.render()
        frames = renderer.get_frames()
        if frames.ndim == 4:
            print("History saved")
            return frames
        else:
            print("Rendering finished, but no frames produced")
            print("History:")
            print(history)
            return None
    except Exception as e:
        print(e)
        print("Exception occurred, no frames produced")
        print("History:")
        print(history)
        return None


if __name__ == "__main__":
    sim_history_file_name = "/home/mlw0504/ray_results/CustomPPO_2023-05-11_15-25-59/CustomPPO_VoxcraftSingleRewardPatchEnvironment_0a923_00000_0_2023-05-11_15-25-59/data/run_it_506_rew_19.308299979542475.history"

    with open(
        sim_history_file_name,
        "r",
    ) as file:
        sim_history = file.read()
    frames = render(sim_history)
    if frames is not None:
        path = os.path.join(
            os.path.dirname(sim_history_file_name),
            f"rendered.gif",
        )
        print(f"Saving rendered results to {path}")
        wait = create_video_subproc(
            [f for f in frames],
            path=os.path.dirname(path),
            filename=f"rendered",
            extension=".gif",
        )
        wait()
