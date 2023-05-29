import cc3d
import numpy as np


def get_robot_voxels_from_voxels(voxels: np.ndarray, occupied: np.ndarray = None):
    occupied = occupied if occupied is not None else voxels != 0
    labels, label_num = cc3d.connected_components(
        occupied, connectivity=6, return_N=True, out_dtype=np.uint32
    )
    count = np.bincount(labels.reshape(-1), minlength=label_num)
    # Ignore label 0, which is non-occupied space
    count[0] = 0
    largest_connected_component = labels == np.argmax(count)
    robot_voxels = np.where(largest_connected_component, voxels, 0)
    robot_occupied = robot_voxels != 0
    return robot_voxels, robot_occupied


def get_representation_from_robot_voxels(
    robot_voxels: np.ndarray, robot_occupied: np.ndarray = None
):
    robot_occupied = robot_occupied if robot_occupied is not None else robot_voxels != 0
    x_occupied = [
        x for x in range(robot_occupied.shape[0]) if np.any(robot_occupied[x])
    ]
    y_occupied = [
        y for y in range(robot_occupied.shape[1]) if np.any(robot_occupied[:, y])
    ]
    z_occupied = [
        z for z in range(robot_occupied.shape[2]) if np.any(robot_occupied[:, :, z])
    ]
    min_x = min(x_occupied)
    max_x = max(x_occupied) + 1
    min_y = min(y_occupied)
    max_y = max(y_occupied) + 1
    min_z = min(z_occupied)
    max_z = max(z_occupied) + 1
    representation = []

    for z in range(min_z, max_z):
        layer_representation = (
            robot_voxels[min_x:max_x, min_y:max_y, z]
            .astype(int)
            .flatten(order="F")
            .tolist(),
            None,
            None,
            None,
        )
        representation.append(layer_representation)
    return (max_x - min_x, max_y - min_y, max_z - min_z), representation
