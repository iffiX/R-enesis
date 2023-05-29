import cc3d
import numpy as np
from scipy.spatial import ConvexHull
from renesis.utils.plotter import Plotter


def max_z(end_positions):
    _, max_z = get_min_max_z(end_positions)
    return max_z


def table(start_positions):
    _, max_z = get_min_max_z(start_positions)
    num_at_z, num_not_at_z = get_num_at_z(start_positions, max_z)
    # Account for 0 being height with one voxel.
    height = max_z + 1
    surface = num_at_z
    excess = num_not_at_z or 1
    stability = get_stability(start_positions, max_z) or 1
    return height * surface * (stability / excess)


def get_min_max_z(end_positions):
    max_z = -np.inf
    min_z = np.inf
    for p in end_positions:
        if p[2] > max_z:
            max_z = p[2]
        if p[2] < min_z:
            min_z = p[2]
    return min_z, max_z


def get_num_at_z(x, z):
    at_z = 0
    not_at_z = 0
    for p in x:
        if np.floor(p[2]) == np.floor(z):
            at_z += 1
        else:
            not_at_z += 1
    return at_z, not_at_z


def prepare_points_for_convex_hull(x):
    q = set()
    for p in x:
        q.add(p)
        q.add((p[0] + 1, p[1]))
        q.add((p[0], p[1] + 1))
        q.add((p[0] + 1, p[1] + 1))
    x = list(q)
    return x


def get_bounding_box_sizes(X):
    occupied = X != 0
    x_occupied = [x for x in range(occupied.shape[0]) if np.any(occupied[x])]
    y_occupied = [y for y in range(occupied.shape[1]) if np.any(occupied[:, y])]
    z_occupied = [z for z in range(occupied.shape[2]) if np.any(occupied[:, :, z])]
    min_x = min(x_occupied)
    max_x = max(x_occupied) + 1
    min_y = min(y_occupied)
    max_y = max(y_occupied) + 1
    min_z = min(z_occupied)
    max_z = max(z_occupied) + 1
    return max_x - min_x, max_y - min_y, max_z - min_z


def get_convex_hull_area(x):
    x = prepare_points_for_convex_hull(x)
    if len(x) == 0:
        return 0
    if len(x) == 1:
        return 0
    if len(x) == 2:
        return np.sqrt((x[0][0] - x[0][1]) ** 2 + (x[1][0] - x[1][1]) ** 2)

    return ConvexHull(x).volume


def get_convex_hull_volume(x):
    if len(x) == 0:
        return 0
    x, _, _ = Plotter.get_vertices_of_voxel(x)
    return ConvexHull(x).volume


def max_hull_volume_min_density(x):
    if len(x) == 0:
        return 0
    return get_convex_hull_volume(x) / len(x)


def get_stability(x, max_z):
    descending_positions = sorted(x, key=lambda p: p[2], reverse=True)
    # Top layer is not considered.
    while np.floor(descending_positions[0][2]) == np.floor(max_z):
        descending_positions.pop(0)
        if len(descending_positions) == 0:
            return 0

    stability = 0
    # Calculate the convex hull for each layer.
    z = descending_positions[0][2]
    current_layer = []
    for p in descending_positions:
        if np.floor(p[2]) == np.floor(z):
            current_layer.append((p[0], p[1]))
        else:
            stability += get_convex_hull_area(current_layer)
            current_layer = []
            current_layer.append((p[0], p[1]))
            z = p[2]
    stability += get_convex_hull_area(current_layer)

    return stability


def has_fallen(start_positions, end_positions, threshold=0.25):
    # ! Incorrect
    Z_initial = np.array(start_positions)[:, :2]
    Z_final = np.array(end_positions)[:, :2]
    difference = np.abs(Z_final - Z_initial)
    return np.any(difference >= threshold)


def distance_traveled(start_com, end_com):
    return np.linalg.norm(end_com - start_com).mean()


def max_volume(X):
    s = get_surface_area(X)
    v = get_volume(X)
    if v == 0 or s == 0:
        return 0
    return v / s


def max_surface_area(X):
    s = get_surface_area(X)
    v = get_volume(X)
    if v == 0 or s == 0:
        return 0
    return s / v


def get_volume(X):
    return np.sum(X != 0)


def pad_voxels(X):
    new_voxels = np.zeros(
        (X.shape[0] + 2, X.shape[1] + 2, X.shape[2] + 2), dtype=X.dtype
    )
    new_voxels[1:-1, 1:-1, 1:-1] = X
    return new_voxels


def get_surface_voxels(X):
    if X.shape[0] > 2 and X.shape[1] > 2 and X.shape[2] > 2:
        padded_X = pad_voxels(X)
        coords = np.stack(
            np.meshgrid(
                list(range(1, padded_X.shape[0] - 1)),
                list(range(1, padded_X.shape[1] - 1)),
                list(range(1, padded_X.shape[2] - 1)),
                indexing="ij",
            )
        ).reshape(3, 1, -1)
        offsets = (
            np.array(
                [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
            )
            .transpose(1, 0)
            .reshape(3, -1, 1)
        )
        # shape [3, 6, X_space_size]
        neighbor_coords = coords + offsets
        # Note: includes voxels of a internal hole surface
        # shape [6, X_space_size]
        neighbor_is_empty = np.any(
            padded_X[neighbor_coords[0], neighbor_coords[1], neighbor_coords[2]] == 0,
            axis=0,
        )
        # shape [X_space_size]
        self_is_not_empty = padded_X[coords[0, 0], coords[1, 0], coords[2, 0]] != 0
        return np.sum(np.logical_and(self_is_not_empty, neighbor_is_empty))
    else:
        return get_volume(X)


def get_surface_area(X):
    padded_X = pad_voxels(X)

    coords = np.stack(
        np.meshgrid(
            list(range(1, padded_X.shape[0] - 1)),
            list(range(1, padded_X.shape[1] - 1)),
            list(range(1, padded_X.shape[2] - 1)),
            indexing="ij",
        )
    ).reshape(3, 1, -1)
    offsets = (
        np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
        .transpose(1, 0)
        .reshape(3, -1, 1)
    )
    # shape [3, 6, X_space_size]
    neighbor_coords = coords + offsets
    # shape [6, X_space_size]
    neighbor_is_empty = (
        padded_X[neighbor_coords[0], neighbor_coords[1], neighbor_coords[2]] == 0,
    )
    # shape [1, X_space_size]
    self_is_not_empty = padded_X[coords[0], coords[1], coords[2]] != 0
    surfaces = np.sum(np.logical_and(self_is_not_empty, neighbor_is_empty))
    return surfaces


def get_section_num(X):
    X = np.copy(X)
    X[X == 0] = X.max() + 1
    _labels, label_num = cc3d.connected_components(
        X, connectivity=6, return_N=True, out_dtype=np.uint64
    )
    return label_num


def get_reflection_symmetry(X):
    # Note: this function is based on the assumption that
    # voxels X are symmetric with respect to the mid
    # cross section plane in each direction. Mid plane is
    # defined as the plane overlapping the mid point of
    # the representation space in each direction.

    # Eg: if we use GMM model, since the action space is
    # symmetric with respect to the mid cross section plane
    # we can use this metric to get the reflection symmetry.

    # just check non-zero voxel difference ratio in each direction
    # harmonic_mean(symmetric_voxels / left_voxels, symmetric_voxels / right_voxels)
    symmetric_scores = []
    for axis in range(3):
        axis_length = X.shape[axis]
        left_index = tuple(
            slice(0, X.shape[ax]) if ax != axis else slice(0, axis_length // 2)
            for ax in range(3)
        )
        left_section = X[left_index]
        right_index = tuple(
            slice(0, X.shape[ax])
            if ax != axis
            else slice(axis_length - axis_length // 2, X.shape[ax])
            for ax in range(3)
        )
        right_section = X[right_index]
        flipped_right_section = np.flip(right_section, axis=axis)
        left_voxels = np.sum(left_section != 0)
        right_voxels = np.sum(right_section != 0)
        symmetric_voxels = np.sum(
            np.logical_and(left_section == flipped_right_section, left_section != 0)
        )
        symmetric_scores.append(
            2 / (left_voxels / symmetric_voxels + right_voxels / symmetric_voxels)
        )
    return np.mean(symmetric_scores)
