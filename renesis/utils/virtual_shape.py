import numpy as np
from renesis.env_model.growth import GrowthModel


def scale(ratio, min, max):
    return min + (max - min) * ratio


def generate_3d_shape(
    dimension_size,
    steps,
    same_threshold=2,
    change_material_when_same_major_prob=0,
    change_material_when_same_minor_prob=0.2,
    fill_num=(1, 2, 3),
    material_num=3,
    seed=42,
):
    model = GrowthModel(
        materials=list(range(material_num + 1)),
        max_dimension_size=dimension_size,
        max_view_size=3,
        actuation_features=(),
    )
    rand = np.random.RandomState(seed)
    for i in range(steps):
        view = np.round(model.observe() * material_num).astype(int).squeeze(axis=3)
        center = view[1, 1, 1]
        adjacent = np.array(
            (
                view[0, 1, 1],
                view[2, 1, 1],
                view[1, 0, 1],
                view[1, 2, 1],
                view[1, 1, 0],
                view[1, 1, 2],
            )
        )
        same = np.sum(adjacent == center)
        if (
            i == 0
            or (
                same <= same_threshold
                and rand.rand() < change_material_when_same_major_prob
            )
            or rand.rand() < change_material_when_same_minor_prob
        ):
            new = rand.choice([i for i in range(1, material_num + 1) if i != center])
        else:
            new = int(center)
        configuration = np.zeros([6, material_num + 1, 1])
        # default is null material
        configuration[:, 0, :] = 1
        avail_positions = model.get_valid_position_indices()
        rand.shuffle(avail_positions)
        new_positions = avail_positions[: rand.choice(fill_num)]
        for pos in new_positions:
            configuration[pos, 0, :] = 0
            configuration[pos, new, :] = 1
        model.step(configuration)
        if model.is_finished():
            break
    st = model.radius
    end = model.radius + model.max_dimension_size
    return model.voxels[st:end, st:end, st:end].squeeze(-1)


def generate_sphere(dimension_size, radius_ratio=0.25, material=1):
    voxels = np.zeros([dimension_size, dimension_size, dimension_size], dtype=np.int)
    center_voxel_offset = dimension_size // 2
    indices = list(range(-center_voxel_offset, dimension_size - center_voxel_offset,))
    coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
    # coords shape [coord_num, 3]
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    distance = np.linalg.norm(coords, axis=1)
    all_material = np.where(distance < dimension_size * radius_ratio, material, 0)
    voxels[
        coords[:, 0] + center_voxel_offset,
        coords[:, 1] + center_voxel_offset,
        coords[:, 2] + center_voxel_offset,
    ] = all_material
    return voxels


def generate_random_ellipsoids(dimension_size, num=5, materials=(1, 2, 3), seed=42):
    voxels = np.zeros([dimension_size, dimension_size, dimension_size], dtype=np.int)
    center_voxel_offset = dimension_size // 2
    indices = list(range(-center_voxel_offset, dimension_size - center_voxel_offset,))
    coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
    # coords shape [coord_num, 3]
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    rand = np.random.RandomState(seed)
    ellipsoids = [
        (
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            rand.choice(materials),
        )
        for _ in range(num)
    ]
    all_material = np.zeros((dimension_size ** 3,), dtype=np.int)
    for ellipsoid in ellipsoids:
        x_diff = coords[:, 0] - ellipsoid[0]
        y_diff = coords[:, 1] - ellipsoid[1]
        z_diff = coords[:, 2] - ellipsoid[2]
        all_material = np.where(
            x_diff ** 2 / ellipsoid[3] ** 2
            + y_diff ** 2 / ellipsoid[4] ** 2
            + z_diff ** 2 / ellipsoid[5] ** 2
            <= 1,
            ellipsoid[6],
            all_material,
        )
    voxels[
        coords[:, 0] + center_voxel_offset,
        coords[:, 1] + center_voxel_offset,
        coords[:, 2] + center_voxel_offset,
    ] = all_material
    return voxels


def generate_inversed_random_ellipsoids(dimension_size, num=5, material=1, seed=42):
    voxels = np.full(
        [dimension_size, dimension_size, dimension_size], material, dtype=np.int
    )
    center_voxel_offset = dimension_size // 2
    indices = list(range(-center_voxel_offset, dimension_size - center_voxel_offset,))
    coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
    # coords shape [coord_num, 3]
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    rand = np.random.RandomState(seed)
    ellipsoids = [
        (
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), -dimension_size / 3, dimension_size / 3),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
            scale(rand.rand(), min(1, dimension_size / 10), dimension_size / 6),
        )
        for _ in range(num)
    ]
    all_material = np.full((dimension_size ** 3,), material, dtype=np.int)
    for ellipsoid in ellipsoids:
        x_diff = coords[:, 0] - ellipsoid[0]
        y_diff = coords[:, 1] - ellipsoid[1]
        z_diff = coords[:, 2] - ellipsoid[2]
        all_material = np.where(
            x_diff ** 2 / ellipsoid[3] ** 2
            + y_diff ** 2 / ellipsoid[4] ** 2
            + z_diff ** 2 / ellipsoid[5] ** 2
            <= 1,
            0,
            all_material,
        )
    voxels[
        coords[:, 0] + center_voxel_offset,
        coords[:, 1] + center_voxel_offset,
        coords[:, 2] + center_voxel_offset,
    ] = all_material
    return voxels


def generate_cross(dimension_size, length_ratio=0.5, thickness_ratio=0.1, material=1):
    voxels = np.zeros([dimension_size, dimension_size, dimension_size], dtype=np.int)
    center_voxel_offset = dimension_size // 2
    indices = list(range(-center_voxel_offset, dimension_size - center_voxel_offset,))
    coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
    # coords shape [coord_num, 3]
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    distance_1 = np.sqrt(coords[:, 1] ** 2 + ((coords[:, 0] - coords[:, 2]) ** 2) / 2)
    distance_2 = np.sqrt(coords[:, 1] ** 2 + ((coords[:, 0] + coords[:, 2]) ** 2) / 2)
    bar_1 = np.logical_and(
        distance_1 < min(dimension_size * thickness_ratio, 1),
        np.abs(coords[:, 0] + coords[:, 2]) < dimension_size * length_ratio,
    )
    bar_2 = np.logical_and(
        distance_2 < min(dimension_size * thickness_ratio, 1),
        np.abs(-coords[:, 0] + coords[:, 2]) < dimension_size * length_ratio,
    )
    bound = dimension_size * length_ratio * 0.5
    ends = (
        np.linalg.norm(coords - np.array([bound, 0, bound]), axis=1)
        < min(dimension_size * thickness_ratio, 1),
        np.linalg.norm(coords - np.array([bound, 0, -bound]), axis=1)
        < min(dimension_size * thickness_ratio, 1),
        np.linalg.norm(coords - np.array([-bound, 0, bound]), axis=1)
        < min(dimension_size * thickness_ratio, 1),
        np.linalg.norm(coords - np.array([-bound, 0, -bound]), axis=1)
        < min(dimension_size * thickness_ratio, 1),
    )
    all_material = np.where(
        np.any(np.stack((bar_1, bar_2) + ends, axis=0), axis=0), material, 0,
    )
    voxels[
        coords[:, 0] + center_voxel_offset,
        coords[:, 1] + center_voxel_offset,
        coords[:, 2] + center_voxel_offset,
    ] = all_material
    return voxels
