import cc3d
import numpy as np
from typing import List
from collections import OrderedDict
from gym.spaces import Box, Dict
from scipy.stats import multivariate_normal
from .base import BaseModel


def is_voxel_continuous(occupied: np.ndarray):
    _labels, label_num = cc3d.connected_components(
        occupied, connectivity=6, return_N=True, out_dtype=np.uint64
    )
    return label_num <= 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip(x):
    return (np.clip(x, -2, 2) + 2) / 4


normalize = clip


class GMMModel(BaseModel):
    def __init__(
        self,
        materials=(0, 1, 2),
        dimension_size=20,
        max_gaussian_num=100,
        cutoff=1e-2,
    ):
        """
        Tail bound from
        https://math.stackexchange.com/questions/4103823/
        tail-probabilities-of-multi-variate-normal
        """
        super().__init__()
        self.materials = materials
        self.dimension_size = dimension_size
        self.center_voxel_offset = self.dimension_size // 2
        self.max_gaussian_num = max_gaussian_num
        self.cutoff = cutoff

        # A list of arrays of shape [7],
        # first 3 elements are mean (x, y, z)
        # Next 3 elements are the diagonal of the covariance matrix
        # Last element is the material index
        self.gaussians = []  # type: List[np.ndarray]
        self.voxels = None
        self.occupied = None
        self.invalid_count = 0
        self.is_robot_valid = False
        self.update_voxels()

    @property
    def action_space(self):
        # shape [7]
        return Box(low=0, high=1, shape=(6 + len(self.materials),))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(6 + len(self.materials),))

    def reset(self):
        self.steps = 0
        self.gaussians = []
        self.update_voxels()

    def is_finished(self):
        # return self.invalid_count > 1
        # return self.steps != 0 and not self.is_robot_valid
        return False

    def is_robot_invalid(self):
        return not self.is_robot_valid

    def step(self, action: np.ndarray):
        # print(action)
        self.gaussians.append(action)
        self.update_voxels()
        self.steps += 1

    def observe(self):
        if self.gaussians:
            return self.gaussians[-1]
        else:
            return np.zeros((6 + len(self.materials),), dtype=np.float32)

    def get_robot(self):
        x_occupied = [
            x for x in range(self.occupied.shape[0]) if np.any(self.occupied[x])
        ]
        y_occupied = [
            y for y in range(self.occupied.shape[1]) if np.any(self.occupied[:, y])
        ]
        z_occupied = [
            z for z in range(self.occupied.shape[2]) if np.any(self.occupied[:, :, z])
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
                self.voxels[min_x:max_x, min_y:max_y, z]
                .astype(int)
                .flatten(order="F")
                .tolist(),
                None,
                None,
                None,
            )
            representation.append(layer_representation)
        return (max_x - min_x, max_y - min_y, max_z - min_z), representation

    def get_voxels(self):
        return (
            self.voxels
            if self.voxels is not None
            else np.zeros([self.dimension_size] * 3, dtype=np.float32)
        )

    def get_state_data(self):
        return np.stack(self.gaussians)

    def scale(self, action):
        # print(action)
        max_radius = self.dimension_size / 2
        return np.array(
            [-max_radius, -max_radius, -max_radius, 0.1, 0.1, 0.1]
            + [0] * len(self.materials)
        ) + action * np.array(
            [
                max_radius * 2,
                max_radius * 2,
                max_radius * 2,
                max_radius / 6 - 0.1,
                max_radius / 6 - 0.1,
                max_radius / 6 - 0.1,
            ]
            + [1] * len(self.materials)
        )

    def update_voxels(self):
        # generate coordinates
        # Eg: if dimension size is 20, indices are [-10, ..., 9]
        # if dimension size if 21, indices are [-10, ..., 10]
        indices = list(
            range(
                -self.center_voxel_offset,
                self.dimension_size - self.center_voxel_offset,
            )
        )
        coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
        # coords shape [coord_num, 3]
        coords = np.transpose(coords.reshape([coords.shape[0], -1]))
        all_values = []
        for gaussian in self.gaussians:
            gaussian = self.scale(gaussian)
            distribution = multivariate_normal(gaussian[:3], np.diag(gaussian[3:6]))
            values = distribution.pdf(coords)
            # t = -np.log(self.cutoff)
            # cutoff_distance = np.sqrt(
            #     (
            #         np.sum(gaussian[3:])
            #         + 2 * np.sqrt(t) * np.linalg.norm(gaussian[3:6], ord=None)
            #         + 2 * t * np.max(gaussian[3:6])
            #     )
            # )
            # print(cutoff_distance)
            # values = np.where(
            #     np.linalg.norm(coords - gaussian[:3], axis=1, ord=2) < cutoff_distance,
            #     values,
            #     0,
            # )
            values = np.where(values > np.max(values) * self.cutoff, values, 0)
            all_values.append(values)

        self.voxels = np.zeros(
            [self.dimension_size] * 3,
            dtype=float,
        )

        if self.gaussians:
            # all_values shape [coord_num, gaussian_num]
            all_values = np.stack(all_values, axis=1)
            material_map = np.array(
                [
                    self.materials[int(np.argmax(gaussian[6:]))]
                    for gaussian in self.gaussians
                ]
            )
            material = np.where(
                np.any(all_values > 0, axis=1),
                material_map[np.argmax(all_values, axis=1)],
                0,
            )

            self.voxels[
                coords[:, 0] + self.center_voxel_offset,
                coords[:, 1] + self.center_voxel_offset,
                coords[:, 2] + self.center_voxel_offset,
            ] = material
        # print("outputs:")
        # print(outputs)
        # print("voxels:")
        # print(self.voxels)
        self.occupied = self.voxels[:, :, :] != 0
        prev_is_valid = self.is_robot_valid
        self.is_robot_valid = is_voxel_continuous(self.occupied) and np.any(
            self.occupied
        )
        if self.steps != 0 and not prev_is_valid and not self.is_robot_valid:
            self.invalid_count += 1
        else:
            self.invalid_count = 0


class GMMObserveWithVoxelModel(GMMModel):
    def __init__(
        self,
        materials=(0, 1, 2),
        dimension_size=20,
        max_gaussian_num=100,
        cutoff=1e-2,
    ):
        super().__init__(materials, dimension_size, max_gaussian_num, cutoff)
        self.prev_voxels = np.zeros([self.dimension_size**3], dtype=np.float32)

    @property
    def observation_space(self):
        # Flatten observation space into Box space since ray trajectory view
        # doesn't support Dict space for the shift function.
        # See:
        # https://github.com/ray-project/ray/blob/
        # 5b99dd9b79d8e1d26b0254ab2f3d270331369a0e/rllib/policy/policy.py#L1513

        return Box(
            low=np.zeros(
                (1 + 6 + len(self.materials) + self.dimension_size**3,),
                dtype=np.float32,
            ),
            high=np.array(
                [self.max_gaussian_num]
                + [1] * (6 + len(self.materials))
                + [len(self.materials) - 1] * self.dimension_size**3,
                dtype=np.float32,
            ),
        )

    def step(self, action: np.ndarray):
        self.prev_voxels = (
            self.voxels
            if self.voxels is not None
            else np.zeros([self.dimension_size**3], dtype=np.float32)
        )
        super().step(action)

    def observe(self):
        # V: voxels, a: action
        # Let an n step episode be:
        # V_0 --a_0--> V_1 --a_1--> V_2 ... V_n-1 --a_n-1-->V_n
        # Then:
        # obs_0 = 0
        # obs_1 = (1, a_0, V_0)
        # obs_2 = (2, a_1, V_1)
        # ...
        # This is the observation used to generate the last action a_n-1
        # obs_n-1 = (n-1, a_n-2, V_n-2)
        # This is the last observation after applying the last action a_n-1
        # obs_n = (n, a_n-1, V_n-1)
        return np.concatenate(
            [
                np.array([len(self.gaussians)], dtype=np.float32),
                super().observe(),
                self.prev_voxels.reshape(-1),
            ],
            axis=0,
        )


class GMMObserveWithVoxelAndRemainingStepsModel(GMMModel):
    def __init__(
        self,
        materials=(0, 1, 2),
        dimension_size=20,
        max_gaussian_num=100,
        cutoff=1e-2,
        reset_seed=42,
        reset_remaining_steps_range=None,
    ):
        super().__init__(materials, dimension_size, max_gaussian_num, cutoff)
        self.prev_voxels = np.zeros([self.dimension_size**3], dtype=np.float32)
        self.reset_remaining_steps_range = reset_remaining_steps_range or (
            max_gaussian_num,
            max_gaussian_num,
        )
        self.reset_rand = np.random.RandomState(reset_seed)
        self.remaining_steps = 0
        self.initial_remaining_steps = 0

    @property
    def observation_space(self):
        return Box(
            low=np.zeros(
                (2 + 6 + len(self.materials) + self.dimension_size**3,),
                dtype=np.float32,
            ),
            high=np.array(
                [self.max_gaussian_num]
                + [self.max_gaussian_num]
                + [1] * (6 + len(self.materials))
                + [len(self.materials) - 1] * self.dimension_size**3,
                dtype=np.float32,
            ),
        )

    def reset(self):
        if self.reset_remaining_steps_range[0] == self.reset_remaining_steps_range[1]:
            self.initial_remaining_steps = (
                self.remaining_steps
            ) = self.reset_remaining_steps_range[0]
        else:
            self.initial_remaining_steps = (
                self.remaining_steps
            ) = self.reset_rand.randint(
                max(self.reset_remaining_steps_range[0], 1),
                self.reset_remaining_steps_range[1] + 1,
            )
        print(self.initial_remaining_steps)
        super().reset()

    def step(self, action: np.ndarray):
        self.prev_voxels = (
            self.voxels
            if self.voxels is not None
            else np.zeros([self.dimension_size**3], dtype=np.float32)
        )
        super().step(action)
        self.remaining_steps -= 1

    def is_finished(self):
        return self.remaining_steps == 0

    def observe(self):
        # V: voxels, a: action
        # Let an n step episode be:
        # V_0 --a_0--> V_1 --a_1--> V_2 ... V_n-1 --a_n-1-->V_n
        # Then:
        # obs_0 = 0
        # obs_1 = (1, a_0, V_0)
        # obs_2 = (2, a_1, V_1)
        # ...
        # This is the observation used to generate the last action a_n-1
        # obs_n-1 = (n-1, a_n-2, V_n-2)
        # This is the last observation after applying the last action a_n-1
        # obs_n = (n, a_n-1, V_n-1)
        return np.concatenate(
            [
                np.array(
                    [
                        len(self.gaussians),
                        # self.remaining_steps / (self.initial_remaining_steps + 1e-3),
                        self.remaining_steps,
                    ],
                    dtype=np.float32,
                ),
                super().observe(),
                self.prev_voxels.reshape(-1),
            ],
            axis=0,
        )
