import cc3d
import numpy as np
from typing import List
from gym.spaces import Box
from .base import BaseModel
from .gmm import normalize, is_voxel_continuous


class PatchModel(BaseModel):
    def __init__(
        self,
        materials=(0, 1, 2),
        dimension_size=20,
        patch_size=1,
        max_patch_num=100,
    ):
        super().__init__()
        self.materials = materials
        self.dimension_size = dimension_size
        self.center_voxel_offset = self.dimension_size // 2
        self.patch_size = patch_size
        self.max_patch_num = max_patch_num

        # A list of arrays of shape [3 + len(self.materials)],
        # first 3 elements are mean (x, y, z)
        # Remaining elements are material weights
        self.patches = []  # type: List[np.ndarray]
        self.prev_voxels = np.zeros([self.dimension_size] * 3, dtype=np.float32)
        self.voxels = np.zeros([self.dimension_size] * 3, dtype=np.float32)
        self.occupied = np.zeros([self.dimension_size] * 3, dtype=np.bool)
        self.invalid_count = 0
        self.is_robot_valid = False
        self.update_voxels()

    @property
    def action_space(self):
        return Box(low=0, high=1, shape=(3 + len(self.materials),))

    @property
    def observation_space(self):
        return Box(
            low=np.array(
                (min(min(self.materials), 0),) * self.dimension_size**3,
                dtype=np.float32,
            ),
            high=np.array(
                (max(max(self.materials), 0),) * self.dimension_size**3,
                dtype=np.float32,
            ),
        )

    def reset(self):
        self.steps = 0
        self.patches = []
        self.update_voxels()
        self.prev_voxels = self.voxels

    def is_finished(self):
        return self.steps >= self.max_patch_num

    def is_robot_invalid(self):
        return not self.is_robot_valid

    def step(self, action: np.ndarray):
        self.prev_voxels = self.voxels
        self.patches.append(action)
        self.update_voxels()
        self.steps += 1

    def observe(self):
        return self.voxels.reshape(-1)

    def get_robot(self):
        labels, label_num = cc3d.connected_components(
            self.occupied, connectivity=6, return_N=True, out_dtype=np.uint32
        )
        count = np.bincount(labels.reshape(-1), minlength=label_num)
        # Ignore label 0, which is non-occupied space
        count[0] = 0
        largest_connected_component = labels == np.argmax(count)
        largest_connected_component_voxels = np.where(
            largest_connected_component, self.voxels, 0
        )

        x_occupied = [
            x
            for x in range(largest_connected_component.shape[0])
            if np.any(largest_connected_component[x])
        ]
        y_occupied = [
            y
            for y in range(largest_connected_component.shape[1])
            if np.any(largest_connected_component[:, y])
        ]
        z_occupied = [
            z
            for z in range(largest_connected_component.shape[2])
            if np.any(largest_connected_component[:, :, z])
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
                largest_connected_component_voxels[min_x:max_x, min_y:max_y, z]
                .astype(int)
                .flatten(order="F")
                .tolist(),
                None,
                None,
                None,
            )
            representation.append(layer_representation)
        return (max_x - min_x, max_y - min_y, max_z - min_z), representation

    def get_largest_connected_component_voxels(self):
        labels, label_num = cc3d.connected_components(
            self.occupied, connectivity=6, return_N=True, out_dtype=np.uint32
        )
        count = np.bincount(labels.reshape(-1), minlength=label_num)
        # Ignore label 0, which is non-occupied space
        count[0] = 0
        largest_connected_component = labels == np.argmax(count)
        largest_connected_component_voxels = np.where(
            largest_connected_component, self.voxels, 0
        )
        return largest_connected_component_voxels

    def get_voxels(self):
        return self.voxels

    def get_state_data(self):
        return np.stack(self.patches), self.voxels

    def scale(self, action):
        min_value = -self.center_voxel_offset - 0.5
        return np.array(
            [min_value, min_value, min_value] + [0] * len(self.materials)
        ) + action * np.array(
            [self.dimension_size, self.dimension_size, self.dimension_size]
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
        patch_radius = self.patch_size / 2
        for idx, patch in enumerate(self.patches):
            patch = self.scale(patch)
            covered = np.all(
                np.array(
                    [
                        coords[:, 0] >= patch[0] - patch_radius,
                        coords[:, 0] < patch[0] + patch_radius,
                        coords[:, 1] >= patch[1] - patch_radius,
                        coords[:, 1] < patch[1] + patch_radius,
                        coords[:, 2] >= patch[2] - patch_radius,
                        coords[:, 2] < patch[2] + patch_radius,
                    ]
                ),
                axis=0,
            )
            # later added patches has a higher weight,
            # so previous patches will be overwritten
            # Add 1 so that the first patch is not zero
            # because idx starts from 0
            all_values.append(covered * (idx + 1))

        self.voxels = np.zeros(
            [self.dimension_size] * 3,
            dtype=np.float32,
        )

        if self.patches:
            # all_values shape [coord_num, patch_num]
            all_values = np.stack(all_values, axis=1)
            material_map = np.array(
                [self.materials[int(np.argmax(patch[3:]))] for patch in self.patches]
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

        # self.occupied = self.voxels[:, :, :] != 0
        # prev_is_valid = self.is_robot_valid
        # self.is_robot_valid = is_voxel_continuous(self.occupied) and np.any(
        #     self.occupied
        # )
        # if self.steps != 0 and not prev_is_valid and not self.is_robot_valid:
        #     self.invalid_count += 1
        # else:
        #     self.invalid_count = 0

        self.occupied = self.voxels[:, :, :] != 0
        self.is_robot_valid = np.any(self.occupied)
