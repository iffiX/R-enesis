import numpy as np
from typing import Tuple
from collections import deque
from gym.spaces import Box
from .base import BaseModel


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


class GrowthModel(BaseModel):
    """
    The growth model. A couple of voxels is chosen to be added in
    each action.

    Use the local context to decide what pattern to generate next.
    IE the the configuration of voxels added depend on the proportion
    of the voxel types.

    """

    def __init__(
        self,
        materials=(0, 1, 2),
        max_dimension_size=50,
        max_view_size=21,
        actuation_features=("amplitude", "frequency", "phase_offset"),
        amplitude_range=(0, 1),
        frequency_range=(0, 1),
        phase_offset_range=(-1, 1),
    ):
        super().__init__()
        if max_dimension_size < 5:
            raise ValueError(
                f"Max dimension size is too small, got {max_dimension_size}, "
                f"should be something larger than 5"
            )
        if max_view_size % 2 != 1:
            raise ValueError(
                f"Max view size must be an odd number, got {max_view_size}"
            )
        for a in actuation_features:
            assert a in ("amplitude", "frequency", "phase_offset")
        assert len(actuation_features) > 0

        self.materials = materials
        self.max_dimension_size = max_dimension_size
        self.max_view_size = max_view_size
        self.actuation_features = actuation_features
        self.amplitude_range = amplitude_range
        self.frequency_range = frequency_range
        self.phase_offset_range = phase_offset_range

        self.radius = self.max_view_size // 2
        self.actual_dimension_size = max_dimension_size + self.radius * 2
        self.center_voxel_pos = np.asarray((self.actual_dimension_size // 2,) * 3)
        self.voxels = None
        self.occupied = None
        self.occupied_positions = []
        self.occupied_values = []
        self.num_non_zero_voxel = 0
        self.num_voxels = 0
        self.body = None

        self.action_shape = (6, len(self.materials), 1 + len(self.actuation_features))
        self.view_shape = (self.max_view_size,) * 3 + (
            1 + len(self.actuation_features),
        )
        self.reset()

    @property
    def action_space(self):
        return Box(low=0, high=1, shape=np.prod(self.action_shape))

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=self.view_shape)

    def reset(self):
        self.voxels = np.zeros(
            [self.actual_dimension_size] * 3 + [1 + len(self.actuation_features)],
            dtype=float,
        )
        self.occupied = np.zeros([self.actual_dimension_size] * 3, dtype=bool)
        self.occupied_positions = []
        self.occupied_values = []
        self.num_non_zero_voxel = 0
        self.num_voxels = 0
        self.steps = 0
        self.body = deque([])

        # Create the empty origin voxel, since it is empty we
        # restrict the attached voxel number of the first step
        # to 1 to prevent generating non-attaching voxels.
        self.body.appendleft(self.center_voxel_pos)

    def is_finished(self):
        return len(self.body) == 0

    def is_robot_empty(self):
        return self.num_non_zero_voxel == 0

    def step(self, action: np.ndarray):
        configuration = sigmoid(action.reshape(self.action_shape))
        configuration = self.mask_configuration(configuration)
        voxel = self.body.pop()
        self.attach_voxels(configuration, voxel)
        self.steps += 1

    def observe(self):
        """
        Get local view using the first voxel in queue as the center.
        Returns:
            Numpy float array of shape [max_view_size, max_view_size, max_view_size, 1 + actuation_features_num]
        """
        if len(self.body) == 0:
            return np.zeros(
                (self.max_view_size,) * 3 + (1 + len(self.actuation_features),)
            )
        voxel = self.body[-1]
        radius = self.max_view_size // 2
        starts = voxel - radius
        ends = voxel + radius + 1
        out = np.copy(
            self.voxels[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]]
        )
        out[:, :, :, 0] /= max(self.materials)
        return out

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
                self.voxels[min_x:max_x, min_y:max_y, z, 0]
                .astype(int)
                .flatten(order="F")
                .tolist(),
                self.rescale(
                    self.voxels[
                        min_x:max_x,
                        min_y:max_y,
                        z,
                        1 + self.actuation_features.index("amplitude"),
                    ],
                    self.amplitude_range,
                )
                .flatten(order="F")
                .tolist()
                if "amplitude" in self.actuation_features
                else None,
                self.rescale(
                    self.voxels[
                        min_x:max_x,
                        min_y:max_y,
                        z,
                        1 + self.actuation_features.index("frequency"),
                    ],
                    self.frequency_range,
                )
                .flatten(order="F")
                .tolist()
                if "frequency" in self.actuation_features
                else None,
                self.rescale(
                    self.voxels[
                        min_x:max_x,
                        min_y:max_y,
                        z,
                        1 + self.actuation_features.index("phase_offset"),
                    ],
                    self.phase_offset_range,
                )
                .flatten(order="F")
                .tolist()
                if "phase_offset" in self.actuation_features
                else None,
            )
            representation.append(layer_representation)
        return (max_x - min_x, max_y - min_y, max_z - min_z), representation

    def mask_configuration(self, configuration: np.ndarray):
        """
        Mask invalid configuration in the current step
        Eg: mask invalid attach position, mask invalid material (not implemented)
        """
        masked_configuration = configuration.copy()
        if self.steps == 0:
            # return first valid position, which is x_negative
            masked_configuration[1:] = 0
        else:
            valid_position_indices = set(self.get_valid_position_indices())
            for i in range(6):
                if i not in valid_position_indices:
                    masked_configuration[i] = 0
        return masked_configuration

    def get_valid_position_indices(self):
        """
        Returns: A list of position indices from 0 to 5.
        """
        voxel = self.body[-1]
        valid_position_indices = []
        for idx, offset in enumerate(
            np.asarray(
                [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
            )
        ):
            pos = voxel + offset
            if (
                np.all(np.array((self.radius,) * 3) <= pos)
                and np.all(np.array((self.max_dimension_size + self.radius,) * 3) > pos)
                and not self.occupied[pos[0], pos[1], pos[2]]
            ):
                valid_position_indices.append(idx)
        return valid_position_indices

    def attach_voxels(
        self, configuration: np.ndarray, current_voxel: Tuple[int, int, int]
    ):
        """
        Attach a configuration of voxels to the current voxel.

        Args:
            configuration: an array of shape [6, material_num, 4]
            current_voxel: current voxel coordinate

        Note:
            Directions order: "negative_x", "positive_x", "negative_y",
                "positive_y", "negative_z", "positive_z",
        """
        # print(configuration)
        for direction in range(6):
            if np.all(configuration[direction, :, 0] == 0):
                continue
            material = int(np.argmax(configuration[direction, :, 0]))
            actuation = configuration[direction, material, 1:]
            if direction == 0:
                self.create_new_voxel(
                    current_voxel + np.asarray((-1, 0, 0)), material, actuation,
                )

            elif direction == 1:
                self.create_new_voxel(
                    current_voxel + np.asarray((1, 0, 0)), material, actuation,
                )

            elif direction == 2:
                self.create_new_voxel(
                    current_voxel + np.asarray((0, -1, 0)), material, actuation,
                )

            elif direction == 3:
                self.create_new_voxel(
                    current_voxel + np.asarray((0, 1, 0)), material, actuation,
                )

            elif direction == 4:
                self.create_new_voxel(
                    current_voxel + np.asarray((0, 0, -1)), material, actuation,
                )

            elif direction == 5:
                self.create_new_voxel(
                    current_voxel + np.asarray((0, 0, 1)), material, actuation,
                )

    def create_new_voxel(
        self, coordinates: np.ndarray, material: int, actuation: np.ndarray
    ):
        """
        Args:
            coordinates: array of shape [3], x, y, z coords
            material: material index
            actuation: array of shape [3], amplitude, frequency and phase shift
        """
        self.occupied_positions.append(coordinates.tolist())
        self.occupied_values.append(material)
        self.num_voxels += 1
        self.voxels[coordinates[0], coordinates[1], coordinates[2], 0] = material
        self.occupied[coordinates[0], coordinates[1], coordinates[2]] = True

        # Only store actuation data if a non-0 voxel is attached
        # Since we can only attach new voxels to a non-0 voxel, only append
        # it to the body queue when voxel is not 0.
        if material != 0:
            self.num_non_zero_voxel += 1
            self.voxels[coordinates[0], coordinates[1], coordinates[2], 1:] = actuation
            self.body.appendleft(coordinates)
        # print(f"New voxel at {coordinates}, material {material}, actuation {actuation}")

    @staticmethod
    def rescale(data: np.ndarray, rescale_range: Tuple[float, float]):
        return data * (rescale_range[1] - rescale_range[0]) + rescale_range[0]
