import numpy as np
from typing import Tuple
from collections import deque


class GrowthFunction:
    """Generate patterns given patterns.

    Use the local context to decide what pattern to generate next.
    IE the the configuration of voxels added depend on the proportion
    of the voxel types.

    """

    def __init__(
        self, materials=(0, 1, 2), max_dimension_size=50, max_view_size=21,
    ):
        if max_dimension_size < 5:
            raise ValueError(
                f"Max dimension size is too small, got {max_dimension_size}, "
                f"should be something larger than 5"
            )
        if max_view_size % 2 != 1:
            raise ValueError(
                f"Max view size must be an odd number, got {max_view_size}"
            )
        self.materials = materials
        self.max_dimension_size = max_dimension_size
        self.max_view_size = max_view_size

        self.radius = self.max_view_size // 2
        self.actual_dimension_size = max_dimension_size + self.radius * 2
        self.center_voxel_pos = np.asarray((self.actual_dimension_size // 2,) * 3)
        self.voxels = None
        self.occupied = None
        self.occupied_positions = []
        self.occupied_values = []
        self.num_non_zero_voxel = 0
        self.num_voxels = 0
        self.steps = 0
        self.body = None

        self.reset()

    @property
    def action_shape(self):
        return 6, len(self.materials), 4

    @property
    def view_shape(self):
        return (self.max_view_size,) * 3 + (4,)

    def reset(self):
        self.voxels = np.zeros([self.actual_dimension_size] * 3 + [4], dtype=float)
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

    def building(self):
        """Returns True if there is more to build."""

        return len(self.body) > 0

    def step(self, configuration: np.ndarray):
        """Add one configuration."""
        configuration = self.mask_configuration(configuration)
        voxel = self.body.pop()
        self.attach_voxels(configuration, voxel)
        self.steps += 1

    def get_local_view(self):
        """
        Get local view using the first voxel in queue as the center.
        Returns:
            Numpy float array of shape [max_view_size, max_view_size, max_view_size, 4]
        """
        if len(self.body) == 0:
            return np.zeros((self.max_view_size,) * 3 + (4,))
        voxel = self.body[-1]
        radius = self.max_view_size // 2
        starts = voxel - radius
        ends = voxel + radius + 1
        return self.voxels[
            starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]
        ]

    def get_representation(
        self, amplitude_range=None, frequency_range=None, phase_shift_range=None
    ):
        """
        Returns:
            A tuple of size containing (x, y, z)
            A list of tuples of length z (voxel max height), each tuple is of form
            (material, amplitude, frequency, phase shift), and each element in tuple
            is a list of length x*y, where x and y are the bounding box sizes
            of all voxels.
        """
        amplitude_range = amplitude_range or (0, 1)
        frequency_range = frequency_range or (0, 1)
        phase_shift_range = phase_shift_range or (-1, 1)
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
                    self.voxels[min_x:max_x, min_y:max_y, z, 1], amplitude_range
                )
                .flatten(order="F")
                .tolist(),
                self.rescale(
                    self.voxels[min_x:max_x, min_y:max_y, z, 2], frequency_range
                )
                .flatten(order="F")
                .tolist(),
                self.rescale(
                    self.voxels[min_x:max_x, min_y:max_y, z, 3], phase_shift_range
                )
                .flatten(order="F")
                .tolist(),
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
        Returns: A list of position indicies from 0 to 5.
        """
        voxel = self.body[-1]
        valid_position_indices = []
        for idx, offset in enumerate(
            np.asarray(
                [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
            )
        ):
            pos = voxel + offset
            if np.all(np.array((self.radius,) * 3) <= pos) and np.all(
                np.array((self.max_dimension_size + self.radius,) * 3) > offset
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
        if material != 0:
            self.num_non_zero_voxel += 1
        self.voxels[coordinates[0], coordinates[1], coordinates[2], 0] = material
        self.voxels[coordinates[0], coordinates[1], coordinates[2], 1:] = actuation
        self.occupied[coordinates[0], coordinates[1], coordinates[2]] = True
        self.body.appendleft(coordinates)
        # print(f"New voxel at {coordinates}, material {material}, actuation {actuation}")

    @staticmethod
    def rescale(data: np.ndarray, rescale_range: Tuple[float, float]):
        return data * (rescale_range[1] - rescale_range[0]) + rescale_range[0]
