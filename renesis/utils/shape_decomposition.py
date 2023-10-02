import cc3d
import numpy as np
from scipy import ndimage as ndi

import matplotlib.pyplot as plt


class ShapeDecomposition:
    # Based on "A new shape decomposition scheme for graph-based representation"
    def __init__(
        self,
        shape: np.ndarray,
        iterations: int = 10,
        kernel_size: int = 2,
        merge_threshold: float = 2,
    ):
        # Requires 0/1 in shape
        self.shape = shape != 0
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.merge_threshold = merge_threshold

    def get_segments(self):
        # Extract protruded parts by:
        kernel = self.get_kernel(self.kernel_size)
        segments = []
        shape = self.shape
        for iter in range(self.iterations):
            # 1) opening the shape M, gets M*
            opened_shape = ndi.binary_opening(shape, kernel)
            # 2) subtract M- = M - M*
            protrusion = np.logical_and(shape, ~opened_shape)
            if np.sum(protrusion) == 0:
                # No change, store last shape (main body)
                if np.sum(shape) > 0:
                    segments.append(shape)
                break

            # 3) perform segmentation
            labels, label_num = cc3d.connected_components(
                protrusion, connectivity=18, return_N=True, out_dtype=np.uint64
            )
            for label in range(1, label_num + 1):
                protrusion_segment = labels == label
                coords = np.stack(
                    np.meshgrid(
                        list(range(1, protrusion_segment.shape[0] - 1)),
                        list(range(1, protrusion_segment.shape[1] - 1)),
                        list(range(1, protrusion_segment.shape[2] - 1)),
                        indexing="ij",
                    )
                ).reshape(3, 1, -1)
                offsets = (
                    np.array(
                        [
                            [-1, 0, 0],
                            [1, 0, 0],
                            [0, -1, 0],
                            [0, 1, 0],
                            [0, 0, -1],
                            [0, 0, 1],
                        ]
                    )
                    .transpose(1, 0)
                    .reshape(3, -1, 1)
                )
                # shape [3, 6, segment_space_size]
                neighbor_coords = coords + offsets
                # shape [segment_space_size]
                neighbor_is_opened = np.any(
                    opened_shape[
                        neighbor_coords[0], neighbor_coords[1], neighbor_coords[2]
                    ]
                    != 0,
                    axis=0,
                )
                # shape [segment_space_size]
                self_is_not_empty = (
                    protrusion_segment[coords[0, 0], coords[1, 0], coords[2, 0]] != 0
                )
                adjacent_voxel_num = np.sum(
                    np.logical_and(self_is_not_empty, neighbor_is_opened)
                )

                # 4) Perform protrusion merge/merge
                if (
                    np.sum(protrusion_segment) / adjacent_voxel_num
                    < self.merge_threshold
                ):
                    labels[protrusion_segment] = 0
                    opened_shape[protrusion_segment] = 1
                else:
                    segments.append(protrusion_segment)

            # No change, store last shape (main body)
            if np.all(shape == opened_shape) > 0:
                segments.append(shape)
                break
            shape = opened_shape
        return segments

    def get_kernel(self, diameter):
        if diameter > 2:
            kernel = np.zeros([diameter] * 3, dtype=np.int)
            center_voxel_offset = diameter // 2
            indices = list(
                range(
                    -center_voxel_offset,
                    diameter - center_voxel_offset,
                )
            )
            coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
            # coords shape [coord_num, 3]
            coords = np.transpose(coords.reshape([coords.shape[0], -1]))
            distance = np.linalg.norm(coords, axis=1)
            kernel[
                coords[:, 0] + center_voxel_offset,
                coords[:, 1] + center_voxel_offset,
                coords[:, 2] + center_voxel_offset,
            ] = np.where(distance <= (diameter - 1) / 2, 1, 0)
            return kernel
        elif diameter == 1:
            return np.array([[[1]]])
        elif diameter == 2:
            return np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
