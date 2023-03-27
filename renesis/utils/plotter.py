import pyvista
import vtk
import pyvista as pv
import numpy as np
from typing import List, Union, overload


class Plotter:
    def __init__(self, interactive=False):
        self.interactive = interactive

    @overload
    def plot_voxel(
        self,
        voxel: np.ndarray,
        palette: Union[List[List[float]], List[str], np.ndarray] = None,
        distance: float = 10,
        plotter: pyvista.Plotter = None,
    ) -> Union[np.ndarray, None]:
        """
        Args:
            voxel: 3D array of voxels.
            palette: Array of shape [m, 3], where m is the number of materials,
                The first material corresponds to material 1. Or a list of strings.
            distance: Camera distance. Set to -1 for auto distance.
            plotter: pyvista plotter

        Returns:
            An image of shape [height, width, 3]
            or None if not interactive or plotter is set
        """
        ...

    @overload
    def plot_voxel(
        self,
        origin: np.ndarray,
        material: np.ndarray,
        palette: Union[List[List[float]], List[str], np.ndarray] = None,
        distance: float = 10,
        plotter: pyvista.Plotter = None,
    ):
        """
        Args:
            origin: Array of shape [n, 3], where n is the number of voxels,
                each row is the origin of a voxel.
            material: Array of shape [n], where n is the number of voxels,
                each row is the material index of a voxel. Material index
                starts from 1 and there should never be a null material 0.
            palette: Array of shape [m, 3], where m is the number of materials,
                The first material corresponds to material 1. Or a list of strings.
            distance: Camera distance. Set to -1 for auto distance.
            plotter: pyvista plotter

        Returns:
            An image of shape [height, width, 3]
            or None if plotter is interactive or plotter is set
        """
        ...

    def plot_voxel(
        self,
        *args,
        palette: Union[List[List[float]], List[str], np.ndarray] = None,
        distance: float = 10,
        plotter: pyvista.Plotter = None,
    ):
        has_input_plotter = plotter is not None
        plotter = plotter or pv.Plotter(off_screen=not self.interactive)

        if len(args) == 1:
            origin, material = self.voxel_array_to_origin_and_material(args[0])
        else:
            origin, material = args
            if len(origin) == 0 or len(material) == 0:
                origin = None
                material = None

        if material is not None:
            if np.any(np.array(material) == 0):
                raise ValueError("Null material in material array")

            points, num_voxels, num_vertices = self.get_vertices_of_voxel(
                np.array(origin)
            )

            if palette is None or (
                isinstance(palette, list) and palette and isinstance(palette[0], str)
            ):
                # Prepare the vtk grid object and plot.
                cells_hex = np.arange(num_vertices).reshape((num_voxels, 8))
                grid = pv.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells_hex}, points)
                plotter.add_mesh(
                    grid,
                    show_edges=True,
                    scalars=np.array(material),
                    show_scalar_bar=False,
                    clim=[1, 4 if palette is None else len(palette)],
                    below_color="black",
                    above_color="white",
                    cmap=palette or ["red", "blue", "green", "yellow"],
                    opacity=0.9,
                )
            else:
                cells_hex = np.arange(8).reshape((1, 8))
                for v in range(num_voxels):
                    grid = pv.UnstructuredGrid(
                        {vtk.VTK_HEXAHEDRON: cells_hex}, points[v * 8 : v * 8 + 8]
                    )
                    plotter.add_mesh(
                        grid,
                        show_edges=True,
                        color=list(palette[int(material[v] - 1)]),
                        opacity=0.9,
                    )
        plotter.add_floor("-z")
        plotter.enable_depth_peeling(1)
        camera, focus, viewup = plotter.get_default_cam_pos()
        plotter.camera_position = [
            (camera[0] - distance, camera[1] - distance, camera[2]),
            focus,
            viewup,
        ]
        if not has_input_plotter:
            if self.interactive:
                plotter.show()
                img = None
            else:
                img = plotter.screenshot(return_img=True)
            plotter.close()
            pv.close_all()
            return img

    def plot_voxel_error(
        self,
        voxel_ref: np.ndarray,
        voxel_input: np.ndarray,
        palette: Union[List[List[float]], List[str], np.ndarray] = None,
        distance: float = 10,
    ):
        """
        Args:
            voxel_ref: 3D array of shape reference.
            voxel_input: 3D array of shape input.
            palette: Array of shape [m, 3], where m is the number of materials,
                The first material corresponds to material 1. Or a list of strings.
            distance: Camera distance. Set to -1 for auto distance.
            plotter: pyvista plotter

        Returns:
            An image of shape [height, width, 3], or None if plotter is interactive
        """
        if voxel_ref.shape != voxel_input.shape:
            raise ValueError(
                "Reference voxel shape must be the same as input voxel shape"
            )
        plotter = pv.Plotter(off_screen=not self.interactive, shape=(1, 3))

        plotter.subplot(0, 0)
        plotter.add_text("Reference", font_size=10)
        self.plot_voxel(
            *self.voxel_array_to_origin_and_material(voxel_ref),
            palette=palette,
            distance=distance,
            plotter=plotter,
        )

        plotter.subplot(0, 1)
        plotter.add_text("Input", font_size=10)
        self.plot_voxel(
            *self.voxel_array_to_origin_and_material(voxel_input),
            palette=palette,
            distance=distance,
            plotter=plotter,
        )

        plotter.subplot(0, 2)
        plotter.add_text("Error", font_size=10)
        error = self.get_voxel_error(voxel_ref, voxel_input)
        # Plot error
        error_origins, error_materials = self.voxel_array_to_origin_and_material(error)
        error_points, error_num_voxels, error_num_vertices = self.get_vertices_of_voxel(
            error_origins
        )
        error_cells_hex = np.arange(error_num_vertices).reshape((error_num_voxels, 8))
        error_grid = pv.UnstructuredGrid(
            {vtk.VTK_HEXAHEDRON: error_cells_hex}, error_points
        )
        plotter.add_mesh(
            error_grid,
            show_edges=True,
            scalars=np.array(error_materials),
            annotations={1: "missing", 2: "excess", 3: "wrong"},
            cmap=["purple", "blue", "orange"],
            clim=[1, 3],
            opacity=0.9,
        )
        # Plot reference edge frame
        ref_origins, ref_materials = self.voxel_array_to_origin_and_material(voxel_ref)
        ref_points, ref_num_voxels, ref_num_vertices = self.get_vertices_of_voxel(
            ref_origins
        )
        ref_cells_hex = np.arange(ref_num_vertices).reshape((ref_num_voxels, 8))
        ref_grid = pv.UnstructuredGrid({vtk.VTK_HEXAHEDRON: ref_cells_hex}, ref_points)
        edge = ref_grid.extract_all_edges()
        plotter.add_mesh(edge, opacity=0.5, color="white")
        plotter.add_floor("-z")
        plotter.enable_depth_peeling(1)
        camera, focus, viewup = plotter.get_default_cam_pos()
        plotter.camera_position = [
            (camera[0] - distance, camera[1] - distance, camera[2]),
            focus,
            viewup,
        ]

        if self.interactive:
            plotter.show()
            img = None
        else:
            img = plotter.screenshot(return_img=True)
        plotter.close()
        pv.close_all()
        return img

    @staticmethod
    def voxel_array_to_origin_and_material(voxel: np.ndarray):
        """
        Args:
            voxel: 3D array of voxels.

        Returns:
            Origin array and material array used for calling plot_voxel()
        """
        positions = np.nonzero(voxel)
        if len(positions[0]) == 0:
            return None, None
        return (
            np.stack(np.nonzero(voxel), axis=1),
            voxel[positions[0], positions[1], positions[2]],
        )

    @staticmethod
    def get_voxel_error(voxel_ref: np.ndarray, voxel_input: np.ndarray):
        missing = np.logical_and(voxel_ref != 0, voxel_input == 0)
        excessive = np.logical_and(voxel_ref == 0, voxel_input != 0)
        wrong = np.logical_and(
            np.logical_and(voxel_ref != 0, voxel_input != 0), voxel_ref != voxel_input,
        )
        voxel_error = np.zeros_like(voxel_ref)
        voxel_error[missing] = 1
        voxel_error[excessive] = 2
        voxel_error[wrong] = 3
        return voxel_error

    @staticmethod
    def get_vertices_of_voxel(origin: np.ndarray):
        """
        Args:
            origin: Array of shape [n, 3], where n is the number of voxels,
               each row is the origin of a voxel.

        Returns:
            Vertices array of shape [8*n, 3],
            Number of voxels (n),
            Number of vertices (8 * n)
        """
        n = origin.shape[0]

        # Add each of the voxel's vertices.
        # Lower level.
        # p4       p3
        #    -----
        #    |   |
        #    -----
        # p1 (x)   p2
        p1, p2, p3, p4 = (
            origin.copy(),
            origin.copy(),
            origin.copy(),
            origin.copy(),
        )
        p2[:, 0] += 1
        p3[:, 0] += 1
        p3[:, 1] += 1
        p4[:, 1] += 1

        # Upper level.
        # p8       p7
        #    -----
        #    |   |
        #    -----
        # p5       p6
        p5 = origin.copy()
        p5[:, 2] += 1
        p6, p7, p8 = (
            p5.copy(),
            p5.copy(),
            p5.copy(),
        )
        p6[:, 0] += 1
        p7[:, 0] += 1
        p7[:, 1] += 1
        p8[:, 1] += 1

        # Weave the eight coordinates of the voxel together.
        num_vertices = n * 8
        points = np.empty((num_vertices, 3), dtype=origin.dtype)
        points[0::8, :] = p1
        points[1::8, :] = p2
        points[2::8, :] = p3
        points[3::8, :] = p4
        points[4::8, :] = p5
        points[5::8, :] = p6
        points[6::8, :] = p7
        points[7::8, :] = p8

        return points.astype(float), n, num_vertices
