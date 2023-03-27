# import numpy as np
# import pyvista as pv
#
# # Create shapes for the dog
# head = pv.ParametricEllipsoid(3, 2, 2)
# muzzle = pv.ParametricEllipsoid(1.5, 1, 1).translate((2.5, 0, 0))
# ears = [
#     pv.Cylinder(radius=0.5, height=2).rotate_y(90).translate((1.5, 1, 2)),
#     pv.Cylinder(radius=0.5, height=2).rotate_y(90).translate((1.5, -1, 2)),
# ]
# body = pv.ParametricEllipsoid(5, 2, 2).translate((5, 0, 0))
# legs = []
# for i in range(4):
#     x_offset = 4 if i < 2 else 8
#     y_offset = 1 if i % 2 == 0 else -1
#     upper_leg = (
#         pv.Cylinder(radius=0.5, height=3)
#         .rotate_y(90)
#         .translate((x_offset, y_offset * 2, -3))
#     )
#     lower_leg = (
#         pv.Cylinder(radius=0.5, height=3)
#         .rotate_y(90)
#         .translate((x_offset, y_offset * 2, -6))
#     )
#     paw = pv.ParametricEllipsoid(1, 0.5, 0.5).translate((x_offset, y_offset * 2, -8))
#     leg_joint = pv.Sphere(radius=0.7).translate((x_offset, y_offset * 2, -3))
#     legs.extend([upper_leg, lower_leg, paw, leg_joint])
#
# tail = pv.Cylinder(radius=0.5, height=4).rotate_y(30).translate((10, 0, 1))
#
# # Display the dog model in a PyVista plot
# plotter = pv.Plotter()
# plotter.add_mesh(head, color="brown")
# plotter.add_mesh(muzzle, color="brown")
# for ear in ears:
#     plotter.add_mesh(ear, color="brown")
# plotter.add_mesh(body, color="brown")
# for leg in legs:
#     plotter.add_mesh(leg, color="brown")
# plotter.add_mesh(tail, color="brown")
# plotter.show()

import numpy as np
import pyvista as pv

# Head (ellipsoid)
head = pv.ParametricEllipsoid(3, 3, 3)
head.translate((5, 0, 0))

# Muzzle (smaller ellipsoid)
muzzle = pv.ParametricEllipsoid(1.5, 1, 1)
muzzle.translate((9, 0, 0))

# Body (cylinder)
body = pv.Cylinder(radius=2.5, height=8)
body.translate((2, 0, 0))
body.rotate_z(90)

# Legs (cylinders)
leg_radius = 0.5
leg_height = 3

front_right_leg_upper = pv.Cylinder(radius=leg_radius, height=leg_height)
front_right_leg_upper.translate((2, 2.5, 0))
front_right_leg_lower = pv.Cylinder(radius=leg_radius, height=leg_height)
front_right_leg_lower.translate((2, 2.5, -3))

front_left_leg_upper = pv.Cylinder(radius=leg_radius, height=leg_height)
front_left_leg_upper.translate((2, -2.5, 0))
front_left_leg_lower = pv.Cylinder(radius=leg_radius, height=leg_height)
front_left_leg_lower.translate((2, -2.5, -3))

rear_right_leg_upper = pv.Cylinder(radius=leg_radius, height=leg_height)
rear_right_leg_upper.translate((8, 2.5, 0))
rear_right_leg_lower = pv.Cylinder(radius=leg_radius, height=leg_height)
rear_right_leg_lower.translate((8, 2.5, -3))

rear_left_leg_upper = pv.Cylinder(radius=leg_radius, height=leg_height)
rear_left_leg_upper.translate((8, -2.5, 0))
rear_left_leg_lower = pv.Cylinder(radius=leg_radius, height=leg_height)
rear_left_leg_lower.translate((8, -2.5, -3))

# Paws (ellipsoids)
paw_radius = 0.8
paws = [pv.ParametricEllipsoid(paw_radius, paw_radius, paw_radius) for _ in range(4)]

paw_positions = [(2, 2.5, -6), (2, -2.5, -6), (8, 2.5, -6), (8, -2.5, -6)]

for i, paw in enumerate(paws):
    paw.translate(paw_positions[i])

# Tail (series of connected cylinders)
tail = pv.MultiBlock()
tail_radius = 0.3
tail_height = 1
tail_parts = 5
tail_curvature = -15

for i in range(tail_parts):
    tail_part = pv.Cylinder(radius=tail_radius, height=tail_height)
    tail_part.translate((i * tail_height, 0, 2.5))
    tail_part.rotate_z(i * tail_curvature)
    tail[i] = tail_part

# Create a plotter and add all shape objects to a single graph
plotter = pv.Plotter()
plotter.add_mesh(head, color="brown")
plotter.add_mesh(muzzle, color="brown")
plotter.add_mesh(body, color="brown")
plotter.add_mesh(front_right_leg_upper, color="brown")
plotter.add_mesh(front_right_leg_lower, color="brown")
plotter.add_mesh(front_left_leg_upper, color="brown")
plotter.add_mesh(front_left_leg_lower, color="brown")
plotter.add_mesh(rear_right_leg_upper, color="brown")
plotter.add_mesh(rear_right_leg_lower, color="brown")
plotter.add_mesh(rear_left_leg_upper, color="brown")
plotter.add_mesh(rear_left_leg_lower, color="brown")

for paw in paws:
    plotter.add_mesh(paw, color="brown")

for tail_part in tail:
    plotter.add_mesh(tail_part, color="brown")

# Show the plot
plotter.show()
