import  numpy as np

from mesh_gen.generate_mesh import MeshProps, create_mesh, extract_mesh_data
from mesh_gen.geometries import Line, Ellipse, Nozzle, Circle

def gen_mesh_fvm(areas, cell_lnscale=2):
    xmin, xmax = -0.25, 6
    ymin, ymax = -2.25, 1.5

    min_area, max_area = areas
    mesh_props = MeshProps(min_area, max_area, lengthscale=cell_lnscale)
    triscale = np.sqrt(2 * min_area)
    lims = [xmin, ymin], [xmax, ymax]


    coords = [
                Line([[xmin, ymin], [xmax, ymin]], False, name="Side"),     # Bottom
                Line([[xmin, ymax], [xmax, ymax]], False, name="Side"),     # Top
                Line([[xmin, ymin], [xmin, ymax]], True, name="Left"),    # Left
                Line([[xmax, ymax], [xmax, ymin]], False, name="Right"),   # Right
                # Line([[0.75, 0.7], [xmax, 0.7]], True, real=False, name=None),  # Refinement wall
                # Circle((0.75, 0.7), 0.15, triscale, hole=True, dist_req=True, name="NavierWall"),
                # Circle((4, -0.15), 0.15, triscale, hole=True, dist_req=True, name="NavierWall"),

                Ellipse((4, -0.15), 0.2, 0.8, angle=-0.3, lengthscale=triscale, lims=lims, hole=True, dist_req=True, name="NavierWall"),
                Nozzle(Xmin=[0, 0], Rt=0.33, Re=1, theta_n_deg=30, theta_exit_deg=15, lengthscale=triscale, lip_size=1., dist_req=True, name="NavierWall"),

    ]

    mesh, marker_tags = create_mesh(coords, mesh_props)
    _point_props, _markers, _edges = extract_mesh_data(mesh)

    points, triangles = _point_props
    _, f_markers = _markers
    int_edges, bound_edges = _edges

    # Change maker back to string
    f_tag = [marker_tags[int(i)] for i in f_markers]

    return points, triangles, (int_edges, bound_edges), f_tag

def gen_mesh_tunnel(areas, cell_lnscale=2):
    xmin, xmax = 0, 3
    ymin, ymax = 0, 1.5

    min_area, max_area = areas
    mesh_props = MeshProps(min_area, max_area, lengthscale=cell_lnscale)
    triscale = np.sqrt(2 * min_area)
    lims = [xmin, ymin], [xmax, ymax]


    coords = [
                Line([[xmin, ymin], [xmax, ymin]], False, name="NavierWall"),     # Bottom
                Line([[xmin, ymax], [xmax, ymax]], False, name="NavierWall"),     # Top
                Line([[xmin, ymin], [xmin, ymax]], False, name="Left"),    # Left
                Line([[xmax, ymax], [xmax, ymin]], False, name="Right"),   # Right
                Line([[0.75, 0.7], [2.5, 0.7]], True, real=False, name=None),  # Refinement wall
                Circle((0.75, 0.7), 0.15, triscale, hole=True, dist_req=True, name="NavierWall"),
                Circle((1.05, 0.9), 0.15, triscale, hole=True, dist_req=True, name="NavierWall"),

        # Circle((4, -0.15), 0.15, triscale, hole=True, dist_req=True, name="NavierWall"),

                # Ellipse((4, -0.15), 0.2, 0.8, angle=-0.3, lengthscale=triscale, lims=lims, hole=True, dist_req=True, name="NavierWall"),
                # Nozzle(Xmin=[0, 0], Rt=0.33, Re=1, theta_n_deg=30, theta_exit_deg=15, lengthscale=triscale, lip_size=1., dist_req=True, name="NavierWall"),

    ]

    mesh, marker_tags = create_mesh(coords, mesh_props, min_angle=30)
    _point_props, _markers, _edges = extract_mesh_data(mesh)

    points, triangles = _point_props
    _, f_markers = _markers
    int_edges, bound_edges = _edges

    # Change maker back to string
    f_tag = [marker_tags[int(i)] for i in f_markers]

    return points, triangles, (int_edges, bound_edges), f_tag

