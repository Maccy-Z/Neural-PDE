import numpy as np
from cprint import c_print
import meshpy.triangle as tri
import threading

from pde.mesh_generation.mesh_gen_utils import (MeshProps, min_dist_to_boundary,
                            plot_mesh, extract_mesh_data)
from pde.mesh_generation.geometries import MeshFacet, Circle, Box, Line, Ellipse, Nozzle
from pde.graph_grid.graph_store import P_Types as PT

# Custom function to control mesh refinement
def refine_fn(vertices, area, props: MeshProps, points, segments):
    # Wrapper function hides exceptions raised here.
    try:
        """ True if area is too big. False if area is small enough"""
        if area < props.min_area:
            return False
        if area > props.max_area:
            return True
        centroid = np.mean(vertices, axis=0)
        dist = min_dist_to_boundary(centroid, points, segments)

        # Increase refinement near the boundaries and if the area is too large
        threshold = (props.max_area - props.min_area) * (1 - np.exp(-dist / props.lengthscale)) + props.min_area

    except Exception as e:
        c_print(f"Exception raised: {e}", color="bright_red")
        print(e)
        raise e
    return area > threshold


def _create_mesh_thread(holes, points, p_marks, segments, seg_marks, mesh_props, dist_p, dist_seg, return_val, min_angle):
    mesh_info = tri.MeshInfo()
    mesh_info.set_holes(holes)
    mesh_info.set_points(points, point_markers=p_marks)
    mesh_info.set_facets(segments, facet_markers=seg_marks)

    # Create the mesh
    mesh = tri.build(mesh_info, refinement_func=lambda x, y: refine_fn(x, y, mesh_props, dist_p, dist_seg), min_angle=min_angle)

    return_val.append(mesh)

def create_mesh(coords: list[MeshFacet], mesh_props: MeshProps, min_angle=None):
    # Collate together all facet objects
    points, segments = np.empty((0, 2)), np.empty((0, 2), dtype=int)
    # Segments for dist calculation
    dist_p, dist_seg = np.empty((0, 2)), np.empty((0, 2), dtype=int)

    holes = []
    seg_marks, p_marks = [], []
    marker_names = {0: "Normal"}

    for i, facets in enumerate(coords):
        cur_p = len(points)

        if facets.real_face:
            points = np.concatenate((points, facets.points))
            segments = np.concatenate((segments, facets.segments + cur_p))

            # Default marker is 0, so start at 1
            mark_id = i + 1
            seg_marks += [mark_id] * len(facets.segments)
            p_marks += [mark_id] * len(facets.points)
            if facets.hole:
                holes += facets.hole

            marker_names[mark_id] = facets.name

        if facets.dist_req:
            cur_dist_p = len(dist_p)
            dist_p = np.concatenate((dist_p, facets.points))
            dist_seg = np.concatenate((dist_seg, facets.segments + cur_dist_p))


    ret_list = []
    thread = threading.Thread(target=_create_mesh_thread, args=(holes, points, p_marks, segments, seg_marks, mesh_props, dist_p, dist_seg, ret_list, min_angle))
    thread.start()
    thread.join()
    mesh = ret_list[0]
    return mesh, marker_names


def gen_points_full():
    min_area = 1e-3
    max_area = 3e-3
    xmin, xmax = 0, 2
    ymin, ymax = 0.0, 1.5
    circle_center = (0.5, 0.4)
    circle_radius = 0.1

    lengthscale = np.sqrt(2*min_area)

    mesh_props = MeshProps(min_area, max_area, lengthscale=0.4)

    coords = [#Box(Xmin, Xmax, hole=False, name="farfield", remove_edge=2),
              Line([[xmin, ymin], [xmax, ymin]], dist_req=True, name="wall_bottom"),
              Line([[xmin, ymax], [xmax, ymax]], True, name="wall_top"),
              Line([[xmin, ymin], [xmin, ymax]], True, name="wall_left"),
              Line([[xmax, ymax], [xmax, ymin]], True, name="wall_right"),


              Circle((0.9, 0.75), 0.2, lengthscale, True, name="circle", lims=[[xmin, ymin], [xmax, ymax]]),
              # Ellipse((2.0, 1), 0.2, 0.75, angle=np.pi/3, lengthscale=lengthscale, hole=True, dist_req=True, name=PT.DirichBC),
              ]
    #
    # coords = [#Box(Xmin, Xmax, hole=False, name="farfield", remove_edge=2),
    #             Line([[xmin, ymin], [1., ymin]], dist_req=True, name="wall_bottom"),
    #             Line([[xmin, ymax], [1., ymax]], True, name="wall_top"),
    #             Line([[xmin, ymin], [xmin, ymax]], True, name="wall_left"),
    #             Line([[1., ymax], [1., ymin]], True, name="wall_right"),
    #
    #             Line([[1.025, ymin], [2., ymin]], dist_req=True, name="wall_bottom"),
    #             Line([[1.025, ymax], [2, ymax]], True, name="wall_top"),
    #             Line([[1.025, ymin], [1.025, ymax]], True, name="wall_left"),
    #             Line([[2, ymax], [2, ymin]], True, name="wall_right"),
    #
    #           # Circle(circle_center, circle_radius, lengthscale, True, name=PT.DirichBC),
    #           ]

    mesh, marker_tags = create_mesh(coords, mesh_props)
    point_props, markers, _edges = extract_mesh_data(mesh)
    points, triangles = point_props
    p_markers, _ = markers
    int_edges, bound_edges = _edges

    p_tags = [marker_tags[int(i)] for i in p_markers]
    return points, triangles, (int_edges, bound_edges), p_tags


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


def main():
    #exit(4)
    try:
        points, triangles, (int_edges, bound_edges), f_tag = gen_mesh_fvm(0, 2, 0, 2)
        #pickle.dump((points, p_tags), sys.stdout.buffer)
    except Exception as e:
        raise e


if __name__ == "__main__":
    #from pde.graph_grid.graph_store import P_Types as PT
    main()
