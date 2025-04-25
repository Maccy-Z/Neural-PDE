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


def _create_mesh_thread(holes, points, p_marks, segments, seg_marks, mesh_props, dist_p, dist_seg, return_val):
    mesh_info = tri.MeshInfo()
    mesh_info.set_holes(holes)
    mesh_info.set_points(points, point_markers=p_marks)
    mesh_info.set_facets(segments, facet_markers=seg_marks)

    # Create the mesh
    mesh = tri.build(mesh_info, refinement_func=lambda x, y: refine_fn(x, y, mesh_props, dist_p, dist_seg), min_angle=30)

    return_val.append(mesh)

def create_mesh(coords: list[MeshFacet], mesh_props: MeshProps):
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
    thread = threading.Thread(target=_create_mesh_thread, args=(holes, points, p_marks, segments, seg_marks, mesh_props, dist_p, dist_seg, ret_list))
    thread.start()
    thread.join()
    mesh = ret_list[0]
    return mesh, marker_names


def gen_points_full():
    min_area = 7e-3
    max_area = 10e-3
    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    circle_center = (0.5, 0.4)
    circle_radius = 0.1

    lengthscale = np.sqrt(2*min_area)
    # print(lengthscale)

    mesh_props = MeshProps(min_area, max_area, lengthscale=0.4)

    coords = [#Box(Xmin, Xmax, hole=False, name="farfield", remove_edge=2),
              Line([xmin, ymin], [xmax, ymin], True, name=PT.DirichBC),
              Line([xmin, ymax], [xmax, ymax], True, name=PT.DirichBC),
              Line([xmin, ymin], [xmin, ymax], True, name=PT.NeumOffsetBC),
              Line([xmax, ymax], [xmax, ymin], True, name=PT.DirichBC),
              Circle(circle_center, circle_radius, lengthscale, True, name=PT.DirichBC),
              Circle((1.0, 0.5), circle_radius, lengthscale, True, name=PT.DirichBC),
              Circle((1.0, 0.8), circle_radius, lengthscale, True, name=PT.DirichBC),
              Ellipse((2.0, 1), 0.2, 0.75, np.pi/3, lengthscale, True, dist_req=True, name=PT.DirichBC),
        # Line([1, 0.1], [1, 0.5], name="Inlet1")
              ]
    mesh, marker_tags = create_mesh(coords, mesh_props)
    point_props, markers, _ = extract_mesh_data(mesh)
    points, _ = point_props
    p_markers, _ = markers

    p_tags = [marker_tags[int(i)] for i in p_markers]

    #plot_mesh(mesh)
    return points, p_tags


def generate_box_points_spacing(xmax, ymax, spacing=1.0):
    """
    Generates an array of 2D points outlining a box from [0, 0] to [xmax, ymax],
    with approximately the specified spacing between consecutive points.

    Parameters:
    - xmax (float): The maximum x-coordinate of the box.
    - ymax (float): The maximum y-coordinate of the box.
    - spacing (float, optional): Desired spacing between points. Default is 1.0.

    Returns:
    - numpy.ndarray: An array of shape (N, 2) containing the 2D points,
                     where N depends on the spacing and box dimensions.
    """
    if spacing <= 0:
        raise ValueError("Spacing must be a positive number.")

    def compute_num_points(length, _spacing):
        return max(int(np.ceil(length / _spacing)), 1)  # At least 1 point

    num_points_bottom = compute_num_points(xmax, spacing)
    num_points_right = compute_num_points(ymax, spacing)
    num_points_top = compute_num_points(xmax, spacing)
    num_points_left = compute_num_points(ymax, spacing)

    # Bottom edge: from (0, 0) to (xmax, 0)
    bottom = np.linspace([0, 0], [xmax, 0], num=num_points_bottom, endpoint=True)
    # Right edge: from (xmax, 0) to (xmax, ymax)
    right = np.linspace([xmax, spacing], [xmax, ymax], num=num_points_right, endpoint=False)
    # Top edge: from (xmax, ymax) to (0, ymax)
    top = np.linspace([xmax, ymax], [0, ymax], num=num_points_top, endpoint=True)
    # Left edge: from (0, ymax) to (0, 0)
    left = np.linspace([0, spacing], [0, ymax], num=num_points_left, endpoint=False)

    box_points = np.vstack((bottom, right, top, left))
    idxs = ["Wall" for _ in range(len(bottom))] + ["Right" for _ in range(len(right))] + ["Wall" for _ in range(len(top))] + ["Left" for _ in range(len(left))]
    return box_points, idxs


def gen_mesh_fvm(areas, cell_lnscale=2):
    xmin, xmax = -0.25, 8
    ymin, ymax = -1.75, 1.25

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
                #Ellipse((1, 1.5), 1.5, 0.6, 0, triscale, lims=lims, hole=True, dist_req=True, name="NavierWall"),
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
