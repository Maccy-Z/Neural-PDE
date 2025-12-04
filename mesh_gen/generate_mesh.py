import numpy as np
from cprint import c_print
import meshpy.triangle as tri
import threading

from mesh_gen.mesh_gen_utils import MeshProps, min_dist_to_boundary, extract_mesh_data, gen_rand_ellipses
from mesh_gen.geometries import MeshFacet, Circle, Line, Ellipse, Nozzle

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
    max_area = 2e-3
    xmin, xmax = 0, 2
    ymin, ymax = 0.0, 1.5

    lengthscale = np.sqrt(2*min_area)

    mesh_props = MeshProps(min_area, max_area, lengthscale=0.4)

    coords = [
              Line([[xmin, ymin], [xmax, ymin]], dist_req=True, name="wall_bottom"),
              Line([[xmin, ymax], [xmax, ymax]], True, name="wall_top"),
              Line([[xmin, ymin], [xmin, ymax]], True, name="wall_left"),
              Line([[xmax, ymax], [xmax, ymin]], True, name="wall_right"),

              Ellipse(center=(0.7, 0.75), semi_major_axis=0.2, eccentricity=0.75, angle=np.pi/3,
                      lengthscale=lengthscale, hole=True, dist_req=True, name="circle"),
              ]

    mesh, marker_tags = create_mesh(coords, mesh_props)
    point_props, markers, _edges = extract_mesh_data(mesh)
    points, triangles = point_props
    p_markers, _ = markers
    int_edges, bound_edges = _edges

    p_tags = [marker_tags[int(i)] for i in p_markers]
    return points, triangles, (int_edges, bound_edges), p_tags

def gen_mesh_random():
    min_area = 0.5e-3
    max_area = 2e-3
    xmin, xmax = 0, 2
    ymin, ymax = 0.0, 1.5

    lengthscale = np.sqrt(2*min_area)

    mesh_props = MeshProps(min_area, max_area, lengthscale=0.4)

    coords = [
              Line([[xmin, ymin], [xmax, ymin]], dist_req=True, name="wall_bottom"),
              Line([[xmin, ymax], [xmax, ymax]], True, name="wall_top"),
              Line([[xmin, ymin], [xmin, ymax]], True, name="wall_left"),
              Line([[xmax, ymax], [xmax, ymin]], True, name="wall_right"),

              # Ellipse(center=(0.7, 0.75), semi_major_axis=0.2, eccentricity=0.75, angle=np.pi/3,
              #         lengthscale=lengthscale, hole=True, dist_req=True, name="circle"),
              ]

    _, rand_ellipses = gen_rand_ellipses(3, (xmax-xmin, ymax-ymin), 0.1, 0.2, 0.5, 0.9, min_gap=0.05)

    for spec in rand_ellipses:
        e = Ellipse(center=spec['center'], semi_major_axis=spec['semi_major'], eccentricity=spec['eccentricity'], angle=spec['angle'],
                      lengthscale=lengthscale, hole=True, dist_req=True, name="circle")
        coords.append(e)


    mesh, marker_tags = create_mesh(coords, mesh_props)
    point_props, markers, _edges = extract_mesh_data(mesh)
    points, triangles = point_props
    p_markers, _ = markers
    int_edges, bound_edges = _edges

    p_tags = [marker_tags[int(i)] for i in p_markers]
    return points, triangles, (int_edges, bound_edges), p_tags


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
