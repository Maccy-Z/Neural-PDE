import numpy as np
import multiprocessing as mp
import queue
from cprint import c_print
import logging

def min_dist_to_boundary(point, seg_points, segment_indices):
    """
    Calculate the minimum distance from a point to a list of segments defined by indices.
    :param point: The point (x, y) as a 1D NumPy array.
    :param seg_points: A 2D NumPy array of shape (n, 2) representing all points.
    :param segment_indices: A 2D NumPy array of shape (m, 2), each row containing two indices
                            into the `points` array, representing the start and end of a segment.
    :return: The minimum distance from the point to the segments.
    """
    # Extract segment start and end points from the points array
    segment_starts = seg_points[segment_indices[:, 0]]
    segment_ends = seg_points[segment_indices[:, 1]]

    # Vector from start to end of each segment
    segment_vectors = segment_ends - segment_starts
    # Vector from start of each segment to the point
    point_vectors = point - segment_starts

    # Project point_vectors onto segment_vectors
    projection_lengths = np.einsum('ij,ij->i', point_vectors, segment_vectors) / (np.einsum('ij,ij->i', segment_vectors, segment_vectors)+1e-8)
    projection_lengths = np.clip(projection_lengths, 0, 1)

    # Closest points on each segment to the point
    closest_points = segment_starts + (projection_lengths[:, np.newaxis] * segment_vectors)

    # Distances from the point to each closest point on the segments
    distances = np.linalg.norm(point - closest_points, axis=1)

    return np.min(distances)

def extract_interor_edges(triangles):
    """
    Extract all edges from the elements of the mesh.
    """
    # Extract all edges from the elements
    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])

    # Sort the indices within each edge to ensure consistency
    all_edges = np.sort(edges, axis=1)
    unique_all_edges, all_counts = np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)
    interior_edges = unique_all_edges[all_counts == 2]
    return interior_edges

def extract_mesh_data(mesh):
    """ Extract the mesh data from the mesh object. """
    points, triangles, bound_edges = mesh.points, mesh.elements, mesh.facets
    p_markers, f_markers = mesh.point_markers, mesh.facet_markers

    points, triangles, bound_edges = np.array(points), np.array(triangles), np.array(bound_edges)
    p_markers, f_markers = np.array(p_markers), np.array(f_markers)

    assert len(triangles) != 0, "No triangles found in the mesh."

    int_edges = extract_interor_edges(triangles)
    return (points, triangles), (p_markers, f_markers), (int_edges, bound_edges)

def _create_mesh_thread(holes, points, p_marks, segments, seg_marks, mesh_props, dist_p, dist_seg, min_angle, out_queue):
    from meshpy import triangle as tri

    # Custom function to control mesh refinement
    def refine_fn(vertices, area, props, points, segments):
        """ Return True if area is too big, False if area is small enough """
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

    mesh_info = tri.MeshInfo()
    mesh_info.set_holes(holes)
    mesh_info.set_points(points, point_markers=p_marks)
    mesh_info.set_facets(segments, facet_markers=seg_marks)

    mesh = tri.build(mesh_info, refinement_func=lambda x, y: refine_fn(x, y, mesh_props, dist_p, dist_seg), min_angle=min_angle)
    mesh_specs = extract_mesh_data(mesh)
    out_queue.put(("ok", mesh_specs))


def safe_run(args):
    """ Run mesh generation in a separate process to catch crashes/hangs, and repeat if it happens. """
    while True:
        ctx = mp.get_context("spawn")  # safer / more predictable on many platforms
        out_queue = ctx.Queue()
        p = ctx.Process(target=_create_mesh_thread, args=tuple(args + [out_queue]))
        p.start()
        try:
            # Wait up to 20 seconds for a message from the child
            status, payload = out_queue.get(timeout=10)
        except queue.Empty:
            # Child didn't send anything in time (hung or crashed)
            if p.is_alive():
                p.terminate()
            p.join()
            logging.warning("Mesh generation process timed out.")
            continue
        if status == "error":
            logging.warning("Exception in mesh generation process:\n" + payload[1])
            continue
        else:
            mesh_specs = payload
            p.terminate()
            p.join()
            break

    return mesh_specs


def create_mesh(coords: list, mesh_props, min_angle=None):
    # Collate together all facet objects
    points, segments = np.zeros((0, 2)), np.zeros((0, 2), dtype=int)
    # Segments for dist calculation
    dist_p, dist_seg = np.zeros((0, 2)), np.zeros((0, 2), dtype=int)

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

    mesh_specs = safe_run([holes, points, p_marks, segments, seg_marks, mesh_props, dist_p, dist_seg, min_angle])
    return mesh_specs, marker_names
