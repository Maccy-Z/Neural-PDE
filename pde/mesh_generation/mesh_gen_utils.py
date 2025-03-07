from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


@dataclass
class MeshProps:
    min_area: float
    max_area: float
    lengthscale: float


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

    int_edges = extract_interor_edges(triangles)
    return (points, triangles), (p_markers, f_markers), (int_edges, bound_edges)


def plot_mesh(points, p_markers):
    # Plot the points
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(p_markers), vmax=max(p_markers))
    plt.scatter(
        points[:, 0],  # X coordinates
        points[:, 1],  # Y coordinates
        c=p_markers,  # Color mapping based on p_markers
        cmap=cmap,  # Colormap
        norm=norm,  # Normalization
        marker='o',  # Marker style
        s=15  # Marker size
    )
    # Plot bounding facets (edges)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('2D CFD Mesh')
    # plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_edges(coords, edge_idx, title=""):
    """ Plot the edges of the mesh.
        coords.shape = (n, 2)
        edge_idx.shape = (m, 2)
    """
    points = coords[edge_idx]   # shape = (m, 2, 2)
    for edge in points:
        plt.plot(edge[:, 0], edge[:, 1], 'k-')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.title(title)
    plt.show()


