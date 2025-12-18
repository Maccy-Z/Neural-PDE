from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon, box
from shapely import affinity


@dataclass
class MeshProps:
    min_area: float
    max_area: float
    lengthscale: float


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


def gen_rand_ellipses(
        n_ellipses,
        domain_size,
        min_major,
        max_major,
        min_ecc,
        max_ecc,
        min_gap=0.0
):
    """
    Generates non-overlapping ellipses within a 2D box with a specified minimum gap.

    Returns:
        valid_ellipses (list of shapely objects): For plotting/intersection checks.
        parameters (list of dicts): The specific numeric parameters (center, e, angle).
    """

    width, height = domain_size
    domain_box = box(0, 0, width, height)

    valid_ellipses = []
    parameters = []

    # Safety counter to prevent infinite loops if the box gets too full
    attempts = 0
    max_attempts = n_ellipses * 1000

    while len(valid_ellipses) < n_ellipses and attempts < max_attempts:
        attempts += 1

        # 1. Randomize parameters
        # Semi-major axis (a)
        a = np.random.uniform(min_major, max_major)

        # Eccentricity (e)
        e = np.random.uniform(min_ecc, max_ecc)

        # Calculate Semi-minor axis (b) based on e = sqrt(1 - b^2/a^2)
        # Therefore b = a * sqrt(1 - e^2)
        b = a * np.sqrt(1 - e ** 2)

        # Angle (theta) in radians
        theta = np.random.uniform(0, np.pi)

        # Center (cx, cy)
        # We perform a rough margin check here so we don't spawn half-out-of-bounds
        cx = np.random.uniform(a, width - a)
        cy = np.random.uniform(a, height - a)

        # 2. Create Geometric Object (using Shapely)
        # Start with a unit circle
        ellipse_geo = Point(0, 0).buffer(1)

        # Scale to dimensions (a, b)
        ellipse_geo = affinity.scale(ellipse_geo, xfact=a, yfact=b)

        # Rotate
        ellipse_geo = affinity.rotate(ellipse_geo, theta)

        # Translate to position
        ellipse_geo = affinity.translate(ellipse_geo, xoff=cx, yoff=cy)

        # 3. Collision Detection

        # Check boundary containment (strict: must be fully inside)
        if not domain_box.contains(ellipse_geo):
            continue
        if ellipse_geo.distance(domain_box.boundary) < min_gap:
            continue

        # Check gap with existing ellipses
        # .distance() returns the minimum Euclidean distance between two geometries
        # If distance is 0, they touch or overlap.
        conflict = False
        for existing in valid_ellipses:
            if ellipse_geo.distance(existing) < min_gap:
                conflict = True
                break

        if not conflict:
            valid_ellipses.append(ellipse_geo)
            parameters.append({
                'center': (cx, cy),
                'semi_major': a,
                'eccentricity': e,
                'angle': theta
            })

    if attempts >= max_attempts:
        print(f"Warning: Could only place {len(valid_ellipses)} ellipses before timing out.")

    return valid_ellipses, parameters


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


