import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, box
from shapely import affinity


def generate_random_ellipses(
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
                'center_x': cx,
                'center_y': cy,
                'semi_major_a': a,
                'semi_minor_b': b,
                'eccentricity': e,
                'angle_deg': theta
            })

    if attempts >= max_attempts:
        print(f"Warning: Could only place {len(valid_ellipses)} ellipses before timing out.")

    return valid_ellipses, parameters


# --- Configuration ---
DOMAIN = (2, 1.4)  # Width, Height
NUM_ELLIPSES = 10
MIN_MAJOR_AXIS = 0.1
MAX_MAJOR_AXIS = 0.2
MIN_ECCENTRICITY = 0.1  # 0 is a circle, close to 1 is a needle
MAX_ECCENTRICITY = 0.95
MIN_GAP = 0.2  # Minimum distance between any two ellipses

# --- Execution ---
shapes, params = generate_random_ellipses(
    NUM_ELLIPSES,
    DOMAIN,
    MIN_MAJOR_AXIS,
    MAX_MAJOR_AXIS,
    MIN_ECCENTRICITY,
    MAX_ECCENTRICITY,
    MIN_GAP
)
print(f'{shapes = }')
print(f'{params = }')
exit(7)
# --- Output Data ---
print(f"Successfully generated {len(params)} ellipses.\n")
print(f"{'ID':<5} {'CenterX':<10} {'CenterY':<10} {'Eccen (e)':<12} {'Angle':<10}")
print("-" * 50)
for i, p in enumerate(params):
    print(f"{i:<5} {p['center_x']:<10.2f} {p['center_y']:<10.2f} {p['eccentricity']:<12.3f} {p['angle_deg']:<10.1f}")

# --- Visualization ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, DOMAIN[0])
ax.set_ylim(0, DOMAIN[1])
ax.set_aspect('equal')
ax.set_title(f"Random Generation of {len(params)} Non-Overlapping Ellipses")

# Draw Boundary
rect = plt.Rectangle((0, 0), DOMAIN[0], DOMAIN[1], linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(rect)

# Draw Ellipses
for poly in shapes:
    # Shapely polygons can be plotted using the exterior coordinates
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='teal', ec='black')

plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()