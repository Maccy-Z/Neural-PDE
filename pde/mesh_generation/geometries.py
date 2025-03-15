import numpy as np
from dataclasses import dataclass


class MeshFacet:
    points: np.ndarray
    segments: np.ndarray
    hole: bool | np.ndarray  # False if its to be filled, otherwise any point inside object
    dist_req: bool  # If segment needs mesh refinement around
    name: any # Tag to be carried through to the mesh

    real_face: bool = True # If segment is used for CFD mesh. Otherwise only for refining.


class Circle(MeshFacet):
    def __init__(self, center, radius, lengthscale, hole: bool = False, dist_req: bool = True, name = None, lims=None):
        """
        Generate points and segments for a circle boundary.
        :param center: Tuple (x, y) for the circle center.
        :param radius: Radius of the circle.
        :param lengthscale: Size of segments
        :param hole: Boolean indicating if the circle is a hole.
        :return: Arrays of points and segments defining the circle boundary.
        """
        self.name = name
        self.dist_req = dist_req

        num_segments = np.ceil(2 * np.pi * radius / lengthscale).astype(int)

        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        self.points = np.column_stack((
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles)
        ))

        if lims is not None:
            self.points = np.clip(self.points, lims[0], lims[1])

        self.segments = np.column_stack((np.arange(num_segments), (np.arange(num_segments) + 1) % num_segments))

        if hole:
            self.hole = center
        else:
            self.hole = False


class Ellipse(MeshFacet):
    def __init__(self, center, semi_major_axis, eccentricity, rotation_angle, lengthscale, lims=None,
                 hole: bool = False, dist_req: bool = True, name = None):
        """
        Generate points and segments for an ellipse boundary using eccentricity and rotation angle.
        :param center: Tuple (x, y) for the ellipse center.
        :param semi_major_axis: Length of the semi-major axis (along the x-axis before rotation).
        :param eccentricity: Eccentricity of the ellipse (0 <= eccentricity < 1).
        :param lengthscale: Size of segments
        :param rotation_angle: Angle in radians to rotate the ellipse (counterclockwise).
        :param hole: Boolean indicating if the ellipse is a hole.
        :return: Arrays of points and segments defining the ellipse boundary.
        """
        self.name = name
        self.dist_req = dist_req

        # Calculate the semi-minor axis using the eccentricity
        semi_minor_axis = semi_major_axis * np.sqrt(1 - eccentricity ** 2)

        perimeter = np.pi * (3 * (semi_major_axis + semi_minor_axis) - np.sqrt((3 * semi_major_axis + semi_minor_axis) * (semi_major_axis + 3 * semi_minor_axis)))
        num_segments = np.ceil(perimeter / lengthscale).astype(int)
        # Generate the angles for the points on the ellipse
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

        # Generate the points for the ellipse before rotation
        ellipse_points = np.column_stack((
            semi_major_axis * np.cos(angles),
            semi_minor_axis * np.sin(angles)
        ))

        # Rotation matrix for the specified angle
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])

        # Rotate the points
        rotated_points = ellipse_points @ rotation_matrix.T

        # Translate the points to the center
        self.points = rotated_points + np.array(center)
        if lims is not None:
            self.points = np.clip(self.points, lims[0], lims[1])

        # Generate the segments to connect the points
        self.segments = np.column_stack((np.arange(num_segments), (np.arange(num_segments) + 1) % num_segments))

        # print(f'{self.points.shape = }, {self.segments.shape = }')
        # print(self.points)
        # exit(7)
        # Handle the hole parameter
        if hole:
            self.hole = center
        else:
            self.hole = False


class Box(MeshFacet):
    def __init__(self, Xmin, Xmax, remove_edge: int = None, hole: bool = False, dist_req: bool = False, name = None):
        """ Remove edge: 0: Left
                         1: Top
                         2: Right
                         3: Bottom
        """
        self.name = name
        self.hole = hole
        self.dist_req = dist_req

        xmin, ymin = Xmin
        xmax, ymax = Xmax

        # Define the corner points of the rectangle (box)
        self.points = np.array([
            (xmin, ymin),  # Point 0
            (xmax, ymin),  # Point 1
            (xmax, ymax),  # Point 2
            (xmin, ymax),  # Point 3
        ])

        # Define the segments (edges) of the rectangle
        self.segments = np.array([
            (3, 0),  # Left edge from Point 3 to Point 0
            (2, 3),  # Top edge from Point 0 to Point 1
            (1, 2),  # Right edge from Point 1 to Point 2
            (1, 0),  # Bottom edge from Point 2 to Point 3
        ])

        if remove_edge is not None:
            self.segments = np.delete(self.segments, remove_edge, axis=0)


class Line(MeshFacet):
    def __init__(self, lims, dist_req: bool = False, name = None, real=True):
        self.name = name
        self.hole = False
        self.dist_req = dist_req
        self.real_face = real

        self.points = np.array(lims)
        self.segments = np.array([(0, 1)])
