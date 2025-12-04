import numpy as np
from dataclasses import dataclass

@dataclass
class MeshFacet:
    points: np.ndarray
    segments: np.ndarray
    hole: bool | list  # False if its to be filled, otherwise any point inside object
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
            self.hole = [center]
        else:
            self.hole = False


class Ellipse(MeshFacet):
    def __init__(self, center, semi_major_axis, eccentricity, angle, lengthscale, lims=None,
                 hole: bool = False, dist_req: bool = True, name = None):
        """
        Generate points and segments for an ellipse boundary using eccentricity and rotation angle.
        :param center: Tuple (x, y) for the ellipse center.
        :param semi_major_axis: Length of the semi-major axis (along the x-axis before rotation).
        :param eccentricity: Eccentricity of the ellipse (0 <= eccentricity < 1).
        :param lengthscale: Size of segments
        :param angle: Angle in radians to rotate the ellipse (counterclockwise).
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
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # Rotate the points
        rotated_points = ellipse_points @ rotation_matrix.T

        # Translate the points to the center
        self.points = rotated_points + np.array(center)
        if lims is not None:
            self.points = np.clip(self.points, lims[0], lims[1])

        # Generate the segments to connect the points
        self.segments = np.column_stack((np.arange(num_segments), (np.arange(num_segments) + 1) % num_segments))

        # Handle the hole parameter
        if hole:
            self.hole = [center]
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
        if hole:
            center = np.mean([Xmin, Xmax], axis=0)
            self.hole = [center]
        else:
            self.hole = False
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


class Nozzle(MeshFacet):
    def __init__(self, Xmin, Rt=1.0, Re=2, theta_n_deg=35, theta_exit_deg=15, lengthscale=0.1, lip_size=0.1, dist_req: bool = True, name=None):
        """
            Generates a mesh (a set of points) for the top half of a Roe rocket nozzle
            with an approximate spacing given by "lengthscale".

            The nozzle is defined by:
              1. Arc 1: A circular arc (radius = 1.5·Rt) from -135° to -90°.
              2. Arc 2: A circular arc (radius = 0.382·Rt) from -90° to (theta_n_deg - 90)°.
                 The end of this arc is the inflection point N.
              3. Parabolic Section: A parabola attached at point N with a slope of tan(theta_n_deg)
                 that meets the exit condition (y = Re) with the derivative equal to tan(theta_exit_deg).
                 The horizontal length L_nozzle of the parabolic part is derived from these conditions.

            Parameters:
              Rt             : Throat radius.
              Re             : Exit radius.
              theta_n_deg    : Angle (in degrees) that sets the end of arc2 (from -90° to theta_n_deg - 90°).
              theta_exit_deg : Desired exit angle (in degrees) for the parabolic outlet (i.e. its tangent at the exit).
              lengthscale    : Approximate distance between consecutive mesh points along the nozzle.
            """
        self.name = name
        self.dist_req = dist_req

        # ===== Arc 1: From -135° to -90° =====
        R1 = 1.5 * Rt
        center1 = np.array([0, R1])  # chosen so that at -90° the point is at (0,0)
        angle1_start = np.deg2rad(-135)
        angle1_end = np.deg2rad(-90)
        arc1_angle_diff = angle1_end - angle1_start  # should be 45° in radians (0.7854)
        arc1_length = R1 * abs(arc1_angle_diff)
        num_points1 = max(int(np.ceil(arc1_length / lengthscale)) + 1, 2)
        arc1_angles = np.linspace(angle1_start, angle1_end, num_points1)
        arc1_x = center1[0] + R1 * np.cos(arc1_angles)
        arc1_y = center1[1] + R1 * np.sin(arc1_angles)

        # ===== Arc 2: From -90° to (theta_n_deg - 90)° =====
        R2 = 0.382 * Rt
        center2 = np.array([0, R2])  # chosen so that at -90° the point is (0,0)
        angle2_start = np.deg2rad(-90)
        angle2_end = np.deg2rad(theta_n_deg - 90)
        arc2_angle_diff = angle2_end - angle2_start  # equals theta_n_deg in radians
        arc2_length = R2 * abs(arc2_angle_diff)
        num_points2 = max(int(np.ceil(arc2_length / lengthscale)) + 1, 2)
        arc2_angles = np.linspace(angle2_start, angle2_end, num_points2)
        arc2_x = center2[0] + R2 * np.cos(arc2_angles)
        arc2_y = center2[1] + R2 * np.sin(arc2_angles)

        # ===== Inflection Point N =====
        # End of Arc 2 (point at angle2_end)
        tN = angle2_end
        N_x = center2[0] + R2 * np.cos(tN)
        N_y = center2[1] + R2 * np.sin(tN)

        # ===== Parabolic Section =====
        # The parabola is defined as:
        #   y(x) = A * (x - N_x)^2 + m_N*(x - N_x) + N_y,
        # where m_N = tan(theta_n_deg) is the slope at N.
        # To have the exit condition at x = N_x + L_nozzle:
        #   y(N_x+L_nozzle) = Re,
        #   y'(N_x+L_nozzle) = m_exit = tan(theta_exit_deg).
        # Solving the derivative condition:
        m_N = np.tan(np.deg2rad(theta_n_deg))
        m_exit = np.tan(np.deg2rad(theta_exit_deg))
        # The horizontal length of the parabola is determined by:
        #   L_nozzle = 2*(Re - N_y)/(m_exit + m_N)
        L_nozzle = 2 * (Re - N_y) / (m_exit + m_N)
        # Then the quadratic coefficient A is:
        A = (m_exit - m_N) / (2 * L_nozzle)

        # To sample the parabolic section with roughly "lengthscale" spacing,
        # we first generate a dense set of points and then re-parameterize by arc length.
        dense_points = 1000
        x_dense = np.linspace(N_x, N_x + L_nozzle, dense_points)
        y_dense = A * (x_dense - N_x) ** 2 + m_N * (x_dense - N_x) + N_y
        # Compute differential arc lengths:
        dx_dense = np.diff(x_dense)
        dy_dense = np.diff(y_dense)
        ds_dense = np.sqrt(dx_dense ** 2 + dy_dense ** 2)
        s_dense = np.concatenate(([0], np.cumsum(ds_dense)))
        total_parabola_length = s_dense[-1]
        # Determine the number of points so that spacing is roughly "lengthscale"
        num_points3 = max(int(np.ceil(total_parabola_length / lengthscale)) + 1, 2)
        # Create a uniform spacing in arc length for the parabolic section:
        s_desired = np.linspace(0, total_parabola_length, num_points3)
        # Interpolate to obtain (x,y) corresponding to these arc-length positions.
        x_parabola = np.interp(s_desired, s_dense, x_dense)
        y_parabola = np.interp(s_desired, s_dense, y_dense)

        # ===== Assemble the Nozzle Mesh (Top Half Only) =====
        # Remove duplicate points at the boundaries (the throat and point N).
        mesh_x = np.concatenate((arc1_x, arc2_x[1:], x_parabola[1:]))
        mesh_y = np.concatenate((arc1_y, arc2_y[1:], y_parabola[1:]))
        mesh_X = np.column_stack((mesh_x, mesh_y))


        # mesh_X = np.concatenate([mesh_X, np.array([[0., Re/4]])], axis=0)
        # Add on boundary points
        min_x, max_x = mesh_X[:, 0].min(), mesh_X[:, 0].max()
        max_y = mesh_X[:, 1].max()
        start_y = mesh_X[0, 1]
        back = np.array([[min_x-lip_size, max_y + lip_size]])
        front = np.array([[max_x, max_y + lip_size]])
        inlet = np.array([[min_x-lip_size, start_y]])
        mesh_X = np.concatenate([mesh_X, front, back, inlet], axis=0)

        # Segment indices
        n_points_up = len(mesh_X)
        segments = np.column_stack((np.arange(n_points_up), (np.arange(n_points_up) + 1) % n_points_up))[:]
        segments = np.delete(segments, [-2, -3], axis=0)

        # Lower part of the nozzle
        mesh_X_low = np.column_stack((mesh_X[:, 0], -mesh_X[:, 1] - Rt))

        # Center first point at (0,0)
        points_all = np.concatenate([mesh_X, mesh_X_low])
        self.points =  points_all + np.array(Xmin) - mesh_X[0]

        # Add on hole point
        hole = np.array([0, Re/4])
        hole_upper = hole - mesh_X[0]
        hole_lower = np.array([0, -Rt-Re/4]) - mesh_X[0]
        self.hole = [hole_upper.tolist(), hole_lower.tolist()]
        # print(f'{self.hole = }')

        self.segments = np.concatenate([segments, segments + n_points_up], axis=0)

        # self.points = mesh_X
        # self.segments = segments
        # print(f'{self.segments = }')
        # print(f'{self.points.shape = }')


def main():
    from matplotlib import pyplot as plt
    nozzle = Nozzle([0, 0], Rt=0.25, Re=1, theta_n_deg=30, theta_exit_deg=15, lip_size=0.5)
    points = nozzle.points

    print(nozzle.segments)

    for start_idx, end_idx in nozzle.segments:
        x_vals = [points[start_idx][0], points[end_idx][0]]
        y_vals = [points[start_idx][1], points[end_idx][1]]
        plt.plot(x_vals, y_vals, 'b-')  # 'b-' means blue line

    plt.scatter(*points.T)
    plt.show()


if __name__ == "__main__":
    main()