import numpy as np
import matplotlib.pyplot as plt


def plot_nozzle_mesh(Rt=1.0, Re=1.5, theta_n_deg=20, theta_exit_deg=10, lengthscale=0.1):
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

    # Plot the mesh points along the top half of the nozzle
    plt.figure(figsize=(8, 4))
    plt.plot(mesh_x, mesh_y, 'bo-', markersize=3, label='Nozzle Mesh (Top Half)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Roe Rocket Nozzle Mesh (Top Half)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return the mesh points if further processing is needed:
    return mesh_x, mesh_y


# Example usage:
if __name__ == '__main__':
    Rt = 1.0  # Throat radius.
    n = 4  # Desired expansion ratio.
    Re = Rt * np.sqrt(n)  # Set exit radius based on expansion ratio.
    theta_n_deg = 33  # End angle for arc2.
    theta_exit_deg = 10  # Exit angle at the parabolic outlet.
    lengthscale = 0.1  # Desired spacing between mesh points.

    mesh_x, mesh_y = plot_nozzle_mesh(Rt, Re, theta_n_deg, theta_exit_deg, lengthscale)
