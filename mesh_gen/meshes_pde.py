import numpy as np

from mesh_gen.mesh_gen_utils import MeshProps, extract_mesh_data, gen_rand_ellipses
from mesh_gen.geometries import MeshFacet, Circle, Line, Ellipse
from mesh_gen.create_mesh import create_mesh


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
    area_lnscale = 0.4
    xmin, xmax = 0, 2
    ymin, ymax = 0.0, 1.5

    lengthscale = np.sqrt(2*min_area)

    mesh_props = MeshProps(min_area, max_area, lengthscale=area_lnscale)

    coords = [
              Line([[xmin, ymin], [xmax, ymin]], dist_req=True, name="Navier_wall"),
              Line([[xmin, ymax], [xmax, ymax]], True, name="Navier_wall"),
              Line([[xmin, ymin], [xmin, ymax]], True, name="wall_left"),
              Line([[xmax, ymax], [xmax, ymin]], True, name="wall_right"),
              ]

    _, rand_ellipses = gen_rand_ellipses(3, (xmax-xmin, ymax-ymin),
                                         min_major=0.15, max_major=0.2, min_ecc=0.1, max_ecc=0.8, min_gap=0.1)

    for spec in rand_ellipses:
        e = Ellipse(center=spec['center'], semi_major_axis=spec['semi_major'], eccentricity=spec['eccentricity'], angle=spec['angle'],
                      lengthscale=lengthscale, hole=True, dist_req=True, name="Navier_wall")
        coords.append(e)

    (point_props, markers, _edges), marker_tags = create_mesh(coords, mesh_props)
    points, triangles = point_props
    p_markers, _ = markers
    _, bound_edges = _edges

    p_tags = [marker_tags[int(i)] for i in p_markers]
    return points, triangles, (None, bound_edges), p_tags


def main():
    np.random.seed(0)

    try:
        points, triangles, (int_edges, bound_edges), f_tag = gen_mesh_random()
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
