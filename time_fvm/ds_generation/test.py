import numpy as np
import pyvista as pv

# -------------------------------------------------------------------
# 1. Create a 2D irregular mesh with a hole
# -------------------------------------------------------------------

# Base mesh: an annulus (disc with inner radius)
base = pv.Disc(
    inner=0.3,   # hole radius
    outer=1.0,   # outer radius
    r_res=6,     # radial resolution
    c_res=90     # circumferential resolution
)

# Make it "irregular": jitter the points slightly in the XY-plane
rng = np.random.default_rng(42)
points = base.points.copy()
jitter = 0.03 * rng.standard_normal(points[:, :2].shape)
points[:, :2] += jitter
base.points = points

# CRITICAL FIX: triangulate so that all cells are triangles
mesh = base.triangulate().clean()  # clean() just to tidy up any degenerate cells

print("Original mesh (after triangulate):")
print("  is_all_triangles:", mesh.is_all_triangles)
print("  Points:", mesh.n_points)
print("  Cells: ", mesh.n_cells)

# -------------------------------------------------------------------
# 2. Decimate while preserving topology (keep the hole!)
# -------------------------------------------------------------------

target_reduction = 0.7  # remove 70% of triangles (keep 30%)

decimated = mesh.decimate_pro(
    reduction=target_reduction,
    preserve_topology=True,        # don't remove holes / change topology
    boundary_vertex_deletion=False # keep boundary vertices (hole boundary too)
)

print("\nDecimated mesh:")
print("  is_all_triangles:", decimated.is_all_triangles)
print("  Points:", decimated.n_points)
print("  Cells: ", decimated.n_cells)

# -------------------------------------------------------------------
# 3. Plot original vs decimated
# -------------------------------------------------------------------

plotter = pv.Plotter(shape=(1, 2), window_size=(900, 400))

# Left: original
plotter.subplot(0, 0)
plotter.add_text("Original mesh", font_size=12)
plotter.add_mesh(mesh, show_edges=True, color="white")
plotter.view_xy()
plotter.camera.Zoom(1.4)

# Right: decimated
plotter.subplot(0, 1)
plotter.add_text(f"Decimated (reduction = {target_reduction:.1f})", font_size=12)
plotter.add_mesh(decimated, show_edges=True, color="white")
plotter.view_xy()
plotter.camera.Zoom(1.4)

plotter.link_views()   # keep camera in sync
plotter.show()
