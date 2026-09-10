"""
AeroTwin — Script 03: 3D Point Cloud Desktop Viewer (PyVista GUI)
"""
import sys
from pathlib import Path
import pyvista as pv
import numpy as np

ply_display = Path("data/output/sample_flight/dense_display.ply")
ply_dense = Path("data/output/sample_flight/dense_pointcloud.ply")

target_ply = ply_display if ply_display.exists() else ply_dense
print(f"Loading {target_ply}...")

pcd = pv.read(str(target_ply))
pts = np.array(pcd.points)

pl = pv.Plotter(window_size=[1400, 850], title="AeroTwin — 3D Digital Twin Viewer")
pl.set_background("#07090e")

if "RGB" in pcd.point_data:
    pl.add_points(pcd, scalars="RGB", rgb=True, point_size=3, render_points_as_spheres=True)
else:
    pl.add_points(pcd, color="#38bdf8", point_size=3, render_points_as_spheres=True)

# Add title banner
pl.add_text("AeroTwin (SIH 2026) - 13.3 Million Point Digital Twin", font_size=12, color="white", position="upper_left")
pl.show()
