"""
scripts/07_metric_measurement.py
AeroTwin — Feature #5: Metric Measurement Tool
Computes metric 3D point-to-point Euclidean distances, vertical structure heights,
surface areas, and bounding volumes from reconstructed coordinates.
"""

import sys
import numpy as np
import pyvista as pv
from pathlib import Path
import json

def measure_distance(p1, p2):
    """Calculates Euclidean distance in real-world metric units."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = np.linalg.norm(p1 - p2)
    dx, dy, dz = np.abs(p1 - p2)
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    vertical_height = dz
    return {
        "3d_distance_m": round(float(dist), 3),
        "horizontal_distance_m": round(float(horizontal_dist), 3),
        "vertical_height_m": round(float(vertical_height), 3)
    }

def measure_bounding_dimensions(pts):
    """Calculates width, length, height and footprint area of a selected cluster (e.g. building/road segment)."""
    pts = np.array(pts)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    dims = maxs - mins
    footprint_area = dims[0] * dims[1]
    return {
        "width_m": round(float(dims[0]), 3),
        "length_m": round(float(dims[1]), 3),
        "height_m": round(float(dims[2]), 3),
        "footprint_area_sq_m": round(float(footprint_area), 2)
    }

if __name__ == "__main__":
    print("AeroTwin Metric Measurement Engine Initialized.")
    # Example unit test
    ptA = [10.5, 20.0, 5.2]
    ptB = [18.2, 34.6, 12.8]
    res = measure_distance(ptA, ptB)
    print("Sample Measurement Result:", json.dumps(res, indent=2))
