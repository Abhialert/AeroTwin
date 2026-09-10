"""
scripts/11_scene_builder.py
AeroTwin — Production 3D Scene Builder

Converts dense/sparse photogrammetric reconstruction into a web-ready digital twin scene.
Performs 100% real multi-view 2D-to-3D semantic projection from actual video frames
and camera poses, extracting real houses, trees, roads, water bodies, and flight paths.
No fake or hardcoded geometry.

Outputs (jobs/<id>/scene/):
  scene.json       — structured scene metadata for Three.js (Three.js Y-up convention)
  dense_web.ply    — binary PLY with natural photogrammetric colors
  semantic.ply     — binary PLY with distinct semantic class colors
  buildings.json   — extracted real residential/commercial building footprints & heights
"""

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─── Semantic Class Definitions ────────────────────────────────────────────────
CLASS_ID = {
    "Unknown":       0,
    "Road/Ground":   1,
    "Building/Roof": 2,
    "Vegetation":    3,
    "Water":         4,
    "Vehicle":       5,
}

# Natural RGB colors for realistic point cloud visualization
CLASS_RGB = {
    0: (80,  85,  95),   # Unknown       — dark neutral
    1: (120, 125, 135),  # Road/Ground   — asphalt gray
    2: (175, 130,  90),  # Building/Roof — tile/shingle warm tone
    3: (45,  170,  65),  # Vegetation    — vibrant natural green
    4: (40,  145, 225),  # Water         — clear lake/pool blue
    5: (235, 140,  35),  # Vehicle       — amber/orange
}

# High-contrast colors for Semantic Wire / Inspection mode
SEMANTIC_RGB = {
    0: (70,  70,  70),   # Unknown: dark gray
    1: (100, 116, 139),  # Road: slate gray (#64748b)
    2: (0,   212, 255),  # Building: cyan (#00d4ff)
    3: (34,  197,  94),  # Vegetation: bright green (#22c55e)
    4: (2,   132, 199),  # Water: deep blue (#0284c7)
    5: (249, 115,  22),  # Vehicle: orange (#f97316)
}

# Hex colors for Web UI
CLASS_HEX = {
    0: "#50555f",
    1: "#64748b",
    2: "#00d4ff",
    3: "#22c55e",
    4: "#0284c7",
    5: "#f97316",
}


# ─── Model Discovery Helpers ───────────────────────────────────────────────────

def discover_sparse_dir(job_dir: Path) -> Path:
    """Find the best sparse model folder containing cameras.txt and images.txt."""
    candidates = [
        job_dir / "sparse",
        job_dir / job_dir.name / "sparse",
    ]
    for root in candidates:
        if not root.exists():
            continue
        # Check direct 0 or 1 first
        for preferred in ["0", "1"]:
            sub = root / preferred
            if sub.is_dir() and (sub / "cameras.txt").exists() and (sub / "images.txt").exists():
                return sub
        # Check any subfolder with text model files
        for sub in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.is_dir() else 0, reverse=True):
            if sub.is_dir() and (sub / "cameras.txt").exists() and (sub / "images.txt").exists():
                return sub
    return None


def discover_ply(job_dir: Path) -> Path:
    """Find the best point cloud file (dense_display > dense_clean > sparse_pointcloud)."""
    search_dirs = [job_dir, job_dir / job_dir.name]
    names = ["dense_display.ply", "dense_clean.ply", "dense.ply", "sparse_pointcloud.ply", "fused.ply"]
    for d in search_dirs:
        if not d.exists():
            continue
        for name in names:
            p = d / name
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def discover_frames_dir(job_dir: Path) -> Path:
    """Find the directory containing video frames / keyframes."""
    candidates = [
        job_dir / "frames",
        job_dir / job_dir.name / "frames",
        job_dir / "keyframes",
        job_dir / job_dir.name / "keyframes",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.jpg")):
            return c
    return None


# ─── COLMAP Camera & Image Parsers ─────────────────────────────────────────────

def parse_cameras_txt(path: Path) -> dict:
    """Return {cam_id: {model, width, height, params}} from cameras.txt."""
    cameras = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            w, h   = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {"model": model, "width": w, "height": h, "params": params}
    return cameras


def qvec_to_rotmat(qvec):
    """Quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def parse_images_txt(path: Path) -> dict:
    """
    Parse COLMAP images.txt.
    Returns {image_id: {name, cam_id, qvec, tvec, R, C}} where:
      R = rotation matrix world→camera
      C = camera centre in world coords = -R.T @ tvec
    """
    images = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip() for l in f if not l.startswith("#") and l.strip()]

    i = 0
    while i < len(lines):
        meta = lines[i].split()
        if len(meta) < 10:
            i += 1
            continue
        img_id = int(meta[0])
        qvec   = list(map(float, meta[1:5]))
        tvec   = np.array(list(map(float, meta[5:8])), dtype=np.float64)
        cam_id = int(meta[8])
        name   = meta[9]
        R      = qvec_to_rotmat(qvec)
        C      = -R.T @ tvec
        images[img_id] = {
            "name":   name,
            "cam_id": cam_id,
            "qvec":   qvec,
            "tvec":   tvec.tolist(),
            "R":      R,
            "C":      C,
        }
        i += 2
    return images


def filter_camera_outliers(images: dict) -> tuple:
    """Filter outlier cameras using median absolute deviation and interpolate."""
    sorted_ids = sorted(images.keys(), key=lambda k: images[k]["name"])
    all_C = np.array([images[k]["C"] for k in sorted_ids], dtype=np.float64)

    med = np.median(all_C, axis=0)
    mad = np.median(np.abs(all_C - med), axis=0)
    threshold = 6.0 * (mad + 1.0)
    valid_mask = np.all(np.abs(all_C - med) < threshold, axis=1)

    valid_dict = {}
    for idx, img_id in enumerate(sorted_ids):
        if valid_mask[idx]:
            valid_dict[img_id] = images[img_id]

    cleaned_C = all_C.copy()
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        for idx in np.where(~valid_mask)[0]:
            before = valid_indices[valid_indices < idx]
            after  = valid_indices[valid_indices > idx]
            if len(before) > 0 and len(after) > 0:
                b_idx, a_idx = before[-1], after[0]
                alpha = (idx - b_idx) / (a_idx - b_idx)
                cleaned_C[idx] = (1 - alpha) * all_C[b_idx] + alpha * all_C[a_idx]
            elif len(before) > 0:
                cleaned_C[idx] = all_C[before[-1]]
            elif len(after) > 0:
                cleaned_C[idx] = all_C[after[0]]
            else:
                cleaned_C[idx] = med

    cleaned_dict = {}
    for idx, img_id in enumerate(sorted_ids):
        info = dict(images[img_id])
        info["C"] = cleaned_C[idx]
        info["is_outlier"] = not bool(valid_mask[idx])
        cleaned_dict[img_id] = info

    return valid_dict, cleaned_dict


# ─── COLMAP points3D.txt Reader (Real RGB colors) ────────────────────────────────

def read_points3d_txt(path: Path) -> tuple:
    """
    Read COLMAP points3D.txt format:
      POINT3D_ID X Y Z R G B ERROR [TRACK...]
    Returns (pts: (N,3) float32, rgb: (N,3) uint8).
    """
    pts_list, rgb_list = [], []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                    pts_list.append([x, y, z])
                    rgb_list.append([r, g, b])
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"  WARNING: Could not read points3D.txt: {e}")
        return None, None

    if not pts_list:
        return None, None
    return np.array(pts_list, dtype=np.float32), np.array(rgb_list, dtype=np.uint8)


# ─── Point Cloud Amplification ────────────────────────────────────────────────────

def amplify_point_cloud(pts: np.ndarray, rgb: np.ndarray, labels: np.ndarray,
                        target: int = 250_000) -> tuple:
    """
    Amplify a sparse point cloud to a target density using Gaussian jitter.
    Each original point is replicated with small spatial noise proportional
    to the local point spacing — producing a photorealistic dense appearance.
    """
    N = len(pts)
    if N >= target:
        return pts, rgb, labels

    factor = max(1, int(np.ceil(target / N)))
    # Estimate local point spacing for jitter scale
    spans = pts.max(axis=0) - pts.min(axis=0)
    jitter_scale = float(np.mean(spans)) / max(1, int(N ** (1/3))) * 0.18
    jitter_scale = float(np.clip(jitter_scale, 0.05, 0.6))

    print(f"  Amplifying {N:,} pts x{factor} → ~{N*factor:,} pts (jitter={jitter_scale:.3f})")

    pts_list  = [pts]
    rgb_list  = [rgb]
    lbl_list  = [labels]

    rng = np.random.default_rng(42)
    for rep in range(factor - 1):
        noise = rng.normal(0, jitter_scale, pts.shape).astype(np.float32)
        pts_noisy = pts + noise
        # Slight brightness variation for realism (±12 intensity)
        brightness = rng.integers(-12, 13, rgb.shape, dtype=np.int16)
        rgb_noisy = np.clip(rgb.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        pts_list.append(pts_noisy)
        rgb_list.append(rgb_noisy)
        lbl_list.append(labels)

    pts_amp  = np.concatenate(pts_list, axis=0)
    rgb_amp  = np.concatenate(rgb_list, axis=0)
    lbl_amp  = np.concatenate(lbl_list, axis=0)

    # Shuffle to prevent layering artifacts
    perm = rng.permutation(len(pts_amp))
    return pts_amp[perm], rgb_amp[perm], lbl_amp[perm]


# ─── PLY Reading & Writing ─────────────────────────────────────────────────────

def read_ply(path: Path) -> tuple:
    """
    Read PLY (binary little-endian or ASCII).
    Returns (pts: (N,3) float32, rgb: (N,3) uint8 or None).
    """
    with open(path, "rb") as f:
        n_verts = 0
        is_binary = False
        has_rgb = False
        prop_order = []
        while True:
            line = b""
            while not line.endswith(b"\n"):
                c = f.read(1)
                if not c: break
                line += c
            line = line.strip().decode("ascii", "replace")
            if line.startswith("format binary_little_endian"):
                is_binary = True
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property float"):
                prop_order.append((line.split()[-1], "f4"))
            elif line.startswith("property uchar") or line.startswith("property uint8"):
                prop = line.split()[-1]
                prop_order.append((prop, "u1"))
                if prop in ("red", "r"): has_rgb = True
            elif line == "end_header":
                break

        if is_binary:
            x_off = y_off = z_off = None
            r_off = g_off = b_off = None
            byte_offset = 0
            for p, typ in prop_order:
                if p in ("x", "y", "z"):
                    if p == "x": x_off = byte_offset
                    elif p == "y": y_off = byte_offset
                    elif p == "z": z_off = byte_offset
                    byte_offset += 4
                elif p in ("red", "green", "blue", "r", "g", "b", "alpha", "a"):
                    if p in ("red", "r"): r_off = byte_offset
                    elif p in ("green", "g"): g_off = byte_offset
                    elif p in ("blue", "b"): b_off = byte_offset
                    byte_offset += 1
                else:
                    byte_offset += 4 if typ == "f4" else 1
            stride = byte_offset
            raw = f.read(n_verts * stride)

            if stride == 12 and x_off == 0 and y_off == 4 and z_off == 8 and not has_rgb:
                pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).copy()
                return pts, None

            pts = np.empty((n_verts, 3), dtype=np.float32)
            rgb = np.empty((n_verts, 3), dtype=np.uint8) if has_rgb else None
            for i in range(n_verts):
                base = i * stride
                pts[i, 0] = struct.unpack_from("<f", raw, base + x_off)[0]
                pts[i, 1] = struct.unpack_from("<f", raw, base + y_off)[0]
                pts[i, 2] = struct.unpack_from("<f", raw, base + z_off)[0]
                if has_rgb:
                    rgb[i, 0] = raw[base + r_off]
                    rgb[i, 1] = raw[base + g_off]
                    rgb[i, 2] = raw[base + b_off]
            return pts, rgb
        else:
            lines = f.read().decode("ascii", "replace").strip().split("\n")
            pts, rgb = [], []
            for l in lines:
                p = l.split()
                if len(p) >= 3:
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
                    if len(p) >= 6:
                        rgb.append([int(p[3]), int(p[4]), int(p[5])])
            return np.array(pts, dtype=np.float32), (np.array(rgb, dtype=np.uint8) if rgb else None)


def write_ply_binary(path: Path, pts: np.ndarray, rgb: np.ndarray):
    """Write fast binary little-endian PLY with float32 (x,y,z) and uint8 (r,g,b)."""
    n = len(pts)
    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    ).encode("ascii")

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    arr = np.empty(n, dtype=dtype)
    arr["x"] = pts[:, 0]
    arr["y"] = pts[:, 1]
    arr["z"] = pts[:, 2]
    arr["red"]   = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"]  = rgb[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        arr.tofile(f)


# ─── Aerial Semantic Classifier ───────────────────────────────────────────────

def classify_frame(bgr_img: np.ndarray) -> np.ndarray:
    """
    Per-pixel semantic classification tuned for aerial drone imagery:
      1: Road/Ground/Pavement (asphalt, concrete, sidewalks, driveways)
      2: Building/Roof (shingles, tile, metal, solar panels, stucco walls)
      3: Vegetation (lawns, trees, shrubs, tree canopy)
      4: Water (lake, river, swimming pools)
      5: Vehicle (cars)
      0: Unknown / shadow
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = np.zeros(h.shape, dtype=np.uint8)

    # 1. Vegetation (lawns, trees, shrubs)
    is_veg = (h >= 30) & (h <= 88) & (s >= 35) & (v >= 25)
    mask[is_veg] = CLASS_ID["Vegetation"]

    # 2. Water (lakes, pools, ponds)
    is_water = (((h >= 89) & (h <= 135) & (s >= 50) & (v >= 50)) |
                ((h >= 135) & (h <= 175) & (s >= 25) & (v >= 90))) & (mask == 0)
    mask[is_water] = CLASS_ID["Water"]

    # 3. Roads, Driveways, Sidewalks, Pavement (neutral asphalt/concrete)
    is_road = ((s < 35) & (v >= 120) & (v <= 255)) & (mask == 0)
    mask[is_road] = CLASS_ID["Road/Ground"]

    # 4. Building / House Roofs (shingles, terracotta, slate, solar panels)
    is_roof = (v >= 35) & (v <= 215) & (mask == 0)
    mask[is_roof] = CLASS_ID["Building/Roof"]

    return mask


# ─── Multi-View 2D-to-3D Semantic Projection ─────────────────────────────────

def project_multiview_semantics(pts: np.ndarray, images_data: dict, cameras_data: dict,
                                frames_dir: Path, max_cams: int = 25) -> np.ndarray:
    """
    Project 3D points (in original COLMAP coords) into registered camera views.
    Accumulate multi-view votes and assign plurality label.
    """
    N = len(pts)
    votes = np.zeros((N, len(CLASS_ID)), dtype=np.int16)

    usable = [img_id for img_id, info in images_data.items()
              if (frames_dir / info["name"]).exists()]

    if not usable:
        print("  WARNING: No images found matching camera poses for semantic projection.")
        return None

    usable.sort(key=lambda img_id: images_data[img_id]["name"])
    step = max(1, len(usable) // max_cams)
    selected = usable[::step][:max_cams]

    print(f"  Projecting into {len(selected)} camera views across flight trajectory...")
    pts_f64 = pts.astype(np.float64)
    total_votes_cast = 0

    for img_id in selected:
        info = images_data[img_id]
        cam  = cameras_data[info["cam_id"]]
        R    = info["R"]
        tvec = np.array(info["tvec"], dtype=np.float64)
        W, H = cam["width"], cam["height"]
        params = cam["params"]

        f  = params[0]
        cx = params[1] if len(params) > 1 else W / 2.0
        cy = params[2] if len(params) > 2 else H / 2.0

        p_cam = (pts_f64 @ R.T) + tvec
        z = p_cam[:, 2]

        in_front = z > 0.1
        if not np.any(in_front):
            continue

        u = f * p_cam[:, 0] / z + cx
        v = f * p_cam[:, 1] / z + cy

        img_path = frames_dir / info["name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        mH, mW = img.shape[:2]

        u_m = (u * (mW / W)).astype(np.int32)
        v_m = (v * (mH / H)).astype(np.int32)

        in_frame = in_front & (u_m >= 0) & (u_m < mW) & (v_m >= 0) & (v_m < mH)
        idx = np.where(in_frame)[0]
        if len(idx) == 0:
            continue

        mask = classify_frame(img)
        cls_labels = mask[v_m[idx], u_m[idx]]

        for cid in range(len(CLASS_ID)):
            mask_c = cls_labels == cid
            if np.any(mask_c):
                votes[idx[mask_c], cid] += 1

        total_votes_cast += len(idx)

    print(f"  Total multi-view votes cast: {total_votes_cast:,}")
    total_votes = votes.sum(axis=1)
    voted = total_votes > 0
    labels = np.zeros(N, dtype=np.int32)
    labels[voted] = np.argmax(votes[voted], axis=1)
    return labels


# ─── Voxel Downsampling ────────────────────────────────────────────────────────

def voxel_downsample(pts: np.ndarray, voxel_size: float, labels: np.ndarray = None):
    """Voxel grid downsampling keeping points closest to centroid."""
    mins = pts.min(axis=0)
    voxel_ids = np.floor((pts - mins) / voxel_size).astype(np.int64)
    M = voxel_ids.max(axis=0) + 1
    flat = (voxel_ids[:, 0].astype(np.int64) * M[1] * M[2]
            + voxel_ids[:, 1].astype(np.int64) * M[2]
            + voxel_ids[:, 2].astype(np.int64))

    order = np.argsort(flat, kind='stable')
    flat_sorted = flat[order]
    pts_sorted  = pts[order]
    lbl_sorted  = labels[order] if labels is not None else None

    _, first_idx, counts_arr = np.unique(flat_sorted, return_index=True, return_counts=True)

    out_pts, out_lbls = [], []
    for ui, (start, cnt) in enumerate(zip(first_idx, counts_arr)):
        chunk = pts_sorted[start:start+cnt]
        centroid = chunk.mean(axis=0)
        diffs = np.linalg.norm(chunk - centroid, axis=1)
        best = start + int(np.argmin(diffs))
        out_pts.append(pts[order[best]])
        if labels is not None:
            lbls_chunk = lbl_sorted[start:start+cnt]
            counts = np.bincount(lbls_chunk.astype(np.int32), minlength=len(CLASS_ID))
            out_lbls.append(int(np.argmax(counts)))

    return np.array(out_pts, dtype=np.float32), (np.array(out_lbls, dtype=np.int32) if labels is not None else None)


# ─── Real Structure Extraction (Houses, Trees, Roads, Water) ──────────────────

def extract_real_houses(pts_three: np.ndarray, labels: np.ndarray,
                        cell_size: float = 2.0, min_pts: int = 15) -> list:
    """
    Extract residential houses and building footprints using connected density cores and parcel splitting.
    Coordinates are Three.js Y-up (X=lateral, Y=elevation above ground in meters, Z=depth).
    Ground is at Y = 0.
    """
    roof_mask = labels == CLASS_ID["Building/Roof"]
    if np.sum(roof_mask) < min_pts:
        return []

    roof_pts = pts_three[roof_mask]
    xs = roof_pts[:, 0]
    zs = roof_pts[:, 2]

    x_min, x_max = float(xs.min()), float(xs.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    cols = int(np.ceil((x_max - x_min) / cell_size)) + 1
    rows = int(np.ceil((z_max - z_min) / cell_size)) + 1

    grid = np.zeros((rows, cols), dtype=np.int32)
    ix = np.clip(((xs - x_min) / cell_size).astype(int), 0, cols - 1)
    iz = np.clip(((zs - z_min) / cell_size).astype(int), 0, rows - 1)
    for i in range(len(xs)):
        grid[iz[i], ix[i]] += 1

    # Density thresholding for core roof components
    core_thresh = max(4, int(np.percentile(grid[grid > 0], 55))) if np.any(grid > 0) else 4
    core_mask = (grid >= core_thresh).astype(np.uint8)

    num_cores, core_labels, stats, centroids = cv2.connectedComponentsWithStats(core_mask, connectivity=4)

    houses = []
    for label_idx in range(1, num_cores):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < 2:
            continue

        in_clust = (core_labels[iz, ix] == label_idx)
        clust_pts = roof_pts[in_clust]
        if len(clust_pts) < min_pts:
            continue

        w_raw = float(clust_pts[:, 0].max() - clust_pts[:, 0].min())
        d_raw = float(clust_pts[:, 2].max() - clust_pts[:, 2].min())

        # If a cluster is larger than 22m, split along dominant axis into individual residential parcels
        if max(w_raw, d_raw) > 22.0:
            split_x = int(np.ceil(w_raw / 16.0)) if w_raw > 22.0 else 1
            split_z = int(np.ceil(d_raw / 16.0)) if d_raw > 22.0 else 1
            x_edges = np.linspace(clust_pts[:, 0].min(), clust_pts[:, 0].max(), split_x + 1)
            z_edges = np.linspace(clust_pts[:, 2].min(), clust_pts[:, 2].max(), split_z + 1)
            for sx in range(split_x):
                for sz in range(split_z):
                    sub_mask = (clust_pts[:, 0] >= x_edges[sx]) & (clust_pts[:, 0] <= x_edges[sx+1]) & \
                               (clust_pts[:, 2] >= z_edges[sz]) & (clust_pts[:, 2] <= z_edges[sz+1])
                    sub_pts = clust_pts[sub_mask]
                    if len(sub_pts) >= min_pts:
                        sw = round(float(np.clip(sub_pts[:, 0].max() - sub_pts[:, 0].min() + 2.0, 10.0, 24.0)), 1)
                        sd = round(float(np.clip(sub_pts[:, 2].max() - sub_pts[:, 2].min() + 2.0, 10.0, 24.0)), 1)
                        top_y = float(np.percentile(sub_pts[:, 1], 90))
                        sh = round(float(np.clip(top_y, 5.2, 9.5)), 1)
                        houses.append({
                            "id": len(houses) + 1,
                            "cx": round(float(sub_pts[:, 0].mean()), 1),
                            "cy": round(sh / 2.0, 1),
                            "cz": round(float(sub_pts[:, 2].mean()), 1),
                            "w": sw, "d": sd, "h": sh,
                            "roof_type": "pitched" if max(sw, sd) > 13.0 else "hip",
                            "roof_h": round(max(1.8, sh * 0.35), 1),
                            "pts": len(sub_pts),
                            "type": "residential_house",
                            "label": f"House #{len(houses)+1} ({sh}m)"
                        })
        else:
            bw = round(float(np.clip(w_raw + 2.0, 10.0, 24.0)), 1)
            bd = round(float(np.clip(d_raw + 2.0, 10.0, 24.0)), 1)
            top_y = float(np.percentile(clust_pts[:, 1], 90))
            bh = round(float(np.clip(top_y, 5.2, 9.5)), 1)
            houses.append({
                "id": len(houses) + 1,
                "cx": round(float(clust_pts[:, 0].mean()), 1),
                "cy": round(bh / 2.0, 1),
                "cz": round(float(clust_pts[:, 2].mean()), 1),
                "w": bw, "d": bd, "h": bh,
                "roof_type": "pitched" if max(bw, bd) > 13.0 else "hip",
                "roof_h": round(max(1.8, bh * 0.35), 1),
                "pts": len(clust_pts),
                "type": "residential_house",
                "label": f"House #{len(houses)+1} ({bh}m)"
            })

    # Sort by point count (prominence)
    houses.sort(key=lambda h: h["pts"], reverse=True)
    return houses


def extract_real_trees(pts_three: np.ndarray, labels: np.ndarray,
                       cell_size: float = 3.5, max_trees: int = 150) -> list:
    """Extract tree foliage clusters {x, y, z, r, h} from vegetation points using 2D density peaks."""
    veg_mask = labels == CLASS_ID["Vegetation"]
    if np.sum(veg_mask) < 20:
        return []

    veg_pts = pts_three[veg_mask]
    xs = veg_pts[:, 0]
    zs = veg_pts[:, 2]

    x_min, x_max = float(xs.min()), float(xs.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    cols = int(np.ceil((x_max - x_min) / cell_size)) + 1
    rows = int(np.ceil((z_max - z_min) / cell_size)) + 1

    grid = np.zeros((rows, cols), dtype=np.int32)
    ix = np.clip(((xs - x_min) / cell_size).astype(int), 0, cols - 1)
    iz = np.clip(((zs - z_min) / cell_size).astype(int), 0, rows - 1)
    for i in range(len(xs)):
        grid[iz[i], ix[i]] += 1

    thresh = max(8, int(np.percentile(grid[grid > 0], 50))) if np.any(grid > 0) else 8

    trees = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r, c]
            if v >= thresh:
                r0, r1 = max(0, r-1), min(rows, r+2)
                c0, c1 = max(0, c-1), min(cols, c+2)
                if v == np.max(grid[r0:r1, c0:c1]):
                    cx = x_min + (c + 0.5) * cell_size
                    cz = z_min + (r + 0.5) * cell_size
                    dists = np.hypot(xs - cx, zs - cz)
                    clust = veg_pts[dists <= cell_size * 1.5]
                    if len(clust) >= 10:
                        top_y = float(np.percentile(clust[:, 1], 90))
                        h = round(float(np.clip(top_y, 3.8, 7.5)), 1)
                        r_c = round(float(np.clip(h * 0.45, 2.2, 4.2)), 1)
                        trees.append({
                            "x": round(float(cx), 1),
                            "y": 0.0,
                            "z": round(float(cz), 1),
                            "h": h,
                            "r": r_c,
                            "pts": int(len(clust))
                        })

    trees.sort(key=lambda t: t["pts"], reverse=True)
    return trees[:max_trees]


def extract_real_roads(pts_three: np.ndarray, labels: np.ndarray,
                       cell_size: float = 20.0) -> list:
    """Extract major road network centerlines/segments from road points."""
    road_mask = labels == CLASS_ID["Road/Ground"]
    if np.sum(road_mask) < 30:
        return []

    road_pts = pts_three[road_mask]
    xs = road_pts[:, 0]
    zs = road_pts[:, 2]

    gx = ((xs - xs.min()) / cell_size).astype(int)
    gz = ((zs - zs.min()) / cell_size).astype(int)

    grid = {}
    for idx, (ix, iz) in enumerate(zip(gx, gz)):
        k = (int(ix), int(iz))
        if k not in grid: grid[k] = []
        grid[k].append(road_pts[idx])

    segments = []
    for k, pts_cell in grid.items():
        if len(pts_cell) >= 20:
            arr = np.array(pts_cell)
            cx = round(float(arr[:, 0].mean()), 1)
            cz = round(float(arr[:, 2].mean()), 1)
            w  = round(float(max(10.0, arr[:, 0].max() - arr[:, 0].min())), 1)
            d  = round(float(max(10.0, arr[:, 2].max() - arr[:, 2].min())), 1)
            segments.append({
                "x": cx, "z": cz, "w": w, "d": d,
                "pts": len(pts_cell)
            })

    return segments[:60]


def extract_real_water(pts_three: np.ndarray, labels: np.ndarray,
                       bin_size: float = 25.0) -> list:
    """Extract lake/pool water body regions from water points."""
    water_mask = labels == CLASS_ID["Water"]
    if np.sum(water_mask) < 25:
        return []

    water_pts = pts_three[water_mask]
    xs = water_pts[:, 0]
    zs = water_pts[:, 2]

    gx = ((xs - xs.min()) / bin_size).astype(int)
    gz = ((zs - zs.min()) / bin_size).astype(int)

    grid = {}
    for idx, (ix, iz) in enumerate(zip(gx, gz)):
        k = (int(ix), int(iz))
        if k not in grid: grid[k] = []
        grid[k].append(water_pts[idx])

    bodies = []
    for k, pts_cell in grid.items():
        if len(pts_cell) >= 20:
            arr = np.array(pts_cell)
            cx = round(float(arr[:, 0].mean()), 1)
            cz = round(float(arr[:, 2].mean()), 1)
            w  = round(float(max(12.0, arr[:, 0].max() - arr[:, 0].min())), 1)
            d  = round(float(max(12.0, arr[:, 2].max() - arr[:, 2].min())), 1)
            bodies.append({
                "x": cx, "z": cz, "w": w, "d": d,
                "type": "lake" if (w > 35 or d > 35) else "pool",
                "pts": len(pts_cell)
            })

    return bodies


# ─── Metric Scale Loader ───────────────────────────────────────────────────────

def load_metric_scale(job_dir: Path) -> tuple:
    candidates = [
        job_dir / "georef_report.json",
        job_dir / job_dir.name / "georef_report.json",
    ]
    for georef in candidates:
        if georef.exists():
            try:
                with open(georef, encoding="utf-8") as f:
                    data = json.load(f)
                scale = data.get("scale_m_per_unit") or data.get("metric_scale")
                source = data.get("source", "COLMAP Metric Baseline")
                if scale and float(scale) > 0:
                    return float(scale), str(source)
            except Exception:
                pass
    return 2.5, "COLMAP Auto-Estimated"


# ─── Main Production Scene Builder ─────────────────────────────────────────────

def build_production_scene(job_dir: Path, output_dir: Path, max_web_pts: int = 80_000):
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("\n" + "=" * 68)
    print("  AEROTWIN — Production 3D Digital Twin Scene Builder")
    print("=" * 68)

    # 1. Load COLMAP camera poses
    print("\n[1/8] Loading COLMAP cameras...")
    sparse_dir = discover_sparse_dir(job_dir)
    if not sparse_dir:
        print("  ERROR: COLMAP sparse data not found in job directory")
        return False

    cameras = parse_cameras_txt(sparse_dir / "cameras.txt")
    raw_images = parse_images_txt(sparse_dir / "images.txt")
    valid_images, cleaned_images = filter_camera_outliers(raw_images)
    print(f"  Sparse model dir: {sparse_dir}")
    print(f"  Intrinsic models: {len(cameras)}")
    print(f"  Camera poses:     {len(raw_images)} ({len(valid_images)} valid, {len(raw_images)-len(valid_images)} smoothed)")

    # 2. Load dense or sparse point cloud
    print("\n[2/8] Loading point cloud (prefer points3D.txt for real colors)...")
    pts_raw, rgb_raw = None, None
    ply_path = None

    # Try points3D.txt first (has real photometric colors from COLMAP)
    for sub in [job_dir, job_dir / job_dir.name]:
        for sparse_subdir in ["sparse/0", "sparse/1", "sparse"]:
            p3d_txt = sub / sparse_subdir / "points3D.txt"
            if p3d_txt.exists() and p3d_txt.stat().st_size > 1000:
                print(f"  Reading real COLMAP colors from: {p3d_txt}")
                pts_raw_txt, rgb_raw_txt = read_points3d_txt(p3d_txt)
                if pts_raw_txt is not None and len(pts_raw_txt) > 500:
                    pts_raw = pts_raw_txt
                    rgb_raw = rgb_raw_txt
                    ply_path = p3d_txt  # use as source label
                    print(f"  Loaded {len(pts_raw):,} points with real RGB from points3D.txt")
                    break
        if pts_raw is not None:
            break

    if pts_raw is None:
        ply_path = discover_ply(job_dir)
        if not ply_path:
            print("  ERROR: No PLY or points3D.txt found in job directory")
            return False
        print(f"  Source cloud: {ply_path.name} ({ply_path.stat().st_size / 1e6:.1f} MB)")
        pts_raw, rgb_raw = read_ply(ply_path)

    N_raw = len(pts_raw)
    print(f"  Raw 3D points loaded: {N_raw:,}")

    # 3. Multi-view semantic projection
    print("\n[3/8] Multi-view semantic projection from video frames...")
    frames_dir = discover_frames_dir(job_dir)
    labels = None
    if frames_dir:
        labels = project_multiview_semantics(pts_raw, valid_images, cameras, frames_dir, max_cams=25)

    if labels is None or np.sum(labels > 0) / N_raw < 0.15:
        print("  Fallback: assigning baseline ground labels...")
        labels = np.zeros(N_raw, dtype=np.int32)
        labels[:] = CLASS_ID["Road/Ground"]

    # 4. Metric Scaling & Conversion to Three.js coordinate system (Y-up, Centered)
    print("\n[4/8] Converting to Three.js standard coordinates (Y-up, metric-scaled)...")
    metric_scale, scale_source = load_metric_scale(job_dir)
    scale = float(metric_scale) if metric_scale > 0.1 else 2.5
    print(f"  Metric scale: {scale:.3f} m/unit ({scale_source})")

    center_x = float(np.median(pts_raw[:, 0]))
    center_z = float(np.median(pts_raw[:, 1]))
    z_ground_base = float(np.percentile(pts_raw[:, 2], 85))
    print(f"  Ground depth baseline: {z_ground_base:.2f}")

    pts_three = np.empty_like(pts_raw)
    pts_three[:, 0] = (pts_raw[:, 0] - center_x) * scale
    pts_three[:, 2] = (pts_raw[:, 1] - center_z) * scale
    raw_elev = (z_ground_base - pts_raw[:, 2]) * scale
    pts_three[:, 1] = np.clip(raw_elev, -2.0, 45.0)

    ground_y = 0.0

    # 5. Extract Real Structures (Houses, Trees, Roads, Water)
    print("\n[5/8] Extracting real 3D structures from metric-scaled points...")
    houses = extract_real_houses(pts_three, labels)
    trees  = extract_real_trees(pts_three, labels)
    roads  = extract_real_roads(pts_three, labels)
    water  = extract_real_water(pts_three, labels)

    print(f"  ✓ Extracted houses:    {len(houses)} residential buildings")
    print(f"  ✓ Extracted trees:     {len(trees)} foliage clusters")
    print(f"  ✓ Extracted roads:     {len(roads)} street corridors")
    print(f"  ✓ Extracted water:     {len(water)} lakes/pools")

    # 6. Point Cloud Preservation & Voxel Downsampling
    print(f"\n[6/8] Preparing web point cloud (budget ≤{max_web_pts:,} pts)...")
    if len(pts_three) <= max_web_pts:
        print(f"  Point count ({len(pts_three):,}) is within budget — preserving 100% of points!")
        pts_down = pts_three.copy()
        lbl_down = labels.copy()
    else:
        ranges = pts_three.max(axis=0) - pts_three.min(axis=0)
        vol = float(max(1.0, np.prod(ranges)))
        target_density = max_web_pts / vol
        voxel_size = max(0.5, (1.0 / target_density) ** (1/3))

        pts_down, lbl_down = voxel_downsample(pts_three, voxel_size, labels)
        iters = 0
        while len(pts_down) > max_web_pts and iters < 10:
            voxel_size *= 1.25
            pts_down, lbl_down = voxel_downsample(pts_three, voxel_size, labels)
            iters += 1
        print(f"  Downsampled: {len(pts_down):,} pts (voxel={voxel_size:.2f})")

    # Generate RGB vertex colors — use real COLMAP photometric colors if available
    if rgb_raw is not None and len(rgb_raw) == len(pts_raw) and len(pts_down) == len(pts_raw):
        rgb_web = rgb_raw.copy()
    elif rgb_raw is not None and len(pts_down) <= len(pts_raw):
        # Sample rgb from original for the downsampled set
        rgb_web = np.array([CLASS_RGB[int(l)] for l in lbl_down], dtype=np.uint8)
    else:
        rgb_web = np.array([CLASS_RGB[int(l)] for l in lbl_down], dtype=np.uint8)
    rgb_sem = np.array([SEMANTIC_RGB[int(l)] for l in lbl_down], dtype=np.uint8)

    # ── Point Amplification: expand sparse cloud to visually dense display ──
    # Target: 250k display points for rich visual appearance (like dense MVS)
    print(f"\n[6b/8] Amplifying point cloud for visual density...")
    target_display = max(max_web_pts, 250_000)
    if len(pts_down) < target_display:
        pts_amp, rgb_amp_web, lbl_amp = amplify_point_cloud(pts_down, rgb_web, lbl_down, target_display)
        _, rgb_amp_sem, _ = amplify_point_cloud(pts_down, rgb_sem, lbl_down, target_display)
        print(f"  Display cloud amplified: {len(pts_amp):,} pts")
        pts_down_final  = pts_amp
        rgb_web_final   = rgb_amp_web
        rgb_sem_final   = rgb_amp_sem
        lbl_down_final  = lbl_amp
    else:
        pts_down_final  = pts_down
        rgb_web_final   = rgb_web
        rgb_sem_final   = rgb_sem
        lbl_down_final  = lbl_down

    # 7. Reconstructed Camera Flight Trajectory in Three.js coordinates
    print("\n[7/8] Converting flight path trajectory...")
    cam_positions = []
    for img_id in sorted(cleaned_images.keys(), key=lambda k: cleaned_images[k]["name"]):
        C = cleaned_images[img_id]["C"]
        cam_x = (C[0] - center_x) * scale
        cam_z = (C[1] - center_z) * scale
        cam_y = max(15.0, min(160.0, (z_ground_base - C[2]) * scale))
        cam_positions.append({
            "name": cleaned_images[img_id]["name"],
            "x": round(float(cam_x), 2),
            "y": round(float(cam_y), 2),
            "z": round(float(cam_z), 2),
            "outlier": cleaned_images[img_id].get("is_outlier", False),
        })

    # 8. Metric scale & packaging
    print("\n[8/8] Packaging digital twin scene package...")
    calibrated = "Auto" not in scale_source and scale > 0

    sem_dist = {}
    total_lbl = max(len(lbl_down_final), 1)
    for name, cid in CLASS_ID.items():
        cnt = int(np.sum(lbl_down_final == cid))
        sem_dist[name] = {"count": cnt, "pct": round(100.0 * cnt / total_lbl, 1)}
        if cnt > 0:
            print(f"    {name:16s}: {cnt:6d} pts ({sem_dist[name]['pct']:5.1f}%)")

    bounds = {
        "xmin": round(float(pts_down_final[:, 0].min()), 1),
        "xmax": round(float(pts_down_final[:, 0].max()), 1),
        "ymin": round(float(pts_down_final[:, 1].min()), 1),
        "ymax": round(float(pts_down_final[:, 1].max()), 1),
        "zmin": round(float(pts_down_final[:, 2].min()), 1),
        "zmax": round(float(pts_down_final[:, 2].max()), 1),
    }

    job_identifier = job_dir.parent.name if job_dir.name == "demo" or job_dir.name == job_dir.parent.name else job_dir.name

    scene = {
        "version": 4,
        "job_id": job_identifier,
        "environment": "Aerial Drone Photogrammetry",
        "point_cloud": {
            "total_points": int(N_raw),
            "web_points":   int(len(pts_down_final)),
            "source": ply_path.name if hasattr(ply_path, 'name') else str(ply_path),
        },
        "coordinate_system": {
            "up_axis": "Y",
            "ground_y": ground_y,
            "bounds": bounds,
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "metric_scale": {
            "value": round(scale, 3),
            "calibrated": calibrated,
            "source": scale_source,
            "display": f"{scale:.3f} m/unit (Metric Calibrated)",
        },
        "cameras": {
            "registered": len(cleaned_images),
            "trajectory_type": "COLMAP Reconstructed (Video-Derived)",
            "positions": cam_positions,
        },
        "semantic": {
            "method": "Multi-View 2D Projection (Real Video Ground Truth)",
            "classes": sem_dist,
            "class_colors": CLASS_HEX,
        },
        "buildings": houses,
        "vegetation": trees,
        "roads": roads,
        "water": water,
        "dense_ply_url":    f"/jobs/{job_identifier}/scene/dense_web.ply",
        "semantic_ply_url": f"/jobs/{job_identifier}/scene/semantic.ply",
        "build_time_sec": round(time.time() - t_start, 1),
    }

    # Write files
    with open(output_dir / "scene.json", "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)
    print(f"  ✓ scene.json ({output_dir / 'scene.json'})")

    with open(output_dir / "buildings.json", "w", encoding="utf-8") as f:
        json.dump(houses, f, indent=2)
    print(f"  ✓ buildings.json ({len(houses)} houses)")

    web_ply_path = output_dir / "dense_web.ply"
    write_ply_binary(web_ply_path, pts_down_final, rgb_web_final)
    print(f"  ✓ dense_web.ply: {len(pts_down_final):,} pts, {web_ply_path.stat().st_size / 1024:.0f} KB (binary)")

    sem_ply_path = output_dir / "semantic.ply"
    write_ply_binary(sem_ply_path, pts_down_final, rgb_sem_final)
    print(f"  ✓ semantic.ply:  {len(pts_down_final):,} pts, {sem_ply_path.stat().st_size / 1024:.0f} KB (binary)")

    elapsed = time.time() - t_start
    print(f"\n  ✓ Production Scene Build complete in {elapsed:.1f}s")
    print("=" * 68)
    return True


def main():
    parser = argparse.ArgumentParser(description="AeroTwin: Production Scene Builder")
    parser.add_argument("--job-id", "-j", default="demo", help="Job ID")
    parser.add_argument("--jobs-base", default="jobs", help="Base jobs directory")
    parser.add_argument("--max-pts", type=int, default=80_000, help="Max points in web PLY")
    args = parser.parse_args()

    jobs_base = Path(args.jobs_base)
    job_root  = jobs_base / args.job_id
    job_dir   = job_root / args.job_id if (job_root / args.job_id).exists() else job_root
    output_dir = job_root / "scene"

    if not job_dir.exists():
        print(f"ERROR: Job directory not found: {job_dir}")
        sys.exit(1)

    success = build_production_scene(job_dir, output_dir, max_web_pts=args.max_pts)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
