"""
scripts/08_semantic_filter.py
AeroTwin — Semantic Classification & Dynamic Object Filtering

Performs:
  - HSV color-space semantic segmentation (road, building, vegetation, sky)
  - MOG2 background subtraction to detect dynamic objects
  
Designed for aerial/nadir drone imagery.

Outputs:
  data/output/<project>/semantics/     — colorized semantic overlay images
  data/output/<project>/dynamic_masks/ — binary foreground masks
  data/output/<project>/semantic_classification_report.json
  data/output/<project>/dynamic_object_report.json
"""

import argparse
import cv2
import json
import sys
import time
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


CLASS_NAMES = {
    0: "Unknown",
    1: "Road/Ground",
    2: "Building/Roof",
    3: "Vegetation",
    4: "Water",
    5: "Vehicle",
}

# BGR colors for overlay visualization
CLASS_COLORS = {
    0: (60,  60,  60),   # Unknown       — dark gray
    1: (110, 115, 125),  # Road/Ground   — asphalt gray
    2: (220, 180, 50),   # Building/Roof — golden cyan/orange
    3: (35,  180, 60),   # Vegetation    — vibrant green
    4: (210, 140, 30),   # Water         — azure blue (BGR)
    5: (30,  120, 240),  # Vehicle       — orange (BGR)
}


def classify_pixel_hsv(bgr_img):
    """
    Per-pixel semantic classification using HSV color space.
    Tuned for aerial/nadir drone imagery across residential, urban, and landscape scenes.
    Returns a per-pixel class mask.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = np.zeros(h.shape, dtype=np.uint8)

    # 1. Vegetation (lawns, trees, shrubs, foliage)
    is_veg = (h >= 30) & (h <= 88) & (s >= 35) & (v >= 25)
    mask[is_veg] = 3

    # 2. Water (lakes, ponds, swimming pools, rivers)
    is_water = (((h >= 89) & (h <= 135) & (s >= 50) & (v >= 50)) |
                ((h >= 135) & (h <= 175) & (s >= 25) & (v >= 90))) & (mask == 0)
    mask[is_water] = 4

    # 3. Roads, Driveways, Sidewalks, Pavement (neutral asphalt/concrete)
    is_road = ((s < 35) & (v >= 120) & (v <= 255)) & (mask == 0)
    mask[is_road] = 1

    # 4. Building / House Roofs (shingles, terracotta, slate, solar panels, walls)
    is_roof = (v >= 35) & (v <= 215) & (mask == 0)
    mask[is_roof] = 2

    return mask


def run_semantic_segmentation(frames_dir: Path, output_dir: Path) -> dict:
    """Process a sample of frames for semantic segmentation."""
    frames = sorted(list(frames_dir.glob("*.jpg")))
    if not frames:
        print(f"  WARNING: No frames in {frames_dir}")
        return {"frames": 0, "results": []}

    semantics_dir = output_dir / "semantics"
    semantics_dir.mkdir(exist_ok=True)

    # Sample at most 30 frames spread across the video
    max_samples = min(30, len(frames))
    step = max(1, len(frames) // max_samples)
    sample_frames = frames[::step][:max_samples]

    print(f"  Semantic segmentation on {len(sample_frames)} frames "
          f"(out of {len(frames)} total extracted)...")

    results = []
    for frame_path in sample_frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        # Work at reduced resolution for speed
        small = cv2.resize(img, (960, 540))
        mask = classify_pixel_hsv(small)
        total = mask.size

        class_stats = {}
        for cls_id, name in CLASS_NAMES.items():
            pct = round(float(np.sum(mask == cls_id) / total * 100), 2)
            class_stats[name] = pct

        # Generate colorized overlay
        overlay = small.copy()
        for cls_id, color in CLASS_COLORS.items():
            region = mask == cls_id
            if np.any(region):
                overlay[region] = (
                    small[region] * 0.4 + np.array(color, dtype=np.float32) * 0.6
                ).clip(0, 255).astype(np.uint8)

        # Save side-by-side original + semantic
        combined = np.concatenate([small, overlay], axis=1)
        out_name = "sem_" + frame_path.name
        cv2.imwrite(str(semantics_dir / out_name), combined,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        results.append({
            "frame": frame_path.name,
            "semantic_image": out_name,
            "classes": class_stats,
        })

    # Compute averages across all frames
    if results:
        avg = {}
        for cls_name in CLASS_NAMES.values():
            avg[cls_name] = round(
                sum(r["classes"].get(cls_name, 0) for r in results) / len(results), 2
            )
    else:
        avg = {}

    report = {
        "method": "HSV_Semantic_Segmentation",
        "frames_analyzed": len(results),
        "total_extracted_frames": len(frames),
        "class_averages": avg,
        "results": results,
    }

    out_path = output_dir / "semantic_classification_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Semantic masks saved: {semantics_dir}")
    if avg:
        print("  Class averages:")
        for cls_name, pct in sorted(avg.items(), key=lambda x: -x[1]):
            if pct > 0.1:
                print(f"    {cls_name:20s}: {pct:.1f}%")

    return report


def run_dynamic_filtering(frames_dir: Path, output_dir: Path) -> dict:
    """Detect dynamic objects using MOG2 background subtraction."""
    frames = sorted(list(frames_dir.glob("*.jpg")))
    if not frames:
        return {"total_frames": 0, "high_motion_count": 0}

    masks_dir = output_dir / "dynamic_masks"
    masks_dir.mkdir(exist_ok=True)

    print(f"  Dynamic filtering on {len(frames)} frames (MOG2)...")

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=80, varThreshold=35, detectShadows=True
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    per_frame = []
    high_motion = []

    for idx, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        small = cv2.resize(img, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fg = bg_sub.apply(gray)
        fg_clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        dynamic_px = int(np.sum(fg_clean > 200))
        ratio = round(dynamic_px / fg_clean.size * 100, 3)

        per_frame.append({"frame": frame_path.name, "dynamic_pct": ratio})
        if ratio > 2.0:
            high_motion.append(frame_path.name)

        # Save sample masks every 20 frames
        if idx % 20 == 0:
            cv2.imwrite(str(masks_dir / ("mask_" + frame_path.name)), fg_clean)

    report = {
        "total_frames": len(frames),
        "high_motion_count": len(high_motion),
        "high_motion_frames": high_motion,
        "per_frame": per_frame,
    }

    out_path = output_dir / "dynamic_object_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ High-motion frames: {len(high_motion)} / {len(frames)}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="AeroTwin: Semantic classification + dynamic object filtering"
    )
    parser.add_argument("--project", "-p", default="sample_flight",
                        help="Project name")
    parser.add_argument("--output-base", default="data/output",
                        help="Base output directory")
    args = parser.parse_args()

    project_dir = Path(args.output_base) / args.project
    frames_dir = project_dir / "frames"

    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  AEROTWIN — Semantic & Dynamic Analysis")
    print("=" * 60)

    t0 = time.time()
    run_semantic_segmentation(frames_dir, project_dir)
    print()
    run_dynamic_filtering(frames_dir, project_dir)
    print(f"\n  Total time: {round(time.time()-t0, 1)}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
