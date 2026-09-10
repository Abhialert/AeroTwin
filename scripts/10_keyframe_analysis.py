"""
scripts/10_keyframe_analysis.py
AeroTwin — Intelligent Keyframe Selection
Analyzes extracted frames to select optimal keyframes based on:
- Blur (Laplacian variance)
- Redundancy (MSE similarity)
- Feature richness (ORB keypoints)

Outputs:
  data/output/<project>/keyframes/       — selected keyframe images
  data/output/<project>/keyframe_selection_report.json
"""

import argparse
import cv2
import numpy as np
import json
import shutil
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def compute_blur_score(image_gray):
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def compute_mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err


def select_keyframes(frames_dir: Path, output_dir: Path):
    """
    Scan extracted frames and select optimal keyframes.
    Copies selected frames into output_dir/keyframes/.
    Returns a statistics dict.
    """
    t_start = time.time()
    frames = sorted(list(frames_dir.glob("*.jpg")))
    if not frames:
        print(f"  ERROR: No frames found in {frames_dir}")
        return None

    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(exist_ok=True, parents=True)

    total_frames = len(frames)
    print(f"\n  Keyframe Selection — analyzing {total_frames} extracted frames")
    print(f"  Output keyframes → {keyframes_dir}")

    # Config
    blur_threshold = 60.0
    redundancy_mse_threshold = 150.0  # Lower = more similar = reject
    min_features = 300
    orb = cv2.ORB_create(nfeatures=2000)

    stats = {
        "total_extracted_frames": total_frames,
        "frames_analyzed": 0,
        "rejected_blurry": 0,
        "rejected_redundant": 0,
        "rejected_low_information": 0,
        "selected_keyframes": 0,
        "keyframe_filenames": [],
    }

    last_kept_gray = None
    kf_idx = 0

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        stats["frames_analyzed"] += 1

        # Resize for fast analysis
        small = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 1. Blur check
        blur = compute_blur_score(gray)
        if blur < blur_threshold:
            stats["rejected_blurry"] += 1
            continue

        # 2. Redundancy check
        if last_kept_gray is not None:
            mse = compute_mse(gray, last_kept_gray)
            if mse < redundancy_mse_threshold:
                stats["rejected_redundant"] += 1
                continue

        # 3. Feature richness
        kp = orb.detect(gray, None)
        if len(kp) < min_features:
            stats["rejected_low_information"] += 1
            continue

        # Selected — copy to keyframes directory
        dest_name = f"keyframe_{kf_idx:05d}.jpg"
        dest = keyframes_dir / dest_name
        shutil.copy2(str(frame_path), str(dest))

        stats["keyframe_filenames"].append(dest_name)
        last_kept_gray = gray
        kf_idx += 1
        stats["selected_keyframes"] += 1

    elapsed = round(time.time() - t_start, 2)
    stats["processing_time_sec"] = elapsed

    print(f"\n  === Keyframe Selection Report ===")
    print(f"  Total extracted frames:  {total_frames}")
    print(f"  Frames analyzed:         {stats['frames_analyzed']}")
    print(f"  Rejected (blurry):       {stats['rejected_blurry']}")
    print(f"  Rejected (redundant):    {stats['rejected_redundant']}")
    print(f"  Rejected (low info):     {stats['rejected_low_information']}")
    print(f"  SELECTED KEYFRAMES:      {stats['selected_keyframes']}")
    print(f"  Processing time:         {elapsed}s")

    if stats["selected_keyframes"] < 5:
        print(f"\n  WARNING: Only {stats['selected_keyframes']} keyframes selected!")
        print(f"  Consider lowering blur/redundancy thresholds.")
        if stats["selected_keyframes"] == 0:
            print("  CRITICAL: 0 keyframes — falling back to using all frames.")
            # Fallback: copy all frames as keyframes
            for i, fp in enumerate(frames):
                dest_name = f"keyframe_{i:05d}.jpg"
                shutil.copy2(str(fp), str(keyframes_dir / dest_name))
                stats["keyframe_filenames"].append(dest_name)
                stats["selected_keyframes"] += 1
            print(f"  Fallback: {stats['selected_keyframes']} frames copied.")

    # Save report
    report_path = output_dir / "keyframe_selection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="AeroTwin: Intelligent keyframe selection from extracted frames"
    )
    parser.add_argument("--project", "-p", default="sample_flight",
                        help="Project name (matches frame extraction output)")
    parser.add_argument("--output-base", default="data/output",
                        help="Base output directory")
    args = parser.parse_args()

    project_dir = Path(args.output_base) / args.project
    frames_dir = project_dir / "frames"

    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        print("Run 01_extract_frames.py first!")
        sys.exit(1)

    result = select_keyframes(frames_dir, project_dir)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
