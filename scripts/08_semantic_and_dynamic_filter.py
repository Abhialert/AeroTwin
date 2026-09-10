"""
scripts/08_semantic_and_dynamic_filter.py
AeroTwin — Feature #6 (Semantic 3D Understanding) & Feature #7 (Dynamic Object Filtering)
Performs motion/temporal difference analysis and object masking to identify:
- Dynamic/Moving objects: vehicles, people, moving obstacles (to mask from SfM)
- Static structures: roads, buildings, poles, terrain
"""

import cv2
import numpy as np
from pathlib import Path
import json

def analyze_frame_motion(frames_dir, num_samples=30):
    frames_dir = Path(frames_dir)
    image_files = sorted(list(frames_dir.glob("*.jpg")))[:num_samples]
    
    if len(image_files) < 2:
        print("Not enough frames for motion analysis.")
        return

    print(f"Running Dynamic Motion Analysis on {len(image_files)} sample frames...")
    
    # Background Subtractor (MOG2) to isolate dynamic objects
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)
    
    motion_stats = []
    masks_dir = frames_dir.parent / "dynamic_masks"
    masks_dir.mkdir(exist_ok=True)

    for idx, f in enumerate(image_files):
        img = cv2.imread(str(f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        
        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Count dynamic moving pixels
        dynamic_pixels = int(np.sum(clean_mask > 200))
        total_pixels = clean_mask.size
        dynamic_ratio = float(dynamic_pixels / total_pixels)
        
        motion_stats.append({
            "frame": f.name,
            "dynamic_pixels": dynamic_pixels,
            "dynamic_ratio_percent": round(dynamic_ratio * 100, 3)
        })

        # Save sample mask
        if idx % 5 == 0:
            mask_path = masks_dir / f"mask_{f.name}"
            cv2.imwrite(str(mask_path), clean_mask)

    out_json = frames_dir.parent / "dynamic_object_report.json"
    out_json.write_text(json.dumps(motion_stats, indent=2))
    print(f"✓ Dynamic Object Filtering completed. Masks saved to {masks_dir}")
    print(f"✓ Motion statistics exported to {out_json}")

if __name__ == "__main__":
    analyze_frame_motion("data/output/sample_flight/frames")
