import cv2
import numpy as np
from pathlib import Path

frames_dir = Path('jobs/demo/demo/frames')
frames = sorted(list(frames_dir.glob('*.jpg')))
print(f'Total frames: {len(frames)}')

sample = frames[::max(1, len(frames)//10)][:10]
for f in sample:
    img = cv2.imread(str(f))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    is_veg = (h >= 30) & (h <= 88) & (s >= 35) & (v >= 25)
    is_road = ((s < 45) & (v >= 115) & (v <= 250)) | ((s < 30) & (v >= 80) & (v < 115))
    is_water = ((h >= 89) & (h <= 135) & (s >= 35) & (v >= 35)) | ((h >= 135) & (h <= 175) & (s >= 15) & (v >= 90))
    is_roof = (((h >= 5) & (h <= 28) & (s >= 35) & (v >= 50)) | ((s < 60) & (v >= 30) & (v < 85))) & ~is_veg & ~is_water
    tot = float(h.size)
    print(f"{f.name}: Veg={np.sum(is_veg)/tot*100:5.1f}% Road={np.sum(is_road)/tot*100:5.1f}% Water={np.sum(is_water)/tot*100:5.1f}% Roof={np.sum(is_roof)/tot*100:5.1f}%")
