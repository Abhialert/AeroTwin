import cv2
import numpy as np
from pathlib import Path
from scripts.test_proj import cams, pts_display, f_val, cx, cy, W, H

img = cv2.imread('jobs/demo/demo/frames/frame_00025.jpg')
img_h, img_w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
is_veg = (h >= 30) & (h <= 88) & (s >= 35) & (v >= 25)
is_road = ((s < 45) & (v >= 115) & (v <= 250)) | ((s < 30) & (v >= 80) & (v < 115))
is_water = ((h >= 89) & (h <= 135) & (s >= 35) & (v >= 35)) | ((h >= 135) & (h <= 175) & (s >= 15) & (v >= 90))
is_roof = (((h >= 5) & (h <= 28) & (s >= 35) & (v >= 50)) | ((s < 60) & (v >= 30) & (v < 85))) & ~is_veg & ~is_water

mask = np.zeros((img_h, img_w), dtype=np.uint8)
mask[is_veg] = 3
mask[is_road & (mask == 0)] = 1
mask[is_water & (mask == 0)] = 6
mask[is_roof & (mask == 0)] = 2

cam = cams['frame_00025.jpg']
p_cam = (pts_display @ cam['R'].T) + cam['tvec']
in_front = p_cam[:, 2] > 0.1
u = f_val * p_cam[:, 0] / p_cam[:, 2] + cx
v = f_val * p_cam[:, 1] / p_cam[:, 2] + cy
u_img = (u * (img_w / W)).astype(np.int32)
v_img = (v * (img_h / H)).astype(np.int32)
in_frame = in_front & (u_img >= 0) & (u_img < img_w) & (v_img >= 0) & (v_img < img_h)

pts_in = pts_display[in_frame]
labels = mask[v_img[in_frame], u_img[in_frame]]

# Extract roof points
roof_pts = pts_in[labels == 2]
print(f"Total roof points: {len(roof_pts)}")

# Fast pure-numpy 2D spatial clustering
cell_size = 14.0
gx = ((roof_pts[:, 0] - roof_pts[:, 0].min()) / cell_size).astype(np.int32)
gy = ((roof_pts[:, 1] - roof_pts[:, 1].min()) / cell_size).astype(np.int32)

grid = {}
for idx, (ix, iy) in enumerate(zip(gx, gy)):
    key = (int(ix), int(iy))
    if key not in grid:
        grid[key] = []
    grid[key].append(idx)

# Connected components over 3x3 neighbor cells
visited = set()
clusters = []
for key in grid:
    if key in visited:
        continue
    queue = [key]
    cluster_indices = []
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        if cur in grid:
            cluster_indices.extend(grid[cur])
            cx_k, cy_k = cur
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                neighbor = (cx_k + dx, cy_k + dy)
                if neighbor not in visited and neighbor in grid:
                    queue.append(neighbor)
    
    if len(cluster_indices) >= 15:
        c_pts = roof_pts[cluster_indices]
        xmin, xmax = c_pts[:, 0].min(), c_pts[:, 0].max()
        ymin, ymax = c_pts[:, 1].min(), c_pts[:, 1].max()
        w = float(xmax - xmin)
        d = float(ymax - ymin)
        if 8.0 <= w <= 80.0 and 8.0 <= d <= 80.0:
            clusters.append({
                'pts': len(c_pts),
                'cx': float((xmin + xmax) / 2.0),
                'cy': float((ymin + ymax) / 2.0),
                'w': round(w, 1),
                'd': round(d, 1),
            })

print(f"Valid residential house footprints detected: {len(clusters)}")
for i, c in enumerate(clusters[:12]):
    print(f"  House {i+1}: center=({c['cx']:.1f}, {c['cy']:.1f}), footprint={c['w']}m x {c['d']}m, pts={c['pts']}")

