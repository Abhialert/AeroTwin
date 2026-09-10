import cv2
import numpy as np
from pathlib import Path
import time

t0 = time.time()

# 1. Load camera intrinsics
with open('jobs/demo/demo/sparse/1/cameras.txt') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p = line.split()
        W, H = int(p[2]), int(p[3])
        f_val, cx, cy = float(p[4]), float(p[5]), float(p[6])
        break

def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y],
    ])

# 2. Load images
with open('jobs/demo/demo/sparse/1/images.txt') as f:
    lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]

cams = {}
for i in range(0, len(lines), 2):
    meta = lines[i].split()
    name = meta[9]
    qvec = list(map(float, meta[1:5]))
    tvec = np.array(list(map(float, meta[5:8])))
    R = qvec_to_rotmat(qvec)
    C = -R.T @ tvec
    cams[name] = {'R': R, 'tvec': tvec, 'C': C}

# 3. Load dense point cloud
with open('jobs/demo/demo/dense_display.ply', 'rb') as f:
    while True:
        l = f.readline().decode('ascii', 'replace').strip()
        if l == 'end_header': break
    pts = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 3)

N = len(pts)

# Filter cams to those with actual frames
frames_dir = Path('jobs/demo/demo/frames')
usable_cams = [c for c in sorted(cams.keys()) if (frames_dir / c).exists()]
print(f"Usable registered cameras with frames: {len(usable_cams)}")

# Sample 25 cameras evenly across flight
step = max(1, len(usable_cams) // 25)
selected_cams = usable_cams[::step][:25]

frame_masks = {}
for name in selected_cams:
    p = frames_dir / name
    img = cv2.imread(str(p))
    if img is None: continue
    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    is_veg = (h >= 30) & (h <= 88) & (s >= 35) & (v >= 25)
    is_water = ((h >= 89) & (h <= 135) & (s >= 50) & (v >= 60)) | ((h >= 135) & (h <= 175) & (s >= 25) & (v >= 100))
    is_road = ((s < 35) & (v >= 125) & (v <= 255))
    is_roof = ~is_veg & ~is_water & ~is_road & (v >= 40) & (v <= 210)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[is_veg] = 3
    mask[is_road & (mask == 0)] = 1
    mask[is_water & (mask == 0)] = 6
    mask[is_roof & (mask == 0)] = 2
    frame_masks[name] = (mask, img_w, img_h)

# Multi-view voting array: N x 7 classes
votes = np.zeros((N, 7), dtype=np.int16)
pts_f64 = pts.astype(np.float64)

for name, (mask, img_w, img_h) in frame_masks.items():
    cam = cams[name]
    p_cam = (pts_f64 @ cam['R'].T) + cam['tvec']
    z = p_cam[:, 2]
    in_front = z > 0.1
    u = f_val * p_cam[:, 0] / z + cx
    v = f_val * p_cam[:, 1] / z + cy
    u_img = (u * (img_w / W)).astype(np.int32)
    v_img = (v * (img_h / H)).astype(np.int32)
    in_frame = in_front & (u_img >= 0) & (u_img < img_w) & (v_img >= 0) & (v_img < img_h)
    idx = np.where(in_frame)[0]
    if len(idx) == 0: continue
    cls_labels = mask[v_img[idx], u_img[idx]]
    for cid in range(7):
        mask_c = cls_labels == cid
        if np.any(mask_c):
            votes[idx[mask_c], cid] += 1

total_votes = votes.sum(axis=1)
voted = total_votes > 0
labels = np.zeros(N, dtype=np.int32)
labels[voted] = np.argmax(votes[voted], axis=1)

print(f"\nMulti-view projection completed in {time.time()-t0:.1f}s. Voted: {np.sum(voted):,} / {N:,}")
for cid, name in [(1, 'Road/Pavement'), (2, 'House/Roof'), (3, 'Vegetation/Trees'), (6, 'Water/Pool/Lake'), (0, 'Unknown/Shadow')]:
    cnt = np.sum(labels == cid)
    print(f"  {name:20s}: {cnt:6d} ({cnt/N*100:4.1f}%)")

# Extract houses across entire scene
roof_pts = pts[labels == 2]
print(f"\nTotal Multi-View Roof Points: {len(roof_pts):,}")
cell_size = 20.0
gx = ((roof_pts[:, 0] - roof_pts[:, 0].min()) / cell_size).astype(np.int32)
gy = ((roof_pts[:, 1] - roof_pts[:, 1].min()) / cell_size).astype(np.int32)
grid = {}
for idx, (ix, iy) in enumerate(zip(gx, gy)):
    key = (int(ix), int(iy))
    if key not in grid: grid[key] = []
    grid[key].append(idx)

visited = set()
houses = []
for key in grid:
    if key in visited: continue
    queue = [key]
    c_idx = []
    while queue:
        cur = queue.pop()
        if cur in visited: continue
        visited.add(cur)
        if cur in grid:
            c_idx.extend(grid[cur])
            cx_k, cy_k = cur
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                nb = (cx_k+dx, cy_k+dy)
                if nb not in visited and nb in grid: queue.append(nb)
    if len(c_idx) >= 20:
        c_pts = roof_pts[c_idx]
        xmin, xmax = c_pts[:, 0].min(), c_pts[:, 0].max()
        ymin, ymax = c_pts[:, 1].min(), c_pts[:, 1].max()
        w = float(xmax - xmin)
        d = float(ymax - ymin)
        if 10.0 <= w <= 80.0 and 10.0 <= d <= 80.0:
            houses.append({
                'cx': round(float((xmin + xmax)/2), 1),
                'cy': round(float((ymin + ymax)/2), 1),
                'w': round(w, 1),
                'd': round(d, 1),
                'pts': len(c_idx)
            })

houses.sort(key=lambda h: h['pts'], reverse=True)
print(f"Total Detected Houses in Neighborhood: {len(houses)}")
for i, h in enumerate(houses[:20]):
    print(f"  House {i+1:2d}: center=({h['cx']:6.1f}, {h['cy']:6.1f}), footprint={h['w']:4.1f}m x {h['d']:4.1f}m, points={h['pts']:4d}")
