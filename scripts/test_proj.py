import numpy as np
from pathlib import Path

# Load camera intrinsics
with open('jobs/demo/demo/sparse/1/cameras.txt') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p = line.split()
        W, H = int(p[2]), int(p[3])
        f_val, cx, cy = float(p[4]), float(p[5]), float(p[6])
        break

print(f"Camera: {W}x{H}, f={f_val}, cx={cx}, cy={cy}")

def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y],
    ])

# Load images
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

# Test projection with dense_display.ply points
with open('jobs/demo/demo/dense_display.ply', 'rb') as f:
    while True:
        l = f.readline().decode('ascii', 'replace').strip()
        if l == 'end_header': break
    pts_display = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 3)

# Test projection of 1000 sample points from dense_display into frame_00025.jpg
sub = pts_display[::250][:1000]
cam = cams['frame_00025.jpg']
p_cam = (sub @ cam['R'].T) + cam['tvec']
in_front = p_cam[:, 2] > 0.1
u = f_val * p_cam[:, 0] / p_cam[:, 2] + cx
v = f_val * p_cam[:, 1] / p_cam[:, 2] + cy
in_frame = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

print(f"dense_display: {np.sum(in_front)}/1000 in front, {np.sum(in_frame)}/1000 in frame")
if np.sum(in_front) > 0:
    print(f"  p_cam z: min={p_cam[:, 2].min():.1f}, med={np.median(p_cam[:, 2]):.1f}, max={p_cam[:, 2].max():.1f}")
    print(f"  u: min={u[in_front].min():.1f}, med={np.median(u[in_front]):.1f}, max={u[in_front].max():.1f}")
    print(f"  v: min={v[in_front].min():.1f}, med={np.median(v[in_front]):.1f}, max={v[in_front].max():.1f}")

# Now test projection with dense_pointcloud_preview.ply points
with open('jobs/demo/demo/dense_pointcloud_preview.ply', 'rb') as f:
    while True:
        l = f.readline().decode('ascii', 'replace').strip()
        if l == 'end_header': break
    raw = f.read(1000 * 27)
dtype = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
])
arr = np.frombuffer(raw, dtype=dtype)
pts_preview = np.column_stack([arr['x'], arr['y'], arr['z']])

p_cam2 = (pts_preview @ cam['R'].T) + cam['tvec']
in_front2 = p_cam2[:, 2] > 0.1
u2 = f_val * p_cam2[:, 0] / p_cam2[:, 2] + cx
v2 = f_val * p_cam2[:, 1] / p_cam2[:, 2] + cy
in_frame2 = in_front2 & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)

print(f"dense_pointcloud_preview: {np.sum(in_front2)}/1000 in front, {np.sum(in_frame2)}/1000 in frame")
if np.sum(in_front2) > 0:
    print(f"  p_cam z: min={p_cam2[:, 2].min():.1f}, med={np.median(p_cam2[:, 2]):.1f}, max={p_cam2[:, 2].max():.1f}")
    print(f"  u: min={u2[in_front2].min():.1f}, med={np.median(u2[in_front2]):.1f}, max={u2[in_front2].max():.1f}")
    print(f"  v: min={v2[in_front2].min():.1f}, med={np.median(v2[in_front2]):.1f}, max={v2[in_front2].max():.1f}")
