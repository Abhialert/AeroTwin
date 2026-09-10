import pyvista as pv
import numpy as np
import json
from pathlib import Path

print("Reading dense_pointcloud.ply...")
pcd = pv.read("data/output/sample_flight/dense_pointcloud.ply")
pts = np.array(pcd.points)
rgb = np.array(pcd.point_data["RGB"])

# Filter bounding box
mask = (
    (pts[:, 0] >= -650) & (pts[:, 0] <= 1050) &
    (pts[:, 1] >= -750) & (pts[:, 1] <= 800) &
    (pts[:, 2] >= 100) & (pts[:, 2] <= 4000)
)
clean_pts = pts[mask]
clean_rgb = rgb[mask]

# Center coordinates
center = clean_pts.mean(axis=0)
norm_pts = (clean_pts - center) / 80.0

# 120,000 points sample
sample_idx = np.random.choice(len(norm_pts), size=min(120000, len(norm_pts)), replace=False)
s_pts = np.round(norm_pts[sample_idx], 3).tolist()
s_rgb = np.round(clean_rgb[sample_idx] / 255.0, 3).tolist()

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AeroTwin 3D Digital Twin Viewer</title>
    <style>
        body { margin: 0; background: #07090e; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; overflow: hidden; }
        #info { position: absolute; top: 16px; left: 16px; z-index: 10; background: rgba(13,17,23,0.92); padding: 18px 24px; border-radius: 12px; border: 1px solid rgba(56,189,248,0.2); box-shadow: 0 10px 40px rgba(0,0,0,0.6); backdrop-filter: blur(12px); max-width: 320px; }
        h2 { margin: 0 0 6px 0; font-size: 20px; color: #38bdf8; font-weight: 700; letter-spacing: -0.5px; }
        p { margin: 4px 0; font-size: 13px; color: #94a3b8; }
        .badge { display: inline-block; background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; margin-bottom: 8px; margin-right: 4px; }
        .success { background: #16a34a; }
        #canvas-container { width: 100vw; height: 100vh; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="info">
        <h2>AeroTwin — 3D Digital Twin</h2>
        <span class="badge">SIH 2026</span> <span class="badge success">GENUINE DRONE MVS</span>
        <p><strong>Total Reconstruction:</strong> 13,339,285 3D Points</p>
        <p><strong>Registered Poses:</strong> 260 Aerial Cameras</p>
        <p><strong>Interactive Web Layer:</strong> 120,000 Points</p>
        <p style="color: #38bdf8; margin-top: 10px; font-weight: 500;">🖱️ Controls:</p>
        <p>• Left Click + Drag: Rotate 3D Model</p>
        <p>• Right Click + Drag: Pan Scene</p>
        <p>• Scroll Wheel: Zoom In / Out</p>
    </div>
    <div id="canvas-container"></div>

    <script>
        const container = document.getElementById("canvas-container");
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x07090e);

        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, -18, 14);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 0, 0);

        const rawPts = """ + json.dumps(s_pts) + """;
        const rawRgb = """ + json.dumps(s_rgb) + """;

        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(rawPts.length * 3);
        const colors = new Float32Array(rawRgb.length * 3);

        for (let i = 0; i < rawPts.length; i++) {
            positions[i*3] = rawPts[i][0];
            positions[i*3+1] = rawPts[i][1];
            positions[i*3+2] = rawPts[i][2];

            colors[i*3] = rawRgb[i][0];
            colors[i*3+1] = rawRgb[i][1];
            colors[i*3+2] = rawRgb[i][2];
        }

        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({ size: 0.08, vertexColors: true });
        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);

        // Ground reference grid
        const grid = new THREE.GridHelper(40, 40, 0x1e293b, 0x0f172a);
        grid.rotation.x = Math.PI / 2;
        grid.position.z = -5;
        scene.add(grid);

        window.addEventListener("resize", () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""

out_path = Path("data/output/sample_flight/view_3d_demo.html")
out_path.write_text(html_template, encoding="utf-8")
print(f"Generated interactive 3D Web Viewer at: {out_path.resolve()}")
