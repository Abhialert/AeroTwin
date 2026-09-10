import pyvista as pv
import numpy as np
import json
from pathlib import Path

print("Reading mesh geometry and dense cloud RGB colors...")
m = pv.read("data/output/sample_flight/aerotwin_surface_mesh.ply")
orig = pv.read("data/output/sample_flight/dense_pointcloud.ply")

pts = np.array(m.points)

# Find nearest RGB color from original dense point cloud using KDTree
from scipy.spatial import cKDTree
tree = cKDTree(np.array(orig.points)[::50]) # subsample original for quick query
orig_rgb = np.array(orig.point_data["RGB"])[::50]
_, idxs = tree.query(pts)
rgb = orig_rgb[idxs]

# Center coordinates for WebGL
center = pts.mean(axis=0)
norm_pts = np.round((pts - center) / 60.0, 3).tolist()
norm_rgb = np.round(rgb / 255.0, 3).tolist()
faces_raw = m.faces.reshape(-1, 4)[:, 1:].tolist()

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AeroTwin 3D Digital Twin Viewer</title>
    <style>
        body { margin: 0; background: #07090e; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; overflow: hidden; }
        #info { position: absolute; top: 16px; left: 16px; z-index: 10; background: rgba(13,17,23,0.92); padding: 18px 24px; border-radius: 12px; border: 1px solid rgba(56,189,248,0.25); box-shadow: 0 10px 40px rgba(0,0,0,0.6); backdrop-filter: blur(12px); max-width: 320px; }
        h2 { margin: 0 0 6px 0; font-size: 20px; color: #38bdf8; font-weight: 700; letter-spacing: -0.5px; }
        p { margin: 4px 0; font-size: 13px; color: #94a3b8; }
        .badge { display: inline-block; background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; margin-bottom: 8px; margin-right: 4px; }
        .success { background: #16a34a; }
        .purple { background: #7c3aed; }
        .btn-group { margin-top: 12px; display: flex; gap: 8px; }
        button { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.4); padding: 7px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s; }
        button:hover { background: #38bdf8; color: #000; }
        #canvas-container { width: 100vw; height: 100vh; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="info">
        <h2>AeroTwin — Pinpoint Twin</h2>
        <span class="badge">SIH 2026</span> <span class="badge success">13.3M MVS</span> <span class="badge purple">POLYGON MESH</span>
        <p><strong>Total Dense Points:</strong> 13,339,285 Points</p>
        <p><strong>Mesh Triangles:</strong> 237,335 Faces</p>
        <p><strong>Features Isolated:</strong> Roads, Roofs, Poles/Walls</p>
        
        <div class="btn-group">
            <button id="btn-points">Point Cloud</button>
            <button id="btn-mesh">Surface Mesh</button>
            <button id="btn-wire">Wireframe</button>
        </div>

        <p style="color: #38bdf8; margin-top: 12px; font-weight: 500;">🖱️ Controls:</p>
        <p>• Left Click: Rotate | Right Click: Pan | Scroll: Zoom</p>
    </div>
    <div id="canvas-container"></div>

    <script>
        const container = document.getElementById("canvas-container");
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x07090e);

        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, -22, 18);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(20, -30, 40);
        scene.add(dirLight);

        const rawPts = """ + json.dumps(norm_pts) + """;
        const rawRgb = """ + json.dumps(norm_rgb) + """;
        const rawFaces = """ + json.dumps(faces_raw) + """;

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

        const indices = [];
        for (let i = 0; i < rawFaces.length; i++) {
            indices.push(rawFaces[i][0], rawFaces[i][1], rawFaces[i][2]);
        }

        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();

        // Point Cloud
        const pointMat = new THREE.PointsMaterial({ size: 0.08, vertexColors: true });
        const pointCloud = new THREE.Points(geometry, pointMat);
        scene.add(pointCloud);

        // Surface Mesh
        const meshMat = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.6,
            metalness: 0.1,
            side: THREE.DoubleSide
        });
        const surfaceMesh = new THREE.Mesh(geometry, meshMat);
        surfaceMesh.visible = false;
        scene.add(surfaceMesh);

        // Grid
        const grid = new THREE.GridHelper(50, 50, 0x1e293b, 0x0f172a);
        grid.rotation.x = Math.PI / 2;
        grid.position.z = -8;
        scene.add(grid);

        // Toggle buttons
        document.getElementById("btn-points").onclick = () => {
            pointCloud.visible = true;
            surfaceMesh.visible = false;
        };
        document.getElementById("btn-mesh").onclick = () => {
            pointCloud.visible = false;
            surfaceMesh.visible = true;
            meshMat.wireframe = false;
        };
        document.getElementById("btn-wire").onclick = () => {
            pointCloud.visible = false;
            surfaceMesh.visible = true;
            meshMat.wireframe = true;
        };

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

out_file = Path("data/output/sample_flight/view_3d_demo.html")
out_file.write_text(html_template, encoding="utf-8")
print(f"Successfully updated {out_file} with Point Cloud, Polygon Surface Mesh, and Wireframe inspection modes.")
