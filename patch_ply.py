import os

results_path = r"c:\Users\ABHISHEK\Desktop\AeroTwin\results.html"

with open(results_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add PLYLoader
if "PLYLoader" not in html:
    html = html.replace(
        '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>',
        '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>\n<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>'
    )

# 2. Add loadRealModel function and replace the init calls
loader_script = """
let realMesh = null;
let realCloud = null;
let colmapCenter = new THREE.Vector3(0,0,0);

function loadRealModel() {
  return new Promise((resolve) => {
    const loader = new THREE.PLYLoader();
    // Try to load the real surface mesh
    loader.load('/data/output/sample_flight/aerotwin_surface_mesh.ply', (geometry) => {
      geometry.computeVertexNormals();
      
      const hasColors = geometry.hasAttribute('color');
      const material = new THREE.MeshStandardMaterial({ 
          color: hasColors ? 0xffffff : 0x8b9db5, 
          roughness: 0.7, 
          vertexColors: hasColors,
          side: THREE.DoubleSide
      });
      
      realMesh = new THREE.Mesh(geometry, material);
      
      // Don't move the mesh, just calculate its center to aim the camera
      geometry.computeBoundingBox();
      geometry.boundingBox.getCenter(colmapCenter);
      
      scene.add(realMesh);
      soB.push(realMesh); // Add to soB so raycasting/measurements work
      
      // Point camera at center of mesh
      ctrl.target.copy(colmapCenter);
      camera.position.set(colmapCenter.x, colmapCenter.y + 120, colmapCenter.z + 200);
      ctrl.update();
      
      document.getElementById('statTri').textContent = ((geometry.index ? geometry.index.count/3 : geometry.attributes.position.count/3)/1000).toFixed(1) + 'K';
      
      resolve(true);
    }, undefined, (e) => {
      console.error('Failed to load PLY, falling back', e);
      resolve(false);
    });
  });
}
"""

if "function loadRealModel" not in html:
    html = html.replace("// Lights", loader_script + "\n// Lights")


# Replace the build procedural calls inside init()
init_original = """  setupLights();
  sub.textContent='Building terrain & roads...'; await slp(80); buildGround();
  sub.textContent='Generating urban structures...'; await slp(80); buildBuildings();
  sub.textContent='Placing vegetation...'; await slp(80); buildTrees();
  sub.textContent='Loading 260 COLMAP camera poses...'; await slp(80); buildFlight();"""

init_new = """  setupLights();
  sub.textContent='Loading real 3D Mesh from COLMAP...';
  const success = await loadRealModel();
  if (!success) {
      sub.textContent='Failed to load real data, using procedural fallback...'; await slp(80);
      buildGround(); buildBuildings(); buildTrees(); buildFlight();
  } else {
      // Fake flight path around the mesh center
      buildFlightReal(colmapCenter);
  }"""

html = html.replace(init_original, init_new)


flight_real = """
function buildFlightReal(center) {
  const poses = [];
  for (let i = 0; i < 260; i++) {
    const a = (i/260) * Math.PI * 4;
    const r = 80 + Math.sin(i*0.1)*20;
    poses.push(new THREE.Vector3(center.x + Math.cos(a)*r, center.y + 50, center.z + Math.sin(a)*r));
  }
  const curve = new THREE.CatmullRomCurve3(poses, false, 'catmullrom', 0.5); soFC = curve;
  soFP = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(1200)),
    new THREE.LineBasicMaterial({ color:0xa855f7, transparent:true, opacity:0.55 }));
  scene.add(soFP);

  // Drone model
  const dg = new THREE.Group();
  dg.add(new THREE.Mesh(new THREE.BoxGeometry(2,0.3,2), new THREE.MeshStandardMaterial({ color:0xdddddd, roughness:0.3, metalness:0.6 })));
  const aM = new THREE.MeshStandardMaterial({ color:0x333333, roughness:0.8 });
  const pM = new THREE.MeshStandardMaterial({ color:0x111111, roughness:0.4, transparent:true, opacity:0.75 });
  [-1,1].forEach(sx => [-1,1].forEach(sz => {
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.07,0.07,1.8,6), aM);
    arm.rotation.z = Math.PI/2; arm.position.set(sx,0,sz); dg.add(arm);
    const prop = new THREE.Mesh(new THREE.CylinderGeometry(0.9,0.9,0.04,8), pM);
    prop.position.set(sx*1.8, 0.12, sz*1.8); prop.name = 'prop'; dg.add(prop);
  }));
  const gim = new THREE.Mesh(new THREE.BoxGeometry(0.5,0.4,0.6), new THREE.MeshStandardMaterial({ color:0x222222, roughness:0.5, metalness:0.8 }));
  gim.position.y = -0.35; dg.add(gim);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.16,0.12,0.35,8), new THREE.MeshStandardMaterial({ color:0x112233, roughness:0.05, metalness:0.95 }));
  lens.rotation.x = Math.PI/2; lens.position.set(0,-0.35,-0.4); dg.add(lens);
  dg.position.copy(poses[0]); scene.add(dg); soDrn = dg;
}
"""

if "function buildFlightReal" not in html:
    html = html.replace("// Raycaster / measurement", flight_real + "\n// Raycaster / measurement")

# Override resetCam to point to the real mesh center
html = html.replace("camera.position.set(0,120,200); ctrl.target.set(0,0,0);", "if(colmapCenter) { camera.position.set(colmapCenter.x, colmapCenter.y+120, colmapCenter.z+200); ctrl.target.copy(colmapCenter); } else { camera.position.set(0,120,200); ctrl.target.set(0,0,0); }")

with open(results_path, "w", encoding="utf-8") as f:
    f.write(html)

print("results.html patched for real PLY loading!")
