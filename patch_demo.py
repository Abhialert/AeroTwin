import os
import re

demo_path = r"c:\Users\ABHISHEK\Desktop\AeroTwin\data\output\sample_flight\demo.html"
results_path = r"c:\Users\ABHISHEK\Desktop\AeroTwin\results.html"

with open(demo_path, "r", encoding="utf-8") as f:
    html = f.read()

# Add IDs to elements we want to update dynamically
html = html.replace('>SITE_001_SAMPLE<', ' id="top-project-name">SITE_001_SAMPLE<')
html = html.replace('>13.3M pts &middot; 260 cams<', ' id="top-stats">13.3M pts &middot; 260 cams<')

html = html.replace('<div class="scval">13.3M</div>', '<div class="scval" id="stat-pts">13.3M</div>')
html = html.replace('<div class="scval">260/278</div>', '<div class="scval" id="stat-cams">260/278</div>')
html = html.replace('<div class="scval">1.083</div>', '<div class="scval" id="stat-scale">1.083</div>')
html = html.replace('<div class="scval">31.4s</div>', '<div class="scval" id="stat-time">31.4s</div>')
html = html.replace('<div class="scval">1,621</div>', '<div class="scval" id="stat-inliers">1,621</div>')

# Replace the semantic bars container with an ID so we can inject new bars
sem_bars_start = html.find('<div class="sembars">')
sem_bars_end = html.find('</div>\n    <div style="margin-top:8px;font-size:9px;', sem_bars_start)
if sem_bars_start != -1 and sem_bars_end != -1:
    html = html[:sem_bars_start] + '<div class="sembars" id="dyn-sembars"></div>' + html[sem_bars_end+6:]

# Add the fetch logic inside init()
init_injection = """
async function init() {
  try {
    const res = await fetch('/api/results');
    const data = await res.json();
    
    // Update Top Bar
    document.getElementById('top-project-name').innerText = data.project || 'AeroTwin';
    const pts = data.colmap_report?.sparse?.num_points3D || 0;
    const cams = data.colmap_report?.sparse?.num_images_registered || 0;
    const total_cams = data.colmap_report?.num_input_frames || 0;
    document.getElementById('top-stats').innerHTML = `${(pts/1000).toFixed(1)}k pts &middot; ${cams} cams`;
    
    // Update Right Panel Stats
    document.getElementById('stat-pts').innerText = pts.toLocaleString();
    document.getElementById('stat-cams').innerText = `${cams}/${total_cams}`;
    
    const scale = data.georef_report?.scale_m_per_unit || 1.083;
    document.getElementById('stat-scale').innerText = scale.toFixed(3);
    
    const total_time = data.colmap_report?.total_time_sec || 0;
    document.getElementById('stat-time').innerText = total_time.toFixed(1) + 's';
    
    // Semantics
    const semDiv = document.getElementById('dyn-sembars');
    if (semDiv && data.semantic_avg) {
        semDiv.innerHTML = '';
        const sorted = Object.entries(data.semantic_avg).sort((a,b)=>b[1]-a[1]);
        const colors = {'Sky':'var(--cyan)', 'Vegetation':'var(--grn)', 'Building/Roof':'#f97316', 'Road/Ground':'#a78bfa', 'Unknown':'#5c6bc0'};
        
        sorted.forEach(([cls, pct]) => {
            const col = colors[cls] || '#94a3b8';
            semDiv.innerHTML += `
              <div class="semrow">
                <div class="semlbl">${cls}</div>
                <div class="semtrack"><div class="semfill" style="width:${pct.toFixed(1)}%;background:${col}"></div></div>
                <div class="sempct">${pct.toFixed(2)}%</div>
              </div>
            `;
        });
    }
    
  } catch(e) {
    console.error('API load error', e);
  }
  
  const sub=document.getElementById('lsub');
"""
html = html.replace("async function init() {\n  const sub=document.getElementById('lsub');", init_injection)

# Write to results.html
with open(results_path, "w", encoding="utf-8") as f:
    f.write(html)

print("results.html patched successfully!")
