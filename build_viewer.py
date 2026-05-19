"""Build self-contained HTML viewer with base64-embedded images."""
import base64
from pathlib import Path

img_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output\individual")
out_html = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output\viewer.html")

samples = ["1-0-512", "1-5-512", "1-10-512", "1-15-512"]
suffixes_step1 = [("original","Original GT"), ("mask_bottom","Bottom Mask (Red)"), ("mask_middle","Middle Mask (Yellow)"), ("mask_top","Top Mask (Blue)")]
suffixes_step2 = [("overlay_bottom","Bottom Overlay"), ("overlay_middle","Middle Overlay"), ("overlay_top","Top Overlay"), ("overlay_all","All Layers Combined")]
suffixes_step3 = [("skel_bottom","Bottom Skeleton (Red)"), ("skel_middle","Middle Skeleton (Yellow)"), ("skel_top","Top Skeleton (Blue)"), ("skel_all","All Skeletons (Depth-colored)")]

def to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Pre-load all images
img_data = {}
for sid in samples:
    for suffixes in [suffixes_step1, suffixes_step2, suffixes_step3]:
        for sfx, _ in suffixes:
            key = f"{sid}_{sfx}"
            p = img_dir / f"{key}.png"
            img_data[key] = to_b64(p)

# Build HTML
def img_cell(sid, sfx, cap):
    b64 = img_data[f"{sid}_{sfx}"]
    return f'''<div class="cell">
      <img src="data:image/png;base64,{b64}" onclick="openLB(this,'{cap}')" alt="{cap}">
      <div class="cap">{cap}</div>
    </div>'''

sample_blocks = ""
for sid in samples:
    step1 = "\n".join(img_cell(sid, s, c) for s, c in suffixes_step1)
    step2 = "\n".join(img_cell(sid, s, c) for s, c in suffixes_step2)
    step3 = "\n".join(img_cell(sid, s, c) for s, c in suffixes_step3)
    sample_blocks += f'''
    <div class="sample-block">
      <div class="sample-header">Sample: {sid}</div>
      <div class="step-label">Step 1 — Original & HSV Mask Separation</div>
      <div class="grid4">{step1}</div>
      <div class="step-label">Step 2 — Color Overlay on Dimmed Original</div>
      <div class="grid4">{step2}</div>
      <div class="step-label">Step 3 — Layer Skeletons (thick) + Depth-aware Combined</div>
      <div class="grid4">{step3}</div>
    </div>'''

html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Depth Layer Viewer</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#111;color:#ddd;font-family:'Segoe UI',sans-serif;padding:20px}}
h1{{text-align:center;font-size:22px;margin-bottom:4px}}
.sub{{text-align:center;color:#888;font-size:13px;margin-bottom:18px}}
.legend{{display:flex;gap:20px;justify-content:center;margin-bottom:22px;padding:10px;background:#1a1a1a;border-radius:8px;max-width:750px;margin-left:auto;margin-right:auto}}
.legend-item{{display:flex;align-items:center;gap:5px;font-size:12px}}
.dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0}}
.sample-block{{max-width:1400px;margin:0 auto 35px;background:#1a1a1a;border-radius:10px;padding:18px;border:1px solid #333}}
.sample-header{{font-size:18px;font-weight:bold;margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid #333}}
.step-label{{font-size:12px;color:#999;text-transform:uppercase;letter-spacing:1.5px;margin:14px 0 6px}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:6px}}
.cell{{position:relative}}
.cell img{{width:100%;display:block;border-radius:4px;cursor:pointer;transition:opacity .15s}}
.cell img:hover{{opacity:.85}}
.cell .cap{{font-size:10px;color:#888;text-align:center;margin-top:2px}}
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.95);z-index:9999;justify-content:center;align-items:center;cursor:zoom-out;flex-direction:column}}
#lb.open{{display:flex}}
#lb img{{max-width:90vw;max-height:85vh;object-fit:contain;border-radius:6px}}
#lb .lc{{color:#aaa;font-size:13px;margin-top:8px}}
#lb .x{{position:absolute;top:12px;right:20px;font-size:30px;color:#fff;cursor:pointer}}
</style></head><body>
<h1>Depth-Aware Color Layer Separation & Skeletonization</h1>
<p class="sub">GT Multi-color Depth-encoded Fluorescence — HSV Layer Analysis</p>
<div class="legend">
  <div class="legend-item"><div class="dot" style="background:#ff5050"></div> Bottom (Red) — Lower</div>
  <div class="legend-item"><div class="dot" style="background:#ffff3c"></div> Middle (Yellow) — Overlap</div>
  <div class="legend-item"><div class="dot" style="background:#508cff"></div> Top (Blue) — Upper</div>
</div>
{sample_blocks}
<div id="lb" onclick="closeLB()">
  <span class="x" onclick="closeLB()">&times;</span>
  <img id="li" src="">
  <div class="lc" id="lc"></div>
</div>
<script>
function openLB(el,c){{event.stopPropagation();document.getElementById("li").src=el.src;document.getElementById("lc").textContent=c;document.getElementById("lb").classList.add("open")}}
function closeLB(){{document.getElementById("lb").classList.remove("open")}}
document.addEventListener("keydown",e=>{{if(e.key==="Escape")closeLB()}});
</script></body></html>'''

out_html.write_text(html, encoding="utf-8")
print(f"Saved: {out_html} ({len(html)//1024} KB)")
