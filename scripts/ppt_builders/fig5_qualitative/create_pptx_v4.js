const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "A4", width: 7.5, height: 10.0 });
pres.layout = "A4";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\paper_panels_depth"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_Depth_Pipeline_v5.pptx"
);

const M = 0.25; // margin
const lfs = 7;

function lbl(slide, text, x, y, w, opts = {}) {
  slide.addText(text, {
    x, y, w: w || 2, h: 0.18, fontSize: opts.fs || lfs,
    fontFace: "Arial", color: opts.c || "333333", bold: true, margin: 0,
  });
}

function sep(slide, y) {
  slide.addShape(pres.shapes.LINE, {
    x: M, y, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
  });
}

// ==================== SLIDE 1: Pipeline ====================
let s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

s1.addText("Depth-aware Vessel Network Analysis Pipeline", {
  x: M, y: 0.08, w: 7.0, h: 0.28,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

const iW = 1.58, iH = 1.3, g = 0.12;
function addRow(sl, y, panels) {
  panels.forEach((p, i) => {
    const x = M + i * (iW + g);
    lbl(sl, p.l, x, y, iW);
    sl.addImage({ path: path.join(P, p.f), x, y: y + 0.18, w: iW, h: iH });
  });
}

// Row 1: Input
addRow(s1, 0.4, [
  { f: "a_bf.png", l: "(a) Brightfield" },
  { f: "b_gt.png", l: "(b) Multi-color GT" },
  { f: "c_r_ch.png", l: "(c) R channel" },
  { f: "d_b_ch.png", l: "(d) B channel" },
]);
sep(s1, 1.98);

// Row 2: Separation
addRow(s1, 2.06, [
  { f: "e_dom.png", l: "(e) R-B dominance" },
  { f: "f_r_sk.png", l: "(f) R skeleton" },
  { f: "g_b_sk.png", l: "(g) B skeleton" },
  { f: "h_rb_overlay.png", l: "(h) R+B overlay" },
]);
sep(s1, 3.64);

// Row 3: Skeleton result (GT overlay)
addRow(s1, 3.72, [
  { f: "i_single.png", l: "(i) Single GT skel" },
  { f: "j_r_br.png", l: "(j) R bridged" },
  { f: "k_b_br.png", l: "(k) B bridged" },
  { f: "l_comb.png", l: "(l) R+B combined" },
]);

// Row 3b: Skeleton result (no background)
const nbY = 3.72 + iH + 0.22;
addRow(s1, nbY, [
  { f: "i_single_nb.png", l: "(i') Single skel only" },
  { f: "j_r_br_nb.png", l: "(j') R skel only" },
  { f: "k_b_br_nb.png", l: "(k') B skel only" },
  { f: "l_comb_nb.png", l: "(l') R+B skel only" },
]);
sep(s1, nbY + iH + 0.1);

// Legend at bottom of slide 1
const legendY = nbY + iH + 0.15;
s1.addText([
  { text: "Red ", options: { color: "CC3333", bold: true, fontSize: 5.5 } },
  { text: "R skel  ", options: { fontSize: 5.5 } },
  { text: "Yellow ", options: { color: "CCAA00", bold: true, fontSize: 5.5 } },
  { text: "Y skel  ", options: { fontSize: 5.5 } },
  { text: "Blue ", options: { color: "3366CC", bold: true, fontSize: 5.5 } },
  { text: "B skel  ", options: { fontSize: 5.5 } },
  { text: "Purple ", options: { color: "B464DC", bold: true, fontSize: 5.5 } },
  { text: "R+B overlap  ", options: { fontSize: 5.5 } },
  { text: "Magenta ", options: { color: "CC00CC", bold: true, fontSize: 5.5 } },
  { text: "JN  ", options: { fontSize: 5.5 } },
  { text: "Yellow○ ", options: { color: "CC9900", bold: true, fontSize: 5.5 } },
  { text: "EP  ", options: { fontSize: 5.5 } },
  { text: "Cyan○ ", options: { color: "00AAAA", bold: true, fontSize: 5.5 } },
  { text: "Conn.EP", options: { fontSize: 5.5 } },
], { x: M, y: legendY, w: 7.0, h: 0.15, fontFace: "Arial", color: "555555" });

// ==================== SLIDE 1b: Gap Bridging ====================
let s1b = pres.addSlide();
s1b.background = { color: "FFFFFF" };

s1b.addText("Tangent-guided Gap Bridging", {
  x: M, y: 0.08, w: 7.0, h: 0.28,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

// (m) Gap Bridging Process - 3 stages
lbl(s1b, "(m) Gap Bridging Process", M, 0.4, 6);
const mW = 2.15, mH = 1.7, mG = 0.18;
const mStart = (7.5 - 3 * mW - 2 * mG) / 2;
["m_before.png", "m_bridging.png", "m_after.png"].forEach((f, i) => {
  s1b.addImage({ path: path.join(P, f), x: mStart + i * (mW + mG), y: 0.6, w: mW, h: mH });
});
["Before (filtered)", "Bridging", "After (connected)"].forEach((t, i) => {
  s1b.addText(t, {
    x: mStart + i * (mW + mG), y: 2.32, w: mW, h: 0.15,
    fontSize: 6, fontFace: "Arial", color: "666666", align: "center", italic: true,
  });
});
[0, 1].forEach(i => {
  const ax = mStart + (i + 1) * (mW + mG) - mG / 2;
  s1b.addText("\u2192", {
    x: ax - 0.15, y: 0.6 + mH / 2 - 0.12, w: 0.3, h: 0.24,
    fontSize: 14, fontFace: "Arial", color: "AAAAAA", align: "center", bold: true,
  });
});

sep(s1b, 2.55);

// (n) Bridge Zoom Detail
lbl(s1b, "(n) Bridge Zoom Detail (Before → Bridging → After)", M, 2.65, 7);
const zW = 0.72, zH = 0.72, zG = 0.04;
const triGap = 0.15;
const tripletW = 3 * zW + 2 * zG;
const totalTriW = 3 * tripletW + 2 * triGap;
const triStart = (7.5 - totalTriW) / 2;

const zoomExamples = [
  { idx: 1, ch: "B", c: "3366CC" },
  { idx: 2, ch: "R", c: "CC3333" },
  { idx: 3, ch: "R", c: "CC3333" },
];

zoomExamples.forEach((ex, tri) => {
  const tripX = triStart + tri * (tripletW + triGap);
  ["before", "bridging", "after"].forEach((stage, si) => {
    s1b.addImage({
      path: path.join(P, `n_zoom_${ex.idx}_${stage}.png`),
      x: tripX + si * (zW + zG), y: 2.95, w: zW, h: zH,
    });
  });
  s1b.addText(`${ex.ch} bridge #${tri + 1}`, {
    x: tripX, y: 3.7, w: tripletW, h: 0.14,
    fontSize: 5.5, fontFace: "Arial", color: ex.c, bold: true, align: "center",
  });
  if (tri === 0) {
    ["before", "bridging", "after"].forEach((t, si) => {
      s1b.addText(t, {
        x: tripX + si * (zW + zG), y: 2.85, w: zW, h: 0.11,
        fontSize: 4.5, fontFace: "Arial", color: "888888", align: "center", italic: true,
      });
    });
  }
});

// Bridge color legend
s1b.addText([
  { text: "Orange ", options: { color: "FFB432", bold: true, fontSize: 6 } },
  { text: "R bridge  ", options: { fontSize: 6 } },
  { text: "Teal ", options: { color: "32FFB4", bold: true, fontSize: 6 } },
  { text: "B bridge  ", options: { fontSize: 6 } },
], { x: M, y: 3.9, w: 7.0, h: 0.15, fontFace: "Arial", color: "555555" });

// ==================== SLIDE 2: Algorithm (PPT text/boxes) ====================
let s2 = pres.addSlide();
s2.background = { color: "16213e" };

s2.addText("Tangent-guided Gap Bridging Algorithm", {
  x: M, y: 0.1, w: 7.0, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: "FFFFFF", bold: true, align: "center",
});

// 5 Step boxes
const steps = [
  { n: "1", t: "Endpoint Detection", d: "Find skeleton pixels with\nneighbor count = 1",
    x: 0.2, y: 0.6, c: "e74c3c" },
  { n: "2", t: "Tangent Estimation", d: "Trace back 10-20 px along\nskeleton → compute direction",
    x: 2.65, y: 0.6, c: "f39c12" },
  { n: "3", t: "Candidate Matching", d: "EP pair distance < 80 px\ncos(θ) > cos(60°)\nvessel mask overlap ≥ 70%",
    x: 5.1, y: 0.6, c: "2ecc71" },
  { n: "4", t: "Bézier Interpolation", d: "P(t) = (1-t)³P₁ + 3(1-t)²tC₁\n      + 3(1-t)t²C₂ + t³P₂\nC₁ = P₁ + 0.4d·t₁",
    x: 1.2, y: 3.0, c: "3498db" },
  { n: "5", t: "Score & Select", d: "score = (cos₁+cos₂)/2\n         - d/dmax · 0.2\n         + vessel_ratio · 0.2\nGreedy: best-first, no reuse",
    x: 4.2, y: 3.0, c: "9b59b6" },
];

steps.forEach(s => {
  const bW = 2.2, bH = 1.8;
  // Box background
  s2.addShape(pres.shapes.RECTANGLE, {
    x: s.x, y: s.y, w: bW, h: bH,
    fill: { color: s.c, transparency: 85 },
    line: { color: s.c, width: 2 },
  });
  // Step number
  s2.addText(`Step ${s.n}`, {
    x: s.x + 0.08, y: s.y + 0.05, w: 1, h: 0.2,
    fontSize: 7, fontFace: "Arial", color: s.c, bold: true, margin: 0,
  });
  // Title
  s2.addText(s.t, {
    x: s.x + 0.08, y: s.y + 0.25, w: bW - 0.15, h: 0.25,
    fontSize: 9, fontFace: "Arial", color: "FFFFFF", bold: true, margin: 0,
  });
  // Description
  s2.addText(s.d, {
    x: s.x + 0.1, y: s.y + 0.55, w: bW - 0.2, h: bH - 0.65,
    fontSize: 7, fontFace: "Consolas", color: "CCCCCC", margin: 0,
  });
});

// Arrows between steps (horizontal)
[{ x1: 2.4, x2: 2.65, y: 1.5 }, { x1: 4.85, x2: 5.1, y: 1.5 }].forEach(a => {
  s2.addShape(pres.shapes.LINE, {
    x: a.x1, y: a.y, w: a.x2 - a.x1, h: 0,
    line: { color: "888888", width: 2, endArrowType: "triangle" },
  });
});
// Down arrow
s2.addShape(pres.shapes.LINE, {
  x: 3.75, y: 2.4, w: 0, h: 0.6,
  line: { color: "888888", width: 2, endArrowType: "triangle" },
});
// Horizontal between 4 and 5
s2.addShape(pres.shapes.LINE, {
  x: 3.4, y: 3.9, w: 0.8, h: 0,
  line: { color: "888888", width: 2, endArrowType: "triangle" },
});

// Scheme diagram: before → after connection
const schY = 5.2;
s2.addText("Bridging Scheme", {
  x: M, y: schY - 0.3, w: 7, h: 0.25,
  fontSize: 10, fontFace: "Arial", color: "FFFFFF", bold: true, align: "center",
});

// Before: two broken skeleton segments
// Left box: "Before"
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: schY, w: 3.0, h: 2.2,
  fill: { color: "FFFFFF", transparency: 92 },
  line: { color: "555555", width: 1 },
});
s2.addText("Before Bridging", {
  x: 0.5, y: schY, w: 3.0, h: 0.25,
  fontSize: 8, fontFace: "Arial", color: "AAAAAA", align: "center", bold: true,
});
// Skeleton stub 1 (line going right-down, then gap, then continuing)
s2.addShape(pres.shapes.LINE, {
  x: 0.8, y: schY + 0.5, w: 1.0, h: 0.6,
  line: { color: "FF5555", width: 4 },
});
// EP1 circle
s2.addShape(pres.shapes.OVAL, {
  x: 1.7, y: schY + 1.0, w: 0.15, h: 0.15,
  fill: { color: "FFFF00" }, line: { color: "FFFF00", width: 1 },
});
// Tangent arrow 1
s2.addShape(pres.shapes.LINE, {
  x: 1.8, y: schY + 1.1, w: 0.5, h: 0.3,
  line: { color: "FFFF00", width: 2, endArrowType: "triangle" },
});
s2.addText("t₁", {
  x: 2.3, y: schY + 1.25, w: 0.3, h: 0.2,
  fontSize: 8, color: "FFFF00", bold: true,
});

// Gap (dashed)
s2.addShape(pres.shapes.LINE, {
  x: 1.8, y: schY + 1.1, w: 0.6, h: 0.3,
  line: { color: "666666", width: 1, dashType: "dash" },
});

// EP2 circle
s2.addShape(pres.shapes.OVAL, {
  x: 2.3, y: schY + 1.3, w: 0.15, h: 0.15,
  fill: { color: "FFFF00" }, line: { color: "FFFF00", width: 1 },
});
// Tangent arrow 2
s2.addShape(pres.shapes.LINE, {
  x: 2.35, y: schY + 1.4, w: -0.5, h: -0.3,
  line: { color: "FFFF00", width: 2, endArrowType: "triangle" },
});
s2.addText("t₂", {
  x: 1.7, y: schY + 0.95, w: 0.3, h: 0.2,
  fontSize: 8, color: "FFFF00", bold: true,
});

// Skeleton stub 2
s2.addShape(pres.shapes.LINE, {
  x: 2.4, y: schY + 1.4, w: 0.8, h: 0.5,
  line: { color: "FF5555", width: 4 },
});

s2.addText("gap", {
  x: 1.8, y: schY + 1.6, w: 0.8, h: 0.2,
  fontSize: 7, color: "999999", align: "center", italic: true,
});

// Right box: "After"
s2.addShape(pres.shapes.RECTANGLE, {
  x: 4.0, y: schY, w: 3.0, h: 2.2,
  fill: { color: "FFFFFF", transparency: 92 },
  line: { color: "555555", width: 1 },
});
s2.addText("After Bridging", {
  x: 4.0, y: schY, w: 3.0, h: 0.25,
  fontSize: 8, fontFace: "Arial", color: "AAAAAA", align: "center", bold: true,
});
// Connected skeleton
s2.addShape(pres.shapes.LINE, {
  x: 4.3, y: schY + 0.5, w: 1.0, h: 0.6,
  line: { color: "FF5555", width: 4 },
});
// Bridge (green)
s2.addShape(pres.shapes.LINE, {
  x: 5.3, y: schY + 1.1, w: 0.6, h: 0.3,
  line: { color: "00FF88", width: 3, dashType: "sysDash" },
});
s2.addText("Bézier bridge", {
  x: 5.1, y: schY + 1.45, w: 1.2, h: 0.2,
  fontSize: 6.5, color: "00FF88", align: "center",
});
s2.addShape(pres.shapes.LINE, {
  x: 5.9, y: schY + 1.4, w: 0.8, h: 0.5,
  line: { color: "FF5555", width: 4 },
});

// Arrow between boxes
s2.addShape(pres.shapes.LINE, {
  x: 3.5, y: schY + 1.1, w: 0.5, h: 0,
  line: { color: "FFFFFF", width: 2, endArrowType: "triangle" },
});

// Zoom examples
const exY = 7.7;
s2.addText("Bridge Examples", {
  x: M, y: exY - 0.25, w: 7, h: 0.2,
  fontSize: 9, fontFace: "Arial", color: "FFFFFF", bold: true, align: "center",
});
// 3 zoom pairs (before/after)
const ezW = 0.9, ezH = 0.9, ezG = 0.06;
const exLabels = ["B bridge", "R bridge", "R bridge"];
const exColors = ["6699FF", "FF6666", "FF6666"];
for (let i = 0; i < 3; i++) {
  const px = M + 0.3 + i * (2 * ezW + ezG + 0.4);
  s2.addImage({ path: path.join(P, `n_zoom_${i + 1}_before.png`), x: px, y: exY, w: ezW, h: ezH });
  s2.addImage({ path: path.join(P, `n_zoom_${i + 1}_after.png`), x: px + ezW + ezG, y: exY, w: ezW, h: ezH });
  s2.addText(exLabels[i], {
    x: px, y: exY + ezH + 0.02, w: 2 * ezW + ezG, h: 0.12,
    fontSize: 5.5, fontFace: "Arial", color: exColors[i],
    bold: true, align: "center",
  });
}

// ==================== SLIDE 3: Quantification ====================
let s3 = pres.addSlide();
s3.background = { color: "FFFFFF" };

s3.addText("Vessel Network Quantification", {
  x: M, y: 0.08, w: 7.0, h: 0.28,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

lbl(s3, "(a) Single GT vs Multi GT (N=215, top 80%)", M, 0.4, 7, { fs: 8 });
s3.addImage({ path: path.join(P, "chart1_single_vs_multi.png"), x: 0.5, y: 0.65, w: 6.5, h: 3.0 });

// Table for chart a values
s3.addTable([
  [{ text: "", options: { bold: true, fill: { color: "f0f0f0" } } },
   { text: "Length", options: { bold: true, fill: { color: "f0f0f0" } } },
   { text: "EP", options: { bold: true, fill: { color: "f0f0f0" } } },
   { text: "JN", options: { bold: true, fill: { color: "f0f0f0" } } }],
  [{ text: "Single GT" }, { text: "1.00" }, { text: "1.00" }, { text: "1.00" }],
  [{ text: "Multi GT" }, { text: "1.26" }, { text: "0.80" }, { text: "1.21" }],
], {
  x: 5.0, y: 0.65, w: 2.3, fontSize: 6.5, fontFace: "Arial",
  border: { type: "solid", pt: 0.3, color: "DDDDDD" },
  colW: [0.7, 0.5, 0.5, 0.5], rowH: 0.17, autoPage: false,
});

sep(s3, 3.8);

lbl(s3, "(b) Virtual Staining Models vs Multi GT", M, 3.88, 7, { fs: 8 });
s3.addImage({ path: path.join(P, "chart2_multi_vs_models.png"), x: 0.15, y: 4.1, w: 7.2, h: 3.1 });

sep(s3, 7.35);

// Summary table
lbl(s3, "(c) Normalized Metrics (Multi GT = 1.0)", M, 7.4, 5, { fs: 8 });
const hdr = { bold: true, color: "FFFFFF", fill: { color: "34495e" } };
const best = { color: "27ae60", bold: true };
s3.addTable([
  [{ text: "Model", options: hdr }, { text: "Length", options: hdr },
   { text: "EP", options: hdr }, { text: "JN", options: hdr }],
  [{ text: "PBBDM", options: { bold: true, color: "2980b9" } },
   { text: "0.95", options: best }, { text: "0.93", options: best }, { text: "0.78", options: best }],
  [{ text: "LBBDM", options: { bold: true, color: "8e44ad" } },
   { text: "1.00", options: best }, { text: "0.65" }, { text: "0.77" }],
  [{ text: "WGANGP" }, { text: "0.80" }, { text: "0.88" }, { text: "0.48" }],
  [{ text: "Pix2pix" }, { text: "0.81" }, { text: "0.83" }, { text: "0.42" }],
  [{ text: "LSGAN" }, { text: "0.70" }, { text: "0.77" }, { text: "0.39" }],
], {
  x: M, y: 7.6, w: 7.0, fontSize: 7, fontFace: "Arial",
  border: { type: "solid", pt: 0.5, color: "CCCCCC" },
  colW: [1.8, 1.6, 1.6, 1.6], rowH: 0.2, autoPage: false,
});

s3.addText("PBBDM achieves the closest match to GT across all multi-color depth metrics", {
  x: M, y: 8.9, w: 6, h: 0.18,
  fontSize: 7, fontFace: "Arial", color: "2980b9", bold: true, italic: true, margin: 0,
});

// ==================== SLIDE 4: Extra Samples ====================
let s4 = pres.addSlide();
s4.background = { color: "FFFFFF" };

s4.addText("Multi-sample Depth-aware Skeleton Comparison", {
  x: M, y: 0.08, w: 7.0, h: 0.28,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

const sids = ["1-19-716", "9-15-512", "16-18-716", "18-19-512", "1-16-512"];
const colW4 = 2.15, colH4 = 1.75, colG = 0.12;
const startX4 = (7.5 - 3 * colW4 - 2 * colG) / 2;

// Column headers
["Skel R (bridged)", "Skel B (bridged)", "R+B Combined"].forEach((t, i) => {
  s4.addText(t, {
    x: startX4 + i * (colW4 + colG), y: 0.38, w: colW4, h: 0.18,
    fontSize: 7, fontFace: "Arial", color: i === 0 ? "CC3333" : i === 1 ? "3366CC" : "333333",
    bold: true, align: "center",
  });
});

sids.forEach((sid, row) => {
  const y = 0.6 + row * (colH4 + 0.22);
  // SID label
  s4.addText(sid, {
    x: 0.02, y: y + colH4 / 2 - 0.1, w: 0.5, h: 0.2,
    fontSize: 5.5, fontFace: "Consolas", color: "999999", rotate: 270, align: "center",
  });
  // 3 panels
  ["r", "b", "comb"].forEach((t, col) => {
    s4.addImage({
      path: path.join(P, `extra_${sid}_${t}.png`),
      x: startX4 + col * (colW4 + colG), y,
      w: colW4, h: colH4,
    });
  });
});

// Save
pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
