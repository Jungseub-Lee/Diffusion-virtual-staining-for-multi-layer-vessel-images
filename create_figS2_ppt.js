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
  "Figure_S2_Bridging.pptx"
);

const M = 0.25;

function lbl(sl, text, x, y, w, opts = {}) {
  sl.addText(text, {
    x, y, w: w || 2, h: 0.18, fontSize: opts.fs || 7,
    fontFace: "Arial", color: opts.c || "333333", bold: true, margin: 0,
  });
}

// ==================== SLIDE 1: Algorithm + Scheme ====================
let s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

s1.addText("Supplementary Figure S2: Tangent-guided Gap Bridging Algorithm", {
  x: M, y: 0.1, w: 7.0, h: 0.3,
  fontSize: 11, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

// Panel A label
lbl(s1, "A  Algorithm Overview", M, 0.45, 4, { fs: 9 });

// 5 step boxes — academic style (white bg, thin colored borders)
const steps = [
  { n: "1", t: "Endpoint Detection", d: "Identify skeleton pixels\nwith neighbor count = 1", c: "c0392b" },
  { n: "2", t: "Tangent Estimation", d: "Trace 10–20 px along skeleton\nfrom each endpoint;\ncompute direction vector", c: "e67e22" },
  { n: "3", t: "Candidate Matching", d: "Pair endpoints where:\n• distance < 80 px\n• cos(θ) > cos(60°)\n• vessel mask overlap ≥ 70%", c: "27ae60" },
  { n: "4", t: "Bézier Interpolation", d: "Connect via cubic Bézier:\nP(t) = (1−t)³P₁ + 3(1−t)²tC₁\n        + 3(1−t)t²C₂ + t³P₂\nC₁ = P₁ + 0.4d · t₁", c: "2980b9" },
  { n: "5", t: "Greedy Selection", d: "Score = (cos₁+cos₂)/2\n           − d/dₘₐₓ · 0.2\n           + overlap · 0.2\nBest-first, no endpoint reuse", c: "8e44ad" },
];

const bW = 1.28, bH = 1.55, bGx = 0.1, bGy = 0.15;
const row1Y = 0.68;
// Row 1: steps 1-3
steps.slice(0, 3).forEach((s, i) => {
  const x = M + i * (bW + bGx);
  s1.addShape(pres.shapes.RECTANGLE, {
    x, y: row1Y, w: bW, h: bH,
    fill: { color: "FFFFFF" }, line: { color: s.c, width: 1.5 },
  });
  s1.addText(`Step ${s.n}`, {
    x: x + 0.05, y: row1Y + 0.04, w: bW - 0.1, h: 0.16,
    fontSize: 6.5, fontFace: "Arial", color: s.c, bold: true, margin: 0,
  });
  s1.addText(s.t, {
    x: x + 0.05, y: row1Y + 0.2, w: bW - 0.1, h: 0.18,
    fontSize: 8, fontFace: "Arial", color: "1a1a1a", bold: true, margin: 0,
  });
  s1.addText(s.d, {
    x: x + 0.08, y: row1Y + 0.42, w: bW - 0.16, h: bH - 0.5,
    fontSize: 6.5, fontFace: "Consolas", color: "555555", margin: 0,
  });
});

// Arrows between 1→2, 2→3
[0, 1].forEach(i => {
  const ax = M + (i + 1) * (bW + bGx) - bGx / 2;
  s1.addText("→", {
    x: ax - 0.08, y: row1Y + bH / 2 - 0.1, w: 0.16, h: 0.2,
    fontSize: 12, color: "AAAAAA", align: "center", bold: true,
  });
});

// Row 2: steps 4-5 (centered)
const row2Y = row1Y + bH + bGy;
const row2StartX = M + 0.35;
steps.slice(3).forEach((s, i) => {
  const x = row2StartX + i * (bW + bGx + 0.4);
  s1.addShape(pres.shapes.RECTANGLE, {
    x, y: row2Y, w: bW + 0.3, h: bH,
    fill: { color: "FFFFFF" }, line: { color: s.c, width: 1.5 },
  });
  s1.addText(`Step ${s.n}`, {
    x: x + 0.05, y: row2Y + 0.04, w: bW, h: 0.16,
    fontSize: 6.5, fontFace: "Arial", color: s.c, bold: true, margin: 0,
  });
  s1.addText(s.t, {
    x: x + 0.05, y: row2Y + 0.2, w: bW + 0.2, h: 0.18,
    fontSize: 8, fontFace: "Arial", color: "1a1a1a", bold: true, margin: 0,
  });
  s1.addText(s.d, {
    x: x + 0.08, y: row2Y + 0.42, w: bW + 0.12, h: bH - 0.5,
    fontSize: 6.5, fontFace: "Consolas", color: "555555", margin: 0,
  });
});

// Arrow 3→4 (down)
s1.addText("↓", {
  x: M + 1.5 * (bW + bGx) - 0.08, y: row1Y + bH - 0.02, w: 0.16, h: bGy + 0.04,
  fontSize: 14, color: "AAAAAA", align: "center", bold: true,
});
// Arrow 4→5
const a45x = row2StartX + bW + 0.3 + 0.15;
s1.addText("→", {
  x: a45x, y: row2Y + bH / 2 - 0.1, w: 0.2, h: 0.2,
  fontSize: 12, color: "AAAAAA", align: "center", bold: true,
});

// ---- Panel B: Bridging Scheme ----
const schY = row2Y + bH + 0.25;
lbl(s1, "B  Bridging Scheme", M, schY, 4, { fs: 9 });

// Before box
const schBoxW = 3.2, schBoxH = 1.6;
s1.addShape(pres.shapes.RECTANGLE, {
  x: M + 0.1, y: schY + 0.22, w: schBoxW, h: schBoxH,
  fill: { color: "FAFAFA" }, line: { color: "CCCCCC", width: 1 },
});
s1.addText("Before", {
  x: M + 0.1, y: schY + 0.24, w: schBoxW, h: 0.2,
  fontSize: 8, fontFace: "Arial", color: "999999", bold: true, align: "center",
});
// Skeleton stubs + gap
s1.addShape(pres.shapes.LINE, {
  x: M + 0.4, y: schY + 0.55, w: 1.1, h: 0.7,
  line: { color: "E04040", width: 4 },
});
s1.addShape(pres.shapes.OVAL, {
  x: M + 1.42, y: schY + 1.17, w: 0.12, h: 0.12,
  fill: { color: "FFDD00" }, line: { color: "FFDD00", width: 1 },
});
s1.addShape(pres.shapes.LINE, {
  x: M + 1.5, y: schY + 1.25, w: 0.5, h: 0.3,
  line: { color: "FFDD00", width: 1.5, dashType: "dash", endArrowType: "triangle" },
});
s1.addText("t₁", { x: M + 2.0, y: schY + 1.4, w: 0.2, h: 0.15, fontSize: 7, color: "CCAA00", bold: true });
// Gap
s1.addShape(pres.shapes.LINE, {
  x: M + 1.5, y: schY + 1.25, w: 0.55, h: 0.25,
  line: { color: "BBBBBB", width: 1, dashType: "dash" },
});
s1.addText("gap", { x: M + 1.5, y: schY + 1.55, w: 0.6, h: 0.15, fontSize: 6, color: "AAAAAA", align: "center", italic: true });
// EP2
s1.addShape(pres.shapes.OVAL, {
  x: M + 1.98, y: schY + 1.42, w: 0.12, h: 0.12,
  fill: { color: "FFDD00" }, line: { color: "FFDD00", width: 1 },
});
s1.addShape(pres.shapes.LINE, {
  x: M + 2.04, y: schY + 1.48, w: -0.5, h: -0.3,
  line: { color: "FFDD00", width: 1.5, dashType: "dash", endArrowType: "triangle" },
});
s1.addText("t₂", { x: M + 1.38, y: schY + 1.05, w: 0.2, h: 0.15, fontSize: 7, color: "CCAA00", bold: true });
// Stub 2
s1.addShape(pres.shapes.LINE, {
  x: M + 2.1, y: schY + 1.5, w: 0.9, h: 0.2,
  line: { color: "E04040", width: 4 },
});

// Arrow between boxes
s1.addText("→", {
  x: M + schBoxW + 0.2, y: schY + 0.22 + schBoxH / 2 - 0.12, w: 0.3, h: 0.24,
  fontSize: 16, color: "888888", align: "center", bold: true,
});

// After box
const afterX = M + schBoxW + 0.6;
s1.addShape(pres.shapes.RECTANGLE, {
  x: afterX, y: schY + 0.22, w: schBoxW, h: schBoxH,
  fill: { color: "FAFAFA" }, line: { color: "CCCCCC", width: 1 },
});
s1.addText("After", {
  x: afterX, y: schY + 0.24, w: schBoxW, h: 0.2,
  fontSize: 8, fontFace: "Arial", color: "999999", bold: true, align: "center",
});
// Connected skeleton
s1.addShape(pres.shapes.LINE, {
  x: afterX + 0.3, y: schY + 0.55, w: 1.1, h: 0.7,
  line: { color: "E04040", width: 4 },
});
s1.addShape(pres.shapes.LINE, {
  x: afterX + 1.4, y: schY + 1.25, w: 0.55, h: 0.25,
  line: { color: "00CC88", width: 3, dashType: "sysDash" },
});
s1.addText("Bézier", { x: afterX + 1.35, y: schY + 1.52, w: 0.7, h: 0.15, fontSize: 6, color: "00AA66", align: "center" });
s1.addShape(pres.shapes.LINE, {
  x: afterX + 1.95, y: schY + 1.5, w: 0.9, h: 0.2,
  line: { color: "E04040", width: 4 },
});

// ==================== SLIDE 2: Panel C — Step-by-step results ====================
let s2 = pres.addSlide();
s2.background = { color: "FFFFFF" };

s2.addText("C  Bridging Results: Per-layer Before → Bridging → After → Combined", {
  x: M, y: 0.08, w: 7.0, h: 0.25,
  fontSize: 10, fontFace: "Arial", color: "1a1a1a", bold: true,
});

// Layout: 2 rows (R skel, B skel) × 3 columns (before/bridging/after) + combined column
const pW = 1.45, pH = 1.18, pGx = 0.08, pGy = 0.08;
const gridX = M + 0.05;
const colLabels = ["Before", "Bridging", "After"];
const rowLabels = [
  { text: "R skeleton\n(Bottom)", color: "CC3333" },
  { text: "B skeleton\n(Top)", color: "3366CC" },
];

// Column headers
colLabels.forEach((t, i) => {
  s2.addText(t, {
    x: gridX + 0.45 + i * (pW + pGx), y: 0.35, w: pW, h: 0.18,
    fontSize: 7, fontFace: "Arial", color: "666666", bold: true, align: "center",
  });
});
s2.addText("Combined", {
  x: gridX + 0.45 + 3 * (pW + pGx) + 0.15, y: 0.35, w: pW, h: 0.18,
  fontSize: 7, fontFace: "Arial", color: "666666", bold: true, align: "center",
});

const rFiles = ["s2_r_before.png", "s2_r_bridging.png", "s2_r_after.png"];
const bFiles = ["s2_b_before.png", "s2_b_bridging.png", "s2_b_after.png"];

// Row labels
rowLabels.forEach((rl, row) => {
  const yBase = 0.55 + row * (pH + pGy);
  s2.addText(rl.text, {
    x: gridX - 0.05, y: yBase + pH / 2 - 0.15, w: 0.5, h: 0.3,
    fontSize: 6.5, fontFace: "Arial", color: rl.color, bold: true, align: "center",
  });
});

// R row
rFiles.forEach((f, col) => {
  const x = gridX + 0.45 + col * (pW + pGx);
  s2.addImage({ path: path.join(P, f), x, y: 0.55, w: pW, h: pH });
  s2.addShape(pres.shapes.RECTANGLE, { x, y: 0.55, w: pW, h: pH, line: { color: "CC3333", width: 1 } });
});

// B row
bFiles.forEach((f, col) => {
  const x = gridX + 0.45 + col * (pW + pGx);
  const y = 0.55 + pH + pGy;
  s2.addImage({ path: path.join(P, f), x, y, w: pW, h: pH });
  s2.addShape(pres.shapes.RECTANGLE, { x, y, w: pW, h: pH, line: { color: "3366CC", width: 1 } });
});

// Combined (spans both rows)
const combX = gridX + 0.45 + 3 * (pW + pGx) + 0.15;
const combH = 2 * pH + pGy;
s2.addImage({ path: path.join(P, "s2_combined.png"), x: combX, y: 0.55, w: pW, h: combH });
s2.addShape(pres.shapes.RECTANGLE, { x: combX, y: 0.55, w: pW, h: combH, line: { color: "555555", width: 1.5 } });

// Arrows between columns
[0, 1].forEach(i => {
  const ax = gridX + 0.45 + (i + 1) * (pW + pGx) - pGx / 2;
  [0.55 + pH / 2, 0.55 + pH + pGy + pH / 2].forEach(ay => {
    s2.addText("→", {
      x: ax - 0.06, y: ay - 0.08, w: 0.12, h: 0.16,
      fontSize: 10, color: "CCCCCC", align: "center", bold: true,
    });
  });
});
// Arrow to combined
const acx = combX - 0.12;
s2.addText("→", {
  x: acx - 0.06, y: 0.55 + combH / 2 - 0.08, w: 0.2, h: 0.16,
  fontSize: 10, color: "CCCCCC", align: "center", bold: true,
});

// Full image with GT overlay
const fullY = 0.55 + combH + 0.25;
lbl(s2, "Full image: Combined skeleton on GT overlay", M, fullY, 5, { fs: 8 });
const fullW = 5.5, fullH = fullW * 0.8;
const fullX = (7.5 - fullW) / 2;
s2.addImage({ path: path.join(P, "s2_combined_gt.png"), x: fullX, y: fullY + 0.22, w: fullW, h: fullH });

// Legend
s2.addText([
  { text: "Red ", options: { color: "CC3333", bold: true, fontSize: 6.5 } },
  { text: "R skel   ", options: { fontSize: 6.5 } },
  { text: "Blue ", options: { color: "3366CC", bold: true, fontSize: 6.5 } },
  { text: "B skel   ", options: { fontSize: 6.5 } },
  { text: "Orange ", options: { color: "FFB432", bold: true, fontSize: 6.5 } },
  { text: "R bridge   ", options: { fontSize: 6.5 } },
  { text: "Teal ", options: { color: "32FFB4", bold: true, fontSize: 6.5 } },
  { text: "B bridge   ", options: { fontSize: 6.5 } },
  { text: "Purple ", options: { color: "B464DC", bold: true, fontSize: 6.5 } },
  { text: "Overlap   ", options: { fontSize: 6.5 } },
  { text: "○ ", options: { color: "CCAA00", bold: true, fontSize: 7 } },
  { text: "EP   ", options: { fontSize: 6.5 } },
  { text: "○ ", options: { color: "00AAAA", bold: true, fontSize: 7 } },
  { text: "Conn.   ", options: { fontSize: 6.5 } },
  { text: "○ ", options: { color: "CC00CC", bold: true, fontSize: 7 } },
  { text: "JN", options: { fontSize: 6.5 } },
], {
  x: M, y: fullY + fullH + 0.28, w: 7.0, h: 0.18,
  fontFace: "Arial", color: "777777", align: "center",
});

// Metrics
s2.addText("R: EP=13(real)+3(conn), JN=15  |  B: EP=13(real)+6(conn), JN=9  |  Combined: EP(real)=26, JN=24", {
  x: M, y: fullY + fullH + 0.48, w: 7.0, h: 0.15,
  fontSize: 6.5, fontFace: "Consolas", color: "888888", align: "center",
});

pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
