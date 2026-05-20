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
  "Figure_S2_Bridging_v2.pptx"
);

const M = 0.2;

function lbl(sl, text, x, y, w, opts = {}) {
  sl.addText(text, {
    x, y, w: w || 3, h: 0.2, fontSize: opts.fs || 8,
    fontFace: opts.ff || "Arial", color: opts.c || "222222", bold: true, margin: 0,
  });
}

// ==================== SLIDE 1: A (Algorithm) + B (Scheme) ====================
let s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

s1.addText("Supplementary Figure S2", {
  x: M, y: 0.08, w: 7.1, h: 0.25,
  fontSize: 11, fontFace: "Arial", color: "1a1a1a", bold: true,
});
s1.addText("Tangent-guided Gap Bridging Algorithm for Depth-separated Skeletons", {
  x: M, y: 0.3, w: 7.1, h: 0.18,
  fontSize: 8, fontFace: "Arial", color: "777777", italic: true,
});

// --- A: Algorithm flowchart (similar to figS2_combined A panel style) ---
lbl(s1, "A", M, 0.52, 0.3, { fs: 12, c: "000000" });

const steps = [
  { n: "1", t: "Endpoint\nDetection", d: "neighbor\ncount = 1", c: "c0392b", bg: "FDEDEC" },
  { n: "2", t: "Tangent\nEstimation", d: "trace 10\u201320 px\nalong skeleton", c: "e67e22", bg: "FEF5E7" },
  { n: "3", t: "Candidate\nMatching", d: "d < 80 px\ncos\u03B8 > cos60\u00B0\noverlap \u2265 70%", c: "27ae60", bg: "EAFAF1" },
  { n: "4", t: "B\u00E9zier\nInterpolation", d: "cubic B\u00E9zier\nC\u2081 = P\u2081+0.4d\u00B7t\u2081", c: "2980b9", bg: "EBF5FB" },
  { n: "5", t: "Score &\nSelect", d: "greedy\nbest-first\nno reuse", c: "8e44ad", bg: "F4ECF7" },
];

const bW = 1.25, bH = 1.15, bGx = 0.07;
const bStartX = M + 0.25;
const bY = 0.72;

steps.forEach((s, i) => {
  const x = bStartX + i * (bW + bGx);
  // Colored background box
  s1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y: bY, w: bW, h: bH,
    fill: { color: s.bg }, line: { color: s.c, width: 1.5 }, rectRadius: 0.06,
  });
  // Step number
  s1.addText(`Step ${s.n}`, {
    x: x + 0.05, y: bY + 0.03, w: bW - 0.1, h: 0.14,
    fontSize: 6, fontFace: "Arial", color: s.c, bold: true, margin: 0,
  });
  // Title
  s1.addText(s.t, {
    x: x + 0.05, y: bY + 0.18, w: bW - 0.1, h: 0.32,
    fontSize: 8, fontFace: "Arial", color: "1a1a1a", bold: true, margin: 0,
  });
  // Description
  s1.addText(s.d, {
    x: x + 0.06, y: bY + 0.52, w: bW - 0.12, h: bH - 0.58,
    fontSize: 6.5, fontFace: "Consolas", color: "666666", margin: 0,
  });
  // Arrow between boxes
  if (i < 4) {
    s1.addText("\u2192", {
      x: x + bW - 0.02, y: bY + bH / 2 - 0.1, w: bGx + 0.04, h: 0.2,
      fontSize: 11, color: "BBBBBB", align: "center", bold: true,
    });
  }
});

// --- B: Bridging Scheme (3-stage: Before → Bridging → After) ---
const schY = bY + bH + 0.2;
lbl(s1, "B", M, schY, 0.3, { fs: 12, c: "000000" });

const schBoxW = 2.1, schBoxH = 1.8;
const schGap = 0.2;
const schStartX = (7.5 - 3 * schBoxW - 2 * schGap) / 2;

// Box titles and styles
const schBoxes = [
  { title: "Before", bg: "FAFAFA" },
  { title: "Bridging", bg: "FFFDE7" },
  { title: "After", bg: "F1F8E9" },
];

schBoxes.forEach((b, i) => {
  const x = schStartX + i * (schBoxW + schGap);
  const y = schY + 0.22;
  s1.addShape(pres.shapes.RECTANGLE, {
    x, y, w: schBoxW, h: schBoxH,
    fill: { color: b.bg }, line: { color: "CCCCCC", width: 1 },
  });
  s1.addText(b.title, {
    x, y: y + 0.02, w: schBoxW, h: 0.2,
    fontSize: 8, fontFace: "Arial", color: "999999", bold: true, align: "center",
  });
});

// Before: broken skeleton stubs + tangent arrows
const sx0 = schStartX;
const sy0 = schY + 0.22;
// Stub 1
s1.addShape(pres.shapes.LINE, { x: sx0+0.25, y: sy0+0.4, w: 0.75, h: 0.55, line: { color: "E04040", width: 4 } });
// EP1
s1.addShape(pres.shapes.OVAL, { x: sx0+0.93, y: sy0+0.88, w: 0.12, h: 0.12, fill: { color: "FFDD00" } });
// t1 arrow
s1.addShape(pres.shapes.LINE, { x: sx0+1.0, y: sy0+0.95, w: 0.4, h: 0.25, line: { color: "FFDD00", width: 1.5, dashType: "dash", endArrowType: "triangle" } });
s1.addText("t\u2081", { x: sx0+1.4, y: sy0+1.05, w: 0.2, h: 0.15, fontSize: 7, color: "CCAA00", bold: true });
// Gap
s1.addShape(pres.shapes.LINE, { x: sx0+1.0, y: sy0+0.95, w: 0.5, h: 0.2, line: { color: "BBBBBB", width: 1, dashType: "dash" } });
s1.addText("gap", { x: sx0+1.05, y: sy0+1.2, w: 0.5, h: 0.14, fontSize: 6, color: "AAAAAA", align: "center", italic: true });
// EP2
s1.addShape(pres.shapes.OVAL, { x: sx0+1.43, y: sy0+1.08, w: 0.12, h: 0.12, fill: { color: "FFDD00" } });
// t2 arrow
s1.addShape(pres.shapes.LINE, { x: sx0+1.49, y: sy0+1.14, w: -0.4, h: -0.25, line: { color: "FFDD00", width: 1.5, dashType: "dash", endArrowType: "triangle" } });
s1.addText("t\u2082", { x: sx0+0.95, y: sy0+0.73, w: 0.2, h: 0.15, fontSize: 7, color: "CCAA00", bold: true });
// Stub 2
s1.addShape(pres.shapes.LINE, { x: sx0+1.55, y: sy0+1.15, w: 0.35, h: 0.25, line: { color: "E04040", width: 4 } });

// Bridging: stubs + green bridge
const bx0 = schStartX + schBoxW + schGap;
s1.addShape(pres.shapes.LINE, { x: bx0+0.25, y: sy0+0.4, w: 0.75, h: 0.55, line: { color: "E04040", width: 4 } });
// Bridge path (green/teal, thicker)
s1.addShape(pres.shapes.LINE, { x: bx0+1.0, y: sy0+0.95, w: 0.5, h: 0.2, line: { color: "00CC88", width: 3, dashType: "sysDash" } });
s1.addText("B\u00E9zier", { x: bx0+0.95, y: sy0+1.2, w: 0.6, h: 0.14, fontSize: 6, color: "00AA66", align: "center" });
// EP dots
s1.addShape(pres.shapes.OVAL, { x: bx0+0.93, y: sy0+0.88, w: 0.12, h: 0.12, fill: { color: "FFDD00" } });
s1.addShape(pres.shapes.OVAL, { x: bx0+1.43, y: sy0+1.08, w: 0.12, h: 0.12, fill: { color: "FFDD00" } });
s1.addShape(pres.shapes.LINE, { x: bx0+1.55, y: sy0+1.15, w: 0.35, h: 0.25, line: { color: "E04040", width: 4 } });

// After: fully connected (no gap, no markers)
const ax0 = schStartX + 2 * (schBoxW + schGap);
s1.addShape(pres.shapes.LINE, { x: ax0+0.25, y: sy0+0.4, w: 0.75, h: 0.55, line: { color: "E04040", width: 4 } });
s1.addShape(pres.shapes.LINE, { x: ax0+1.0, y: sy0+0.95, w: 0.5, h: 0.2, line: { color: "E04040", width: 4 } });
s1.addShape(pres.shapes.LINE, { x: ax0+1.5, y: sy0+1.15, w: 0.35, h: 0.25, line: { color: "E04040", width: 4 } });

// Arrows between scheme boxes
[0, 1].forEach(i => {
  const arX = schStartX + (i + 1) * (schBoxW + schGap) - schGap / 2;
  s1.addText("\u2192", {
    x: arX - 0.08, y: sy0 + schBoxH / 2 - 0.1, w: 0.16, h: 0.2,
    fontSize: 14, color: "888888", align: "center", bold: true,
  });
});

// ==================== SLIDE 2: C (Bridging Results) ====================
let s2 = pres.addSlide();
s2.background = { color: "FFFFFF" };

lbl(s2, "C  Depth-separated Skeleton Bridging Results", M, 0.06, 7, { fs: 10 });

// Row 0: Context (GT, R channel, B channel)
const iW = 1.1, iH = 0.88, iG = 0.06;
const ctx_y = 0.28;
const ctx_labels = ["Multi-color GT", "R channel", "B channel"];
const ctx_files = ["s2c_gt.png", "s2c_r_channel.png", "s2c_b_channel.png"];
const ctx_borders = ["666666", "CC3333", "3366CC"];

ctx_labels.forEach((t, i) => {
  const x = M + i * (iW + iG);
  s2.addText(t, { x, y: ctx_y, w: iW, h: 0.14, fontSize: 6, fontFace: "Arial", color: ctx_borders[i], bold: true, align: "center" });
  s2.addImage({ path: path.join(P, ctx_files[i]), x, y: ctx_y + 0.14, w: iW, h: iH });
  s2.addShape(pres.shapes.RECTANGLE, { x, y: ctx_y + 0.14, w: iW, h: iH, line: { color: ctx_borders[i], width: 1 } });
});

// Row 1-2: R skel (Before→Bridging→After), B skel (Before→Bridging→After)
const stageW = 1.55, stageH = 1.24, stageG = 0.06;
const labelW = 0.4;
const stageStartX = M + labelW;
const stageLabels = ["Before", "Bridging", "After", "Combined"];

// Stage column headers
stageLabels.forEach((t, i) => {
  s2.addText(t, {
    x: stageStartX + i * (stageW + stageG), y: 1.28, w: stageW, h: 0.16,
    fontSize: 6.5, fontFace: "Arial", color: "666666", bold: true, align: "center",
  });
});

const rows = [
  { label: "R skel\n(Bottom)", color: "CC3333", files: ["s2c_r_before.png", "s2c_r_bridging.png", "s2c_r_after.png"] },
  { label: "B skel\n(Top)", color: "3366CC", files: ["s2c_b_before.png", "s2c_b_bridging.png", "s2c_b_after.png"] },
];

rows.forEach((row, ri) => {
  const yBase = 1.46 + ri * (stageH + 0.06);
  // Row label
  s2.addText(row.label, {
    x: M - 0.05, y: yBase + stageH / 2 - 0.15, w: labelW, h: 0.3,
    fontSize: 6.5, fontFace: "Arial", color: row.color, bold: true, align: "center",
  });
  // 3 stage images
  row.files.forEach((f, ci) => {
    const x = stageStartX + ci * (stageW + stageG);
    s2.addImage({ path: path.join(P, f), x, y: yBase, w: stageW, h: stageH });
    s2.addShape(pres.shapes.RECTANGLE, { x, y: yBase, w: stageW, h: stageH, line: { color: row.color, width: 1 } });
  });
  // Arrows
  [0, 1].forEach(ai => {
    const arX = stageStartX + (ai + 1) * (stageW + stageG) - stageG / 2;
    s2.addText("\u2192", {
      x: arX - 0.04, y: yBase + stageH / 2 - 0.08, w: 0.08, h: 0.16,
      fontSize: 9, color: "CCCCCC", align: "center", bold: true,
    });
  });
});

// Combined column (spans both rows)
const combX = stageStartX + 3 * (stageW + stageG);
const combY = 1.46;
const combH = 2 * stageH + 0.06;
s2.addImage({ path: path.join(P, "s2c_combined.png"), x: combX, y: combY, w: stageW, h: combH });
s2.addShape(pres.shapes.RECTANGLE, { x: combX, y: combY, w: stageW, h: combH, line: { color: "555555", width: 1.5 } });

// Arrow to combined
s2.addText("\u2192", {
  x: combX - stageG / 2 - 0.04, y: combY + combH / 2 - 0.08, w: 0.12, h: 0.16,
  fontSize: 9, color: "CCCCCC", align: "center", bold: true,
});

// Full image GT overlay
const fullY = combY + combH + 0.15;
lbl(s2, "Full image overlay", M, fullY, 3, { fs: 7 });
const fullW = 4.8, fullH = fullW * 0.8;
const fullX = (7.5 - fullW) / 2;
s2.addImage({ path: path.join(P, "s2c_full_gt.png"), x: fullX, y: fullY + 0.18, w: fullW, h: fullH });

// ==================== SLIDE 3: Crop examples ====================
let s3 = pres.addSlide();
s3.background = { color: "FFFFFF" };

lbl(s3, "C (cont.)  Zoomed Examples", M, 0.08, 5, { fs: 10 });

const cropTags = ["crop1", "crop2", "crop3"];
const cropTitles = ["Region 1 (left vessels)", "Region 2 (crossing)", "Region 3 (top area)"];
const cW = 2.1, cH = 2.1, cG = 0.12;
const cStartX = (7.5 - 3 * cW - 2 * cG) / 2;

// Row labels
["GT", "Skeleton", "GT + Skeleton"].forEach((t, i) => {
  const y = 0.5 + i * (cH + 0.06);
  s3.addText(t, {
    x: M - 0.15, y: y + cH / 2 - 0.1, w: 0.5, h: 0.2,
    fontSize: 7, fontFace: "Arial", color: "777777", bold: true, rotate: 270, align: "center",
  });
});

cropTags.forEach((tag, col) => {
  const x = cStartX + col * (cW + cG);
  s3.addText(cropTitles[col], {
    x, y: 0.32, w: cW, h: 0.16,
    fontSize: 7, fontFace: "Arial", color: "555555", bold: true, align: "center",
  });

  const suffixes = ["_gt.png", "_skel.png", "_overlay.png"];
  suffixes.forEach((sf, ri) => {
    const y = 0.5 + ri * (cH + 0.06);
    s3.addImage({ path: path.join(P, `s2c_${tag}${sf}`), x, y, w: cW, h: cW });
  });
});

// Legend
s3.addText([
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
  { text: "\u25CB ", options: { color: "CCAA00", bold: true, fontSize: 7 } },
  { text: "EP   ", options: { fontSize: 6.5 } },
  { text: "\u25CB ", options: { color: "00AAAA", bold: true, fontSize: 7 } },
  { text: "Conn.   ", options: { fontSize: 6.5 } },
  { text: "\u25CB ", options: { color: "CC00CC", bold: true, fontSize: 7 } },
  { text: "JN", options: { fontSize: 6.5 } },
], {
  x: M, y: 7.2, w: 7.1, h: 0.2,
  fontFace: "Arial", color: "777777", align: "center",
});

pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
