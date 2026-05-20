const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "A4", width: 7.5, height: 10.0 });
pres.layout = "A4";
pres.author = "Depth-aware Vessel Analysis";
pres.title = "Figure: Multi-color Depth-aware Vessel Analysis Pipeline";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\paper_panels_depth"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_Depth_Pipeline_v3.pptx"
);

const margin = 0.25;
const lfs = 7;

function addLabel(slide, text, x, y, w, opts = {}) {
  slide.addText(text, {
    x, y, w: w || 2, h: 0.18,
    fontSize: opts.fontSize || lfs, fontFace: "Arial",
    color: opts.color || "333333", bold: true, margin: 0,
  });
}

// ==================== SLIDE 1: Pipeline ====================
let s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

s1.addText("Depth-aware Vessel Network Analysis Pipeline", {
  x: margin, y: 0.1, w: 7.0, h: 0.3,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

// 4-column grid
const imgW = 1.58;
const imgH = 1.3;
const gap = 0.12;

function addRow(slide, y, panels) {
  panels.forEach((p, i) => {
    const x = margin + i * (imgW + gap);
    addLabel(slide, p.label, x, y, imgW);
    slide.addImage({ path: path.join(P, p.file), x, y: y + 0.18, w: imgW, h: imgH });
  });
}

// Row 1: Input (y=0.45)
addRow(s1, 0.45, [
  { file: "a_bf.png", label: "(a) Brightfield" },
  { file: "b_gt.png", label: "(b) Multi-color GT" },
  { file: "c_r_ch.png", label: "(c) R channel" },
  { file: "d_b_ch.png", label: "(d) B channel" },
]);

s1.addShape(pres.shapes.LINE, {
  x: margin, y: 2.05, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
});

// Row 2: Separation + Skeleton (y=2.15)
addRow(s1, 2.15, [
  { file: "e_dom.png", label: "(e) R-B dominance" },
  { file: "f_r_sk.png", label: "(f) R skeleton" },
  { file: "g_b_sk.png", label: "(g) B skeleton" },
  { file: "h_rb_overlay.png", label: "(h) R+B overlay" },
]);

s1.addShape(pres.shapes.LINE, {
  x: margin, y: 3.75, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
});

// Row 3: Bridging result (y=3.85)
addRow(s1, 3.85, [
  { file: "i_single.png", label: "(i) Single GT skel" },
  { file: "j_r_br.png", label: "(j) R bridged" },
  { file: "k_b_br.png", label: "(k) B bridged" },
  { file: "l_comb.png", label: "(l) R+B combined" },
]);

s1.addShape(pres.shapes.LINE, {
  x: margin, y: 5.45, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
});

// Row 4: Bridge overview (before → after → final)
addLabel(s1, "(m) Bridging: Before → After → Final", margin, 5.55, 7.0);
s1.addImage({
  path: path.join(P, "bridge_overview.png"),
  x: margin, y: 5.75, w: 7.0, h: 1.85,
});

// Legend
s1.addText([
  { text: "Red ", options: { color: "CC3333", bold: true, fontSize: 6 } },
  { text: "= R skel   ", options: { fontSize: 6 } },
  { text: "Blue ", options: { color: "3366CC", bold: true, fontSize: 6 } },
  { text: "= B skel   ", options: { fontSize: 6 } },
  { text: "Yellow ● ", options: { color: "CCCC00", bold: true, fontSize: 6 } },
  { text: "= EP + tangent   ", options: { fontSize: 6 } },
  { text: "Green ", options: { color: "00CC44", bold: true, fontSize: 6 } },
  { text: "= Bridge path", options: { fontSize: 6 } },
], {
  x: margin, y: 7.65, w: 7.0, h: 0.2,
  fontFace: "Arial", color: "555555",
});

// Row 5: Zoom details (3 best examples)
addLabel(s1, "(n) Bridge Zoom Details (endpoint tangent → Bézier curve)", margin, 7.9, 7.0);
const zW = 1.6;
const zH = 1.6;
const zGap = 0.15;
const zStart = (7.5 - 4 * zW - 3 * zGap) / 2;
for (let i = 0; i < 4; i++) {
  s1.addImage({
    path: path.join(P, `zoom_bridge_${i + 1}.png`),
    x: zStart + i * (zW + zGap), y: 8.1, w: zW, h: zH,
  });
}

// ==================== SLIDE 2: Bridging Algorithm ====================
let s2 = pres.addSlide();
s2.background = { color: "FFFFFF" };

s2.addText("Tangent-guided Gap Bridging Algorithm", {
  x: margin, y: 0.1, w: 7.0, h: 0.3,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

// Algorithm diagram
s2.addImage({
  path: path.join(P, "bridge_algorithm.png"),
  x: 0.3, y: 0.5, w: 6.9, h: 4.3,
});

// More zoom examples below
addLabel(s2, "Bridge Examples (R and B channels)", margin, 5.0, 7.0,
         { fontSize: 9 });
const zW2 = 1.5;
const zH2 = 1.5;
for (let i = 0; i < 4; i++) {
  s2.addImage({
    path: path.join(P, `zoom_bridge_${i + 3}.png`),
    x: margin + i * (zW2 + 0.25), y: 5.3, w: zW2, h: zH2,
  });
}

// Algorithm description text
s2.addText([
  { text: "Key Criteria for Bridge Matching:\n", options: { bold: true, fontSize: 8, breakLine: true } },
  { text: "1. Distance constraint: ", options: { bold: true, fontSize: 7, breakLine: false } },
  { text: "endpoint pair distance < 80 px\n", options: { fontSize: 7, breakLine: true } },
  { text: "2. Angular alignment: ", options: { bold: true, fontSize: 7, breakLine: false } },
  { text: "tangent vectors align with connection direction (cos θ > cos 60°)\n", options: { fontSize: 7, breakLine: true } },
  { text: "3. Vessel mask validation: ", options: { bold: true, fontSize: 7, breakLine: false } },
  { text: "≥70% of bridge path lies within vessel mask\n", options: { fontSize: 7, breakLine: true } },
  { text: "4. Bézier interpolation: ", options: { bold: true, fontSize: 7, breakLine: false } },
  { text: "cubic Bézier with control points from tangent vectors\n", options: { fontSize: 7, breakLine: true } },
  { text: "5. Greedy selection: ", options: { bold: true, fontSize: 7, breakLine: false } },
  { text: "best-scoring pairs first, no endpoint reuse", options: { fontSize: 7 } },
], {
  x: 0.4, y: 7.0, w: 6.7, h: 2.5,
  fontFace: "Arial", color: "333333",
  border: { type: "solid", pt: 0.5, color: "DDDDDD" },
  fill: { color: "F8F8F8" },
});

// ==================== SLIDE 3: Quantification ====================
let s3 = pres.addSlide();
s3.background = { color: "FFFFFF" };

s3.addText("Vessel Network Quantification", {
  x: margin, y: 0.1, w: 7.0, h: 0.3,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

// Chart 1
addLabel(s3, "(a) Single GT vs Multi GT — Normalized (Single GT = 1.0, N=215, top 80%)",
         margin, 0.5, 7.0, { fontSize: 8 });
s3.addImage({
  path: path.join(P, "chart1_single_vs_multi.png"),
  x: 0.5, y: 0.75, w: 6.5, h: 3.2,
});

s3.addShape(pres.shapes.LINE, {
  x: margin, y: 4.1, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
});

// Chart 2
addLabel(s3, "(b) Multi-color GT vs Virtual Staining Models — Normalized (Multi GT = 1.0)",
         margin, 4.2, 7.0, { fontSize: 8 });
s3.addImage({
  path: path.join(P, "chart2_multi_vs_models.png"),
  x: 0.15, y: 4.45, w: 7.2, h: 3.4,
});

// Summary table
const tblY = 8.1;
addLabel(s3, "(c) Normalized Metrics Summary", margin, tblY - 0.2, 5, { fontSize: 8 });

const headerOpts = { bold: true, color: "FFFFFF", fill: { color: "34495e" } };
const bestOpts = { color: "27ae60", bold: true };
const tblData = [
  [
    { text: "Model", options: headerOpts },
    { text: "Length", options: headerOpts },
    { text: "EP", options: headerOpts },
    { text: "JN", options: headerOpts },
  ],
  [{ text: "GT (Multi)", options: { bold: true } }, { text: "1.00" }, { text: "1.00" }, { text: "1.00" }],
  [{ text: "PBBDM", options: { bold: true, color: "2980b9" } },
   { text: "0.95", options: bestOpts }, { text: "0.93", options: bestOpts }, { text: "0.78", options: bestOpts }],
  [{ text: "LBBDM", options: { bold: true, color: "8e44ad" } },
   { text: "1.00", options: bestOpts }, { text: "0.65" }, { text: "0.77" }],
  [{ text: "WGANGP" }, { text: "0.80" }, { text: "0.88" }, { text: "0.48" }],
  [{ text: "Pix2pix" }, { text: "0.81" }, { text: "0.83" }, { text: "0.42" }],
  [{ text: "LSGAN" }, { text: "0.70" }, { text: "0.77" }, { text: "0.39" }],
];

s3.addTable(tblData, {
  x: margin, y: tblY, w: 7.0,
  fontSize: 7, fontFace: "Arial",
  border: { type: "solid", pt: 0.5, color: "CCCCCC" },
  colW: [1.8, 1.6, 1.6, 1.6],
  rowH: 0.2,
  autoPage: false,
});

s3.addText("PBBDM achieves closest match to GT across all multi-color metrics", {
  x: margin, y: tblY + 1.55, w: 6, h: 0.2,
  fontSize: 7, fontFace: "Arial", color: "2980b9", bold: true, italic: true, margin: 0,
});

// Save
pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
