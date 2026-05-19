const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: 10, height: 3.2 });
pres.layout = "WIDE";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\paper_panels_depth"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_1E_Pipeline_v2.pptx"
);

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// Panel label
slide.addText("E", {
  x: 0.08, y: 0.05, w: 0.3, h: 0.3,
  fontSize: 18, fontFace: "Arial", bold: true, color: "000000",
});

// === Layout: 3 main stages with schematic separation ===
// [BF image] --LBBDM--> [Multi-color FL] --Channel Sep--> [Schematic: R/B split → metrics]

const imgS = 1.5; // square image size
const baseY = 0.9;

// ---- Stage 1: BF ----
const x1 = 0.3;
slide.addText("Single-plane Brightfield", {
  x: x1, y: 0.35, w: imgS, h: 0.35,
  fontSize: 9, fontFace: "Arial", bold: true, color: "7f8c8d", align: "center",
});
slide.addImage({ path: path.join(P, "fig1e_bf.png"), x: x1, y: baseY, w: imgS, h: imgS });
slide.addShape(pres.shapes.RECTANGLE, {
  x: x1, y: baseY, w: imgS, h: imgS, line: { color: "7f8c8d", width: 2 },
});

// ---- Arrow 1 with LBBDM label ----
const a1x = x1 + imgS + 0.08;
const aW = 0.85;
const aY = baseY + imgS / 2;
// Arrow line
slide.addShape(pres.shapes.LINE, {
  x: a1x, y: aY, w: aW, h: 0,
  line: { color: "2c3e50", width: 2.5, endArrowType: "triangle" },
});
// LBBDM badge on arrow
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: a1x + aW / 2 - 0.4, y: aY - 0.32, w: 0.8, h: 0.25,
  fill: { color: "2c3e50" }, rectRadius: 0.05,
});
slide.addText("LBBDM", {
  x: a1x + aW / 2 - 0.4, y: aY - 0.32, w: 0.8, h: 0.25,
  fontSize: 8, fontFace: "Arial", bold: true, color: "FFFFFF", align: "center", margin: 0,
});
slide.addText("Virtual Staining", {
  x: a1x, y: aY + 0.08, w: aW, h: 0.18,
  fontSize: 6.5, fontFace: "Arial", color: "888888", align: "center", italic: true,
});

// ---- Stage 2: Multi-color FL ----
const x2 = a1x + aW + 0.08;
slide.addText("Depth-encoded Fluorescence", {
  x: x2, y: 0.35, w: imgS, h: 0.35,
  fontSize: 9, fontFace: "Arial", bold: true, color: "8e44ad", align: "center",
});
slide.addImage({ path: path.join(P, "fig1e_gt.png"), x: x2, y: baseY, w: imgS, h: imgS });
slide.addShape(pres.shapes.RECTANGLE, {
  x: x2, y: baseY, w: imgS, h: imgS, line: { color: "8e44ad", width: 2 },
});

// ---- Arrow 2 ----
const a2x = x2 + imgS + 0.08;
slide.addShape(pres.shapes.LINE, {
  x: a2x, y: aY, w: aW * 0.7, h: 0,
  line: { color: "555555", width: 2.5, endArrowType: "triangle" },
});
slide.addText("RGB\nSeparation", {
  x: a2x, y: aY + 0.08, w: aW * 0.7, h: 0.25,
  fontSize: 6.5, fontFace: "Arial", color: "888888", align: "center", italic: true,
});

// ---- Stage 3: Schematic depth separation + skeleton ----
const x3 = a2x + aW * 0.7 + 0.08;
const schW = 3.8;

slide.addText("Depth-separated Skeletonization & Quantification", {
  x: x3, y: 0.35, w: schW, h: 0.35,
  fontSize: 9, fontFace: "Arial", bold: true, color: "333333", align: "center",
});

// Schematic box background
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: x3, y: baseY - 0.05, w: schW, h: imgS + 0.1,
  fill: { color: "f7f8fa" }, line: { color: "DDDDDD", width: 1 }, rectRadius: 0.08,
});

// Two schematic vessel lines: Top (blue) and Bottom (red)
// Top vessel: horizontal line with branching
const lineY_top = baseY + 0.25;
const lineY_bot = baseY + imgS - 0.3;
const lineStart = x3 + 0.25;
const lineEnd = x3 + schW - 1.3;

// -- Top vessel (blue) --
slide.addShape(pres.shapes.LINE, {
  x: lineStart, y: lineY_top, w: lineEnd - lineStart, h: 0.2,
  line: { color: "2980b9", width: 5 },
});
// Branch
slide.addShape(pres.shapes.LINE, {
  x: lineStart + 0.8, y: lineY_top + 0.1, w: 0.5, h: -0.3,
  line: { color: "2980b9", width: 4 },
});
// EP dot
slide.addShape(pres.shapes.OVAL, {
  x: lineStart + 1.25, y: lineY_top - 0.25, w: 0.1, h: 0.1,
  fill: { color: "f1c40f" },
});
// JN dot
slide.addShape(pres.shapes.OVAL, {
  x: lineStart + 0.75, y: lineY_top + 0.06, w: 0.12, h: 0.12,
  fill: { color: "e91e9c" },
});
slide.addText("Top layer (B>R)", {
  x: lineStart - 0.15, y: lineY_top - 0.42, w: 1.5, h: 0.2,
  fontSize: 7, fontFace: "Arial", bold: true, color: "2980b9",
});

// -- Bottom vessel (red) --
slide.addShape(pres.shapes.LINE, {
  x: lineStart + 0.15, y: lineY_bot - 0.15, w: lineEnd - lineStart - 0.15, h: 0.15,
  line: { color: "c0392b", width: 5 },
});
// Branch
slide.addShape(pres.shapes.LINE, {
  x: lineStart + 1.2, y: lineY_bot - 0.05, w: 0.4, h: 0.35,
  line: { color: "c0392b", width: 4 },
});
// EP dot
slide.addShape(pres.shapes.OVAL, {
  x: lineStart + 1.55, y: lineY_bot + 0.25, w: 0.1, h: 0.1,
  fill: { color: "f1c40f" },
});
// JN dot
slide.addShape(pres.shapes.OVAL, {
  x: lineStart + 1.15, y: lineY_bot - 0.09, w: 0.12, h: 0.12,
  fill: { color: "e91e9c" },
});
slide.addText("Bottom layer (R>B)", {
  x: lineStart - 0.15, y: lineY_bot + 0.2, w: 1.5, h: 0.2,
  fontSize: 7, fontFace: "Arial", bold: true, color: "c0392b",
});

// Crossing indication (dashed oval where they would overlap)
const crossX = lineStart + 0.5;
const crossY = (lineY_top + lineY_bot) / 2 - 0.1;
slide.addShape(pres.shapes.OVAL, {
  x: crossX, y: crossY, w: 0.6, h: 0.4,
  line: { color: "999999", width: 1, dashType: "dash" },
});
slide.addText("crossing\nresolved", {
  x: crossX - 0.05, y: crossY + 0.4, w: 0.7, h: 0.25,
  fontSize: 5.5, fontFace: "Arial", color: "999999", align: "center", italic: true,
});

// -- Right side: metrics summary --
const metX = x3 + schW - 1.15;
const metY = baseY + 0.1;

// Metrics box
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: metX, y: metY, w: 1.0, h: imgS - 0.2,
  fill: { color: "FFFFFF" }, line: { color: "27ae60", width: 1.5 }, rectRadius: 0.06,
});
slide.addText("Per-layer\nMetrics", {
  x: metX, y: metY + 0.02, w: 1.0, h: 0.3,
  fontSize: 8, fontFace: "Arial", bold: true, color: "27ae60", align: "center", margin: 0,
});

const metrics = ["Vessel Area", "Vessel Length", "Junctions", "Endpoints"];
metrics.forEach((m, i) => {
  slide.addText([
    { text: "\u2022 ", options: { color: "27ae60", bold: true } },
    { text: m },
  ], {
    x: metX + 0.05, y: metY + 0.32 + i * 0.2, w: 0.9, h: 0.18,
    fontSize: 7, fontFace: "Arial", color: "333333", margin: 0,
  });
});

// Arrow from schematic to metrics
slide.addShape(pres.shapes.LINE, {
  x: metX - 0.15, y: aY, w: 0.15, h: 0,
  line: { color: "27ae60", width: 1.5, endArrowType: "triangle" },
});

// Legend at very bottom
slide.addText([
  { text: "\u25CF ", options: { color: "e91e9c", fontSize: 8 } },
  { text: "Junction   ", options: { fontSize: 7 } },
  { text: "\u25CF ", options: { color: "f1c40f", fontSize: 8 } },
  { text: "Endpoint   ", options: { fontSize: 7 } },
  { text: "\u2500\u2500 ", options: { color: "2980b9", fontSize: 8 } },
  { text: "Top skel   ", options: { fontSize: 7 } },
  { text: "\u2500\u2500 ", options: { color: "c0392b", fontSize: 8 } },
  { text: "Bottom skel", options: { fontSize: 7 } },
], {
  x: x3, y: baseY + imgS + 0.12, w: schW, h: 0.2,
  fontFace: "Arial", color: "777777", align: "center",
});

pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
