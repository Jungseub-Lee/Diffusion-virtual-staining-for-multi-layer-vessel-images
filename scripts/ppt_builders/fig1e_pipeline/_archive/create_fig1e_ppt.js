const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
// A4 landscape for wide pipeline scheme
pres.defineLayout({ name: "WIDE", width: 10, height: 3.5 });
pres.layout = "WIDE";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\paper_panels_depth"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_1E_Pipeline.pptx"
);

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// Layout: 5 stages with arrows between them
// [BF] → [LBBDM] → [Multi-color FL] → [Top/Bot Sep] → [Skel + Metrics]
const M = 0.2;
const imgW = 1.35;
const imgH = 1.35;
const boxW = 1.35;
const boxH = 1.35;
const arrowW = 0.55;
const imgY = 1.1;
const labelY = 0.35;
const sublabelY = 0.62;

// Calculate positions
const totalW = 5 * imgW + 4 * arrowW;
const startX = (10 - totalW) / 2;

function stageX(i) {
  return startX + i * (imgW + arrowW);
}

// Colors
const C_BF = "7f8c8d";
const C_MODEL = "2c3e50";
const C_MULTI = "8e44ad";
const C_SEP = "555555";
const C_QUANT = "27ae60";

// Panel label
slide.addText("E", {
  x: 0.1,
  y: 0.05,
  w: 0.3,
  h: 0.3,
  fontSize: 18,
  fontFace: "Arial",
  bold: true,
  color: "000000",
});

// === Stage 1: BF ===
slide.addText("Single-plane\nBrightfield", {
  x: stageX(0),
  y: labelY,
  w: imgW,
  h: 0.5,
  fontSize: 9,
  fontFace: "Arial",
  bold: true,
  color: C_BF,
  align: "center",
  valign: "middle",
});
slide.addImage({
  path: path.join(P, "fig1e_bf.png"),
  x: stageX(0),
  y: imgY,
  w: imgW,
  h: imgH,
});
// Border
slide.addShape(pres.shapes.RECTANGLE, {
  x: stageX(0),
  y: imgY,
  w: imgW,
  h: imgH,
  line: { color: C_BF, width: 2 },
});

// Arrow 1
const aY = imgY + imgH / 2;
slide.addText("\u2192", {
  x: stageX(0) + imgW,
  y: aY - 0.2,
  w: arrowW,
  h: 0.4,
  fontSize: 22,
  fontFace: "Arial",
  color: "999999",
  align: "center",
  valign: "middle",
  bold: true,
});
slide.addText("Virtual\nStaining", {
  x: stageX(0) + imgW,
  y: aY + 0.15,
  w: arrowW,
  h: 0.35,
  fontSize: 6,
  fontFace: "Arial",
  color: "AAAAAA",
  align: "center",
  italic: true,
});

// === Stage 2: LBBDM (text box) ===
slide.addText("Generative\nModel", {
  x: stageX(1),
  y: labelY,
  w: imgW,
  h: 0.5,
  fontSize: 9,
  fontFace: "Arial",
  bold: true,
  color: C_MODEL,
  align: "center",
  valign: "middle",
});
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: stageX(1),
  y: imgY,
  w: boxW,
  h: boxH,
  fill: { color: "ecf0f1" },
  line: { color: C_MODEL, width: 2 },
  rectRadius: 0.1,
});
slide.addText("LBBDM", {
  x: stageX(1),
  y: imgY + boxH * 0.2,
  w: boxW,
  h: 0.35,
  fontSize: 16,
  fontFace: "Arial",
  bold: true,
  color: C_MODEL,
  align: "center",
});
slide.addText("Latent\nBrownian Bridge\nDiffusion Model", {
  x: stageX(1),
  y: imgY + boxH * 0.45,
  w: boxW,
  h: 0.55,
  fontSize: 8,
  fontFace: "Arial",
  color: "777777",
  align: "center",
});

// Arrow 2
slide.addText("\u2192", {
  x: stageX(1) + imgW,
  y: aY - 0.2,
  w: arrowW,
  h: 0.4,
  fontSize: 22,
  fontFace: "Arial",
  color: "999999",
  align: "center",
  bold: true,
});

// === Stage 3: Multi-color FL ===
slide.addText("Depth-encoded\nFluorescence", {
  x: stageX(2),
  y: labelY,
  w: imgW,
  h: 0.5,
  fontSize: 9,
  fontFace: "Arial",
  bold: true,
  color: C_MULTI,
  align: "center",
  valign: "middle",
});
slide.addImage({
  path: path.join(P, "fig1e_gt.png"),
  x: stageX(2),
  y: imgY,
  w: imgW,
  h: imgH,
});
slide.addShape(pres.shapes.RECTANGLE, {
  x: stageX(2),
  y: imgY,
  w: imgW,
  h: imgH,
  line: { color: C_MULTI, width: 2 },
});

// Arrow 3
slide.addText("\u2192", {
  x: stageX(2) + imgW,
  y: aY - 0.2,
  w: arrowW,
  h: 0.4,
  fontSize: 22,
  fontFace: "Arial",
  color: "999999",
  align: "center",
  bold: true,
});
slide.addText("RGB Channel\nSeparation", {
  x: stageX(2) + imgW,
  y: aY + 0.15,
  w: arrowW,
  h: 0.35,
  fontSize: 6,
  fontFace: "Arial",
  color: "AAAAAA",
  align: "center",
  italic: true,
});

// === Stage 4: Layer separation (two stacked images) ===
slide.addText("Depth-layer\nSeparation", {
  x: stageX(3),
  y: labelY,
  w: imgW,
  h: 0.5,
  fontSize: 9,
  fontFace: "Arial",
  bold: true,
  color: C_SEP,
  align: "center",
  valign: "middle",
});
const halfH = (imgH - 0.08) / 2;
// Top layer
slide.addImage({
  path: path.join(P, "fig1e_top.png"),
  x: stageX(3),
  y: imgY,
  w: imgW,
  h: halfH,
});
slide.addShape(pres.shapes.RECTANGLE, {
  x: stageX(3),
  y: imgY,
  w: imgW,
  h: halfH,
  line: { color: "2980b9", width: 2 },
});
slide.addText("Top (B>R)", {
  x: stageX(3) - 0.02,
  y: imgY + halfH / 2 - 0.1,
  w: 0.6,
  h: 0.2,
  fontSize: 6,
  fontFace: "Arial",
  bold: true,
  color: "2980b9",
  align: "center",
  fill: { color: "FFFFFF", transparency: 30 },
});
// Bottom layer
slide.addImage({
  path: path.join(P, "fig1e_bot.png"),
  x: stageX(3),
  y: imgY + halfH + 0.08,
  w: imgW,
  h: halfH,
});
slide.addShape(pres.shapes.RECTANGLE, {
  x: stageX(3),
  y: imgY + halfH + 0.08,
  w: imgW,
  h: halfH,
  line: { color: "c0392b", width: 2 },
});
slide.addText("Bot (R>B)", {
  x: stageX(3) - 0.02,
  y: imgY + halfH + 0.08 + halfH / 2 - 0.1,
  w: 0.6,
  h: 0.2,
  fontSize: 6,
  fontFace: "Arial",
  bold: true,
  color: "c0392b",
  align: "center",
  fill: { color: "FFFFFF", transparency: 30 },
});

// Arrow 4
slide.addText("\u2192", {
  x: stageX(3) + imgW,
  y: aY - 0.2,
  w: arrowW,
  h: 0.4,
  fontSize: 22,
  fontFace: "Arial",
  color: "999999",
  align: "center",
  bold: true,
});
slide.addText("Skeletonize\n& Quantify", {
  x: stageX(3) + imgW,
  y: aY + 0.15,
  w: arrowW,
  h: 0.35,
  fontSize: 6,
  fontFace: "Arial",
  color: "AAAAAA",
  align: "center",
  italic: true,
});

// === Stage 5: Skeleton + Metrics ===
slide.addText("Per-layer\nMorphometrics", {
  x: stageX(4),
  y: labelY,
  w: imgW,
  h: 0.5,
  fontSize: 9,
  fontFace: "Arial",
  bold: true,
  color: C_QUANT,
  align: "center",
  valign: "middle",
});
slide.addImage({
  path: path.join(P, "fig1e_skel.png"),
  x: stageX(4),
  y: imgY,
  w: imgW,
  h: imgH,
});
slide.addShape(pres.shapes.RECTANGLE, {
  x: stageX(4),
  y: imgY,
  w: imgW,
  h: imgH,
  line: { color: C_QUANT, width: 2 },
});

// Metrics text below
slide.addText(
  [
    { text: "Vessel Area", options: { breakLine: true } },
    { text: "Vessel Length", options: { breakLine: true } },
    { text: "Junctions", options: { breakLine: true } },
    { text: "Endpoints" },
  ],
  {
    x: stageX(4),
    y: imgY + imgH + 0.05,
    w: imgW,
    h: 0.55,
    fontSize: 7,
    fontFace: "Arial",
    color: C_QUANT,
    align: "center",
    italic: true,
  }
);

// Color legend at bottom
slide.addText(
  [
    { text: "Red ", options: { color: "CC3333", bold: true, fontSize: 7 } },
    { text: "Bottom   ", options: { fontSize: 7 } },
    { text: "Yellow ", options: { color: "CCAA00", bold: true, fontSize: 7 } },
    { text: "Middle   ", options: { fontSize: 7 } },
    { text: "Blue ", options: { color: "3366CC", bold: true, fontSize: 7 } },
    { text: "Top   ", options: { fontSize: 7 } },
    { text: "Purple ", options: { color: "B464DC", bold: true, fontSize: 7 } },
    { text: "Overlap", options: { fontSize: 7 } },
  ],
  {
    x: stageX(4) - 1,
    y: 3.1,
    w: 3,
    h: 0.2,
    fontFace: "Arial",
    color: "777777",
    align: "center",
  }
);

pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
