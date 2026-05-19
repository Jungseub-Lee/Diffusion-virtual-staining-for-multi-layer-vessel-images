const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: 10, height: 3.5 });
pres.layout = "WIDE";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\paper_panels_depth"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_1E_Pipeline_v3b.pptx"
);

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

slide.addText("E", {
  x: 0.08, y: 0.05, w: 0.3, h: 0.3,
  fontSize: 18, fontFace: "Arial", bold: true, color: "000000",
});

// === 6 panels: BF → FL → Top / Bot → R skel + B skel → Combined skel ===
const imgS = 1.15;
const gap = 0.06;
const arrowW = 0.42;
const baseY = 1.05;
const labelY = 0.4;

// Positions
// Group A: [BF] --arrow--> [FL] --arrow--> [Top, Bot stacked] --arrow--> [R skel, B skel stacked] --arrow--> [Combined]
const x_bf = 0.15;
const x_fl = x_bf + imgS + arrowW;
const x_sep = x_fl + imgS + arrowW;
const x_skel = x_sep + imgS + arrowW;
const x_comb = x_skel + imgS + arrowW;

const aY = baseY + imgS / 2;

function addArrow(x1, x2, label) {
  const ax = x1;
  const aw = x2 - x1;
  slide.addShape(pres.shapes.LINE, {
    x: ax, y: aY, w: aw, h: 0,
    line: { color: "888888", width: 2, endArrowType: "triangle" },
  });
  if (label) {
    slide.addText(label, {
      x: ax, y: aY + 0.12, w: aw, h: 0.22,
      fontSize: 6, fontFace: "Arial", color: "AAAAAA", align: "center", italic: true,
    });
  }
}

function addPanel(x, y, w, h, file, borderColor) {
  slide.addImage({ path: path.join(P, file), x, y, w, h });
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, line: { color: borderColor, width: 2 },
  });
}

function addLabel(x, y, w, text, color) {
  slide.addText(text, {
    x, y, w, h: 0.45,
    fontSize: 8, fontFace: "Arial", bold: true, color, align: "center", valign: "middle",
  });
}

// 1. BF
addLabel(x_bf, labelY, imgS, "Single-plane\nBrightfield", "7f8c8d");
addPanel(x_bf, baseY, imgS, imgS, "fig1e_bf.png", "7f8c8d");

// Arrow: LBBDM
const a1_start = x_bf + imgS + 0.03;
const a1_end = x_fl - 0.03;
addArrow(a1_start, a1_end, "");
// LBBDM badge
const badgeW = 0.65;
const badgeX = (a1_start + a1_end) / 2 - badgeW / 2;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: badgeX, y: aY - 0.28, w: badgeW, h: 0.22,
  fill: { color: "2c3e50" }, rectRadius: 0.04,
});
slide.addText("LBBDM", {
  x: badgeX, y: aY - 0.28, w: badgeW, h: 0.22,
  fontSize: 7, fontFace: "Arial", bold: true, color: "FFFFFF", align: "center", margin: 0,
});

// 2. Multi-color FL
addLabel(x_fl, labelY, imgS, "Depth-encoded\nFluorescence", "8e44ad");
addPanel(x_fl, baseY, imgS, imgS, "fig1e_gt.png", "8e44ad");

// Arrow: Channel separation
addArrow(x_fl + imgS + 0.03, x_sep - 0.03, "Channel\nSeparation");

// 3. Top / Bot stacked
addLabel(x_sep, labelY, imgS, "Layer\nSeparation", "555555");
const halfH = (imgS - gap) / 2;
addPanel(x_sep, baseY, imgS, halfH, "fig1e_top.png", "2980b9");
addPanel(x_sep, baseY + halfH + gap, imgS, halfH, "fig1e_bot.png", "c0392b");
// Tiny labels
slide.addText("Top (B>R)", {
  x: x_sep + 0.02, y: baseY + 0.02, w: 0.6, h: 0.15,
  fontSize: 5.5, fontFace: "Arial", bold: true, color: "FFFFFF",
});
slide.addText("Bot (R>B)", {
  x: x_sep + 0.02, y: baseY + halfH + gap + 0.02, w: 0.6, h: 0.15,
  fontSize: 5.5, fontFace: "Arial", bold: true, color: "FFFFFF",
});

// Arrow: Skeletonize
addArrow(x_sep + imgS + 0.03, x_skel - 0.03, "Skeletonize\n+ Bridge");

// 4. R skel / B skel stacked
addLabel(x_skel, labelY, imgS, "Per-layer\nSkeleton", "e67e22");
addPanel(x_skel, baseY, imgS, halfH, "fig1e_skel_b.png", "2980b9");
addPanel(x_skel, baseY + halfH + gap, imgS, halfH, "fig1e_skel_r.png", "c0392b");
slide.addText("B skel", {
  x: x_skel + 0.02, y: baseY + 0.02, w: 0.5, h: 0.15,
  fontSize: 5.5, fontFace: "Arial", bold: true, color: "6699FF",
});
slide.addText("R skel", {
  x: x_skel + 0.02, y: baseY + halfH + gap + 0.02, w: 0.5, h: 0.15,
  fontSize: 5.5, fontFace: "Arial", bold: true, color: "FF6666",
});

// Arrow: Combine
addArrow(x_skel + imgS + 0.03, x_comb - 0.03, "Combine\n& Quantify");

// 5. Combined depth skeleton
addLabel(x_comb, labelY, imgS, "Depth-aware\nQuantification", "27ae60");
addPanel(x_comb, baseY, imgS, imgS, "fig1e_skel.png", "27ae60");

// Metrics below combined
slide.addText("Area | Length | Junctions | Endpoints", {
  x: x_comb - 0.15, y: baseY + imgS + 0.05, w: imgS + 0.3, h: 0.18,
  fontSize: 6, fontFace: "Arial", color: "27ae60", align: "center", italic: true,
});

// Legend
slide.addText([
  { text: "Red ", options: { color: "CC3333", bold: true, fontSize: 6.5 } },
  { text: "Bottom   ", options: { fontSize: 6.5 } },
  { text: "Yellow ", options: { color: "CCAA00", bold: true, fontSize: 6.5 } },
  { text: "Middle   ", options: { fontSize: 6.5 } },
  { text: "Blue ", options: { color: "3366CC", bold: true, fontSize: 6.5 } },
  { text: "Top   ", options: { fontSize: 6.5 } },
  { text: "Purple ", options: { color: "B464DC", bold: true, fontSize: 6.5 } },
  { text: "Overlap", options: { fontSize: 6.5 } },
], {
  x: 0.5, y: 3.05, w: 9, h: 0.2,
  fontFace: "Arial", color: "777777", align: "center",
});

// ==================== SLIDE 2: With EP/JN markers ====================
let slide2 = pres.addSlide();
slide2.background = { color: "FFFFFF" };

slide2.addText("E (with morphometric markers)", {
  x: 0.08, y: 0.05, w: 5, h: 0.3,
  fontSize: 14, fontFace: "Arial", bold: true, color: "000000",
});

// Same layout, but last panel uses marked version
// Stage 1-4 identical, Stage 5 uses fig1e_skel_marked.png

// Reuse same positions
const s2_panels = [
  { x: x_bf, f: "fig1e_bf.png", border: "7f8c8d", label: "Single-plane\nBrightfield", labelC: "7f8c8d", stack: false },
  { x: x_fl, f: "fig1e_gt.png", border: "8e44ad", label: "Depth-encoded\nFluorescence", labelC: "8e44ad", stack: false },
];

s2_panels.forEach(p => {
  slide2.addText(p.label, {
    x: p.x, y: labelY, w: imgS, h: 0.45,
    fontSize: 8, fontFace: "Arial", bold: true, color: p.labelC, align: "center", valign: "middle",
  });
  slide2.addImage({ path: path.join(P, p.f), x: p.x, y: baseY, w: imgS, h: imgS });
  slide2.addShape(pres.shapes.RECTANGLE, {
    x: p.x, y: baseY, w: imgS, h: imgS, line: { color: p.border, width: 2 },
  });
});

// Arrow 1 (LBBDM)
slide2.addShape(pres.shapes.LINE, {
  x: x_bf + imgS + 0.03, y: aY, w: arrowW - 0.06, h: 0,
  line: { color: "888888", width: 2, endArrowType: "triangle" },
});
const bX2 = (x_bf + imgS + 0.03 + x_fl - 0.03) / 2 - 0.325;
slide2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: bX2, y: aY - 0.28, w: 0.65, h: 0.22,
  fill: { color: "2c3e50" }, rectRadius: 0.04,
});
slide2.addText("LBBDM", {
  x: bX2, y: aY - 0.28, w: 0.65, h: 0.22,
  fontSize: 7, fontFace: "Arial", bold: true, color: "FFFFFF", align: "center", margin: 0,
});

// Arrow 2
slide2.addShape(pres.shapes.LINE, {
  x: x_fl + imgS + 0.03, y: aY, w: arrowW * 0.7 - 0.06, h: 0,
  line: { color: "888888", width: 2, endArrowType: "triangle" },
});
slide2.addText("Channel\nSeparation", {
  x: x_fl + imgS + 0.03, y: aY + 0.12, w: arrowW * 0.7, h: 0.22,
  fontSize: 6, fontFace: "Arial", color: "AAAAAA", align: "center", italic: true,
});

// Stage 3: stacked top/bot
slide2.addText("Layer\nSeparation", {
  x: x_sep, y: labelY, w: imgS, h: 0.45,
  fontSize: 8, fontFace: "Arial", bold: true, color: "555555", align: "center", valign: "middle",
});
const hH2 = (imgS - gap) / 2;
slide2.addImage({ path: path.join(P, "fig1e_top.png"), x: x_sep, y: baseY, w: imgS, h: hH2 });
slide2.addShape(pres.shapes.RECTANGLE, { x: x_sep, y: baseY, w: imgS, h: hH2, line: { color: "2980b9", width: 2 } });
slide2.addImage({ path: path.join(P, "fig1e_bot.png"), x: x_sep, y: baseY + hH2 + gap, w: imgS, h: hH2 });
slide2.addShape(pres.shapes.RECTANGLE, { x: x_sep, y: baseY + hH2 + gap, w: imgS, h: hH2, line: { color: "c0392b", width: 2 } });
slide2.addText("Top (B>R)", { x: x_sep + 0.02, y: baseY + 0.02, w: 0.6, h: 0.15, fontSize: 5.5, fontFace: "Arial", bold: true, color: "FFFFFF" });
slide2.addText("Bot (R>B)", { x: x_sep + 0.02, y: baseY + hH2 + gap + 0.02, w: 0.6, h: 0.15, fontSize: 5.5, fontFace: "Arial", bold: true, color: "FFFFFF" });

// Arrow 3
slide2.addShape(pres.shapes.LINE, {
  x: x_sep + imgS + 0.03, y: aY, w: arrowW - 0.06, h: 0,
  line: { color: "888888", width: 2, endArrowType: "triangle" },
});
slide2.addText("Skeletonize\n+ Bridge", {
  x: x_sep + imgS + 0.03, y: aY + 0.12, w: arrowW, h: 0.22,
  fontSize: 6, fontFace: "Arial", color: "AAAAAA", align: "center", italic: true,
});

// Stage 4: stacked skels
slide2.addText("Per-layer\nSkeleton", {
  x: x_skel, y: labelY, w: imgS, h: 0.45,
  fontSize: 8, fontFace: "Arial", bold: true, color: "e67e22", align: "center", valign: "middle",
});
slide2.addImage({ path: path.join(P, "fig1e_skel_b.png"), x: x_skel, y: baseY, w: imgS, h: hH2 });
slide2.addShape(pres.shapes.RECTANGLE, { x: x_skel, y: baseY, w: imgS, h: hH2, line: { color: "2980b9", width: 2 } });
slide2.addImage({ path: path.join(P, "fig1e_skel_r.png"), x: x_skel, y: baseY + hH2 + gap, w: imgS, h: hH2 });
slide2.addShape(pres.shapes.RECTANGLE, { x: x_skel, y: baseY + hH2 + gap, w: imgS, h: hH2, line: { color: "c0392b", width: 2 } });
slide2.addText("B skel", { x: x_skel + 0.02, y: baseY + 0.02, w: 0.5, h: 0.15, fontSize: 5.5, fontFace: "Arial", bold: true, color: "6699FF" });
slide2.addText("R skel", { x: x_skel + 0.02, y: baseY + hH2 + gap + 0.02, w: 0.5, h: 0.15, fontSize: 5.5, fontFace: "Arial", bold: true, color: "FF6666" });

// Arrow 4
slide2.addShape(pres.shapes.LINE, {
  x: x_skel + imgS + 0.03, y: aY, w: arrowW - 0.06, h: 0,
  line: { color: "888888", width: 2, endArrowType: "triangle" },
});
slide2.addText("Combine\n& Quantify", {
  x: x_skel + imgS + 0.03, y: aY + 0.12, w: arrowW, h: 0.22,
  fontSize: 6, fontFace: "Arial", color: "AAAAAA", align: "center", italic: true,
});

// Stage 5: MARKED version
slide2.addText("Depth-aware\nQuantification", {
  x: x_comb, y: labelY, w: imgS, h: 0.45,
  fontSize: 8, fontFace: "Arial", bold: true, color: "27ae60", align: "center", valign: "middle",
});
slide2.addImage({ path: path.join(P, "fig1e_skel_marked.png"), x: x_comb, y: baseY, w: imgS, h: imgS });
slide2.addShape(pres.shapes.RECTANGLE, {
  x: x_comb, y: baseY, w: imgS, h: imgS, line: { color: "27ae60", width: 2 },
});
slide2.addText("Area | Length | Junctions | Endpoints", {
  x: x_comb - 0.15, y: baseY + imgS + 0.05, w: imgS + 0.3, h: 0.18,
  fontSize: 6, fontFace: "Arial", color: "27ae60", align: "center", italic: true,
});

// Legend with EP/JN markers
slide2.addText([
  { text: "Red ", options: { color: "CC3333", bold: true, fontSize: 6.5 } },
  { text: "Bottom   ", options: { fontSize: 6.5 } },
  { text: "Yellow ", options: { color: "CCAA00", bold: true, fontSize: 6.5 } },
  { text: "Middle   ", options: { fontSize: 6.5 } },
  { text: "Blue ", options: { color: "3366CC", bold: true, fontSize: 6.5 } },
  { text: "Top   ", options: { fontSize: 6.5 } },
  { text: "Purple ", options: { color: "B464DC", bold: true, fontSize: 6.5 } },
  { text: "Overlap   ", options: { fontSize: 6.5 } },
  { text: "\u25CB ", options: { color: "CCAA00", bold: true, fontSize: 7 } },
  { text: "Endpoint   ", options: { fontSize: 6.5 } },
  { text: "\u25CB ", options: { color: "CC00CC", bold: true, fontSize: 7 } },
  { text: "Junction", options: { fontSize: 6.5 } },
], {
  x: 0.5, y: 3.05, w: 9, h: 0.2,
  fontFace: "Arial", color: "777777", align: "center",
});

pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
