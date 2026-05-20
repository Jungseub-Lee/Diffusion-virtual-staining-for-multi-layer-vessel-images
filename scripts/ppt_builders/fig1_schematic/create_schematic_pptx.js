const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3" x 7.5"
pres.author = "Diffusion Virtual Staining";
pres.title = "Depth-Aware Vascular Morphometry Schematic";

// ====== COLOR PALETTE ======
const C = {
  bg: "0A0A0A",
  panel: "1A1A2E",
  panelLight: "16213E",
  accent: "0F3460",
  red: "E94560",
  redDark: "B8001F",
  yellow: "F5C542",
  yellowDark: "C49B20",
  blue: "4DA8DA",
  blueDark: "2E86AB",
  cyan: "00D2FF",
  green: "2ECC71",
  greenDark: "27AE60",
  white: "FFFFFF",
  gray: "888888",
  grayLight: "CCCCCC",
  grayDark: "444444",
  textMain: "E8E8E8",
  textSub: "A0A0A0",
  arrowGray: "666666",
  bfBrown: "8B7355",
  vesselPink: "FF6B9D",
  vesselCyan: "00CED1",
};

const slide = pres.addSlide();
slide.background = { color: C.bg };

// ====== HELPER FUNCTIONS ======
const mkShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.4 });

// Draw a curved vessel-like shape using multiple overlapping ovals/rectangles
function drawVessel(slide, points, color, thickness, opacity = 100) {
  // Draw vessel segments between points
  for (let i = 0; i < points.length - 1; i++) {
    const x1 = points[i][0], y1 = points[i][1];
    const x2 = points[i + 1][0], y2 = points[i + 1][1];
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);

    // Draw as a line
    slide.addShape(pres.shapes.LINE, {
      x: x1, y: y1, w: len, h: 0,
      line: { color: color, width: thickness },
      rotate: angle,
    });
  }
}

// Draw a thick stylized vessel using rectangles
function drawThickVessel(slide, x, y, w, h, color, opacity = 100) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color, transparency: 100 - opacity },
    rectRadius: Math.min(w, h) * 0.4,
  });
}

// Draw arrow
function drawArrow(slide, x, y, w, label, color = C.arrowGray) {
  // Arrow line
  slide.addShape(pres.shapes.LINE, {
    x: x, y: y, w: w, h: 0,
    line: { color: color, width: 2.5, endArrowType: "triangle" },
  });
  if (label) {
    slide.addText(label, {
      x: x, y: y - 0.28, w: w, h: 0.25,
      fontSize: 8, fontFace: "Calibri", color: C.textSub,
      align: "center", valign: "middle",
    });
  }
}

// ====== TITLE ======
slide.addText("Depth-Aware Vascular Morphometry", {
  x: 0.3, y: 0.15, w: 12.7, h: 0.5,
  fontSize: 22, fontFace: "Calibri", bold: true,
  color: C.white, align: "left", margin: 0,
});

slide.addText("Single-color vs. Multi-color Depth-encoded Analysis", {
  x: 0.3, y: 0.55, w: 12.7, h: 0.3,
  fontSize: 12, fontFace: "Calibri", italic: true,
  color: C.textSub, align: "left", margin: 0,
});

// ====== LAYOUT: 3 main columns ======
// Col 1 (0.3-4.1): 3D vessel illustration
// Col 2 (4.5-8.3): Single-color path
// Col 3 (8.7-12.9): Multi-color path

const colW = 3.6;
const col1X = 0.3;
const col2X = 4.5;
const col3X = 8.7;
const topY = 1.1;

// ====== COLUMN 1: 3D VESSEL SCHEMATIC ======
// Panel background
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: col1X, y: topY, w: 3.9, h: 5.8,
  fill: { color: C.panel }, rectRadius: 0.15,
  shadow: mkShadow(),
});

slide.addText("3D Vascular Network", {
  x: col1X, y: topY + 0.15, w: 3.9, h: 0.35,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: C.white, align: "center", margin: 0,
});

slide.addText("(MPS cross-section, ~80 µm height)", {
  x: col1X, y: topY + 0.45, w: 3.9, h: 0.25,
  fontSize: 9, fontFace: "Calibri", italic: true,
  color: C.textSub, align: "center", margin: 0,
});

// Draw depth layers
const layerY = [topY + 4.6, topY + 3.3, topY + 2.0]; // bottom, middle, top
const layerColors = [C.red, C.yellow, C.blue];
const layerLabels = ["Bottom layer (~0-25 µm)", "Middle layer (~25-55 µm)", "Top layer (~55-80 µm)"];
const layerLabelColors = [C.red, C.yellow, C.blue];

// Draw depth reference bars on the right
for (let i = 0; i < 3; i++) {
  // Layer background band
  slide.addShape(pres.shapes.RECTANGLE, {
    x: col1X + 0.3, y: layerY[i], w: 3.0, h: 1.0,
    fill: { color: layerColors[i], transparency: 88 },
    line: { color: layerColors[i], width: 0.5, dashType: "dash" },
  });

  // Layer label
  slide.addText(layerLabels[i], {
    x: col1X + 0.35, y: layerY[i] + 0.02, w: 2.9, h: 0.2,
    fontSize: 7.5, fontFace: "Calibri", color: layerLabelColors[i],
    align: "left", margin: 0, italic: true,
  });
}

// Draw vessels in the 3D view
// Vessel A: goes from bottom-left to top-right (spans all layers)
// Bottom segment (red)
drawThickVessel(slide, col1X + 0.5, layerY[0] + 0.35, 1.8, 0.45, C.red, 80);
// Middle segment (yellow) - continuing up
drawThickVessel(slide, col1X + 1.4, layerY[1] + 0.3, 1.2, 0.4, C.yellow, 80);
// Top segment (blue)
drawThickVessel(slide, col1X + 1.8, layerY[2] + 0.3, 1.5, 0.4, C.blue, 80);

// Vessel B: horizontal at top layer (blue only)
drawThickVessel(slide, col1X + 0.4, layerY[2] + 0.55, 2.5, 0.35, C.blue, 80);

// Vessel C: bottom layer horizontal (red)
drawThickVessel(slide, col1X + 1.5, layerY[0] + 0.55, 1.7, 0.35, C.red, 80);

// Vessel D: middle to top (branching)
drawThickVessel(slide, col1X + 2.3, layerY[1] + 0.5, 0.9, 0.3, C.yellow, 70);

// Crossing indicator
slide.addText("Vessels cross at\ndifferent depths", {
  x: col1X + 0.35, y: topY + 5.2, w: 3.2, h: 0.45,
  fontSize: 8.5, fontFace: "Calibri", color: C.grayLight,
  align: "center", valign: "middle", margin: 0,
});

// Depth arrow on the side
slide.addShape(pres.shapes.LINE, {
  x: col1X + 3.55, y: layerY[2] + 0.1, w: 0, h: layerY[0] + 0.9 - layerY[2] - 0.1,
  line: { color: C.grayLight, width: 1.5, endArrowType: "triangle", beginArrowType: "triangle" },
});
slide.addText("~80 µm", {
  x: col1X + 3.3, y: layerY[1] + 0.15, w: 0.6, h: 0.2,
  fontSize: 7, fontFace: "Calibri", color: C.grayLight,
  align: "center", margin: 0, rotate: 90,
});


// ====== COLUMN 2: SINGLE-COLOR PATH ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: col2X, y: topY, w: 3.9, h: 5.8,
  fill: { color: C.panel }, rectRadius: 0.15,
  shadow: mkShadow(),
});

slide.addText("Single-color Projection", {
  x: col2X, y: topY + 0.15, w: 3.9, h: 0.35,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: C.green, align: "center", margin: 0,
});

slide.addText("(Conventional approach)", {
  x: col2X, y: topY + 0.45, w: 3.9, h: 0.25,
  fontSize: 9, fontFace: "Calibri", italic: true,
  color: C.textSub, align: "center", margin: 0,
});

// Single-color projection image area
const sc_imgY = topY + 0.85;
slide.addShape(pres.shapes.RECTANGLE, {
  x: col2X + 0.3, y: sc_imgY, w: 3.3, h: 1.6,
  fill: { color: "0D0D0D" },
  line: { color: C.green, width: 1 },
});

// Draw all vessels in single green color (overlapping)
// Vessel A (full span)
drawThickVessel(slide, col2X + 0.5, sc_imgY + 0.2, 1.8, 0.35, C.green, 70);
drawThickVessel(slide, col2X + 1.3, sc_imgY + 0.5, 1.2, 0.3, C.green, 70);
drawThickVessel(slide, col2X + 1.7, sc_imgY + 0.8, 1.5, 0.3, C.green, 70);
// Vessel B
drawThickVessel(slide, col2X + 0.5, sc_imgY + 0.95, 2.5, 0.3, C.green, 70);
// Vessel C
drawThickVessel(slide, col2X + 1.5, sc_imgY + 0.25, 1.5, 0.3, C.green, 70);

// Arrow down
slide.addShape(pres.shapes.LINE, {
  x: col2X + 1.95, y: sc_imgY + 1.7, w: 0, h: 0.3,
  line: { color: C.green, width: 2, endArrowType: "triangle" },
});

slide.addText("Skeletonize", {
  x: col2X + 0.5, y: sc_imgY + 1.72, w: 1.4, h: 0.25,
  fontSize: 8, fontFace: "Calibri", color: C.textSub, align: "right", margin: 0,
});

// Single-color skeleton
const sc_skelY = sc_imgY + 2.1;
slide.addShape(pres.shapes.RECTANGLE, {
  x: col2X + 0.3, y: sc_skelY, w: 3.3, h: 1.6,
  fill: { color: "0D0D0D" },
  line: { color: C.green, width: 1, dashType: "dash" },
});

// Skeleton lines (green) with crossing junction highlighted
// Horizontal vessel
slide.addShape(pres.shapes.LINE, {
  x: col2X + 0.5, y: sc_skelY + 0.5, w: 2.8, h: 0,
  line: { color: C.green, width: 3 },
});
// Diagonal vessel crossing
slide.addShape(pres.shapes.LINE, {
  x: col2X + 0.8, y: sc_skelY + 1.3, w: 2.0, h: -1.0,
  line: { color: C.green, width: 3 },
});
// Branch
slide.addShape(pres.shapes.LINE, {
  x: col2X + 1.8, y: sc_skelY + 0.5, w: 1.2, h: 0.8,
  line: { color: C.green, width: 3 },
});

// FALSE junction marker at crossing
const falseJX = col2X + 1.95;
const falseJY = sc_skelY + 0.5;
slide.addShape(pres.shapes.OVAL, {
  x: falseJX - 0.15, y: falseJY - 0.15, w: 0.3, h: 0.3,
  fill: { color: C.red, transparency: 30 },
  line: { color: C.red, width: 2 },
});
// False junction label
slide.addText("False\njunction", {
  x: falseJX + 0.2, y: falseJY - 0.35, w: 0.7, h: 0.4,
  fontSize: 7, fontFace: "Calibri", bold: true,
  color: C.red, align: "left", margin: 0,
});

// Real junction marker
const realJX = col2X + 1.8;
const realJY = sc_skelY + 0.5;
// (This overlaps with false - that's the point!)

// Metrics box
const sc_metY = sc_skelY + 1.8;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: col2X + 0.3, y: sc_metY, w: 3.3, h: 1.0,
  fill: { color: "111122" }, rectRadius: 0.08,
  line: { color: C.grayDark, width: 0.5 },
});

slide.addText("Morphometric Results", {
  x: col2X + 0.3, y: sc_metY + 0.05, w: 3.3, h: 0.2,
  fontSize: 8, fontFace: "Calibri", bold: true,
  color: C.textSub, align: "center", margin: 0,
});

slide.addText([
  { text: "Junctions: ", options: { bold: true, color: C.grayLight, fontSize: 9 } },
  { text: "Over-counted ", options: { color: C.red, fontSize: 9 } },
  { text: "(overlapping vessels = false junctions)", options: { color: C.textSub, fontSize: 7.5, breakLine: true } },
  { text: "Length: ", options: { bold: true, color: C.grayLight, fontSize: 9 } },
  { text: "Under-counted ", options: { color: C.yellow, fontSize: 9 } },
  { text: "(merged overlapping segments)", options: { color: C.textSub, fontSize: 7.5, breakLine: true } },
  { text: "Depth info: ", options: { bold: true, color: C.grayLight, fontSize: 9 } },
  { text: "Lost ", options: { color: C.red, fontSize: 9 } },
  { text: "(no height discrimination)", options: { color: C.textSub, fontSize: 7.5 } },
], {
  x: col2X + 0.45, y: sc_metY + 0.28, w: 3.0, h: 0.7,
  margin: 0, paraSpaceAfter: 2,
});


// ====== COLUMN 3: MULTI-COLOR DEPTH PATH ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: col3X, y: topY, w: 3.9, h: 5.8,
  fill: { color: C.panel }, rectRadius: 0.15,
  shadow: mkShadow(),
});

slide.addText("Multi-color Depth-encoded", {
  x: col3X, y: topY + 0.15, w: 3.9, h: 0.35,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: C.cyan, align: "center", margin: 0,
});

slide.addText("(Proposed approach)", {
  x: col3X, y: topY + 0.45, w: 3.9, h: 0.25,
  fontSize: 9, fontFace: "Calibri", italic: true,
  color: C.textSub, align: "center", margin: 0,
});

// Multi-color projection
const mc_imgY = topY + 0.85;
slide.addShape(pres.shapes.RECTANGLE, {
  x: col3X + 0.3, y: mc_imgY, w: 3.3, h: 1.6,
  fill: { color: "0D0D0D" },
  line: { color: C.cyan, width: 1 },
});

// Draw vessels in depth-encoded colors
// Bottom vessel (red)
drawThickVessel(slide, col3X + 0.5, mc_imgY + 0.2, 1.8, 0.3, C.red, 80);
drawThickVessel(slide, col3X + 1.5, mc_imgY + 0.25, 1.5, 0.25, C.red, 80);
// Middle (yellow transition)
drawThickVessel(slide, col3X + 1.3, mc_imgY + 0.5, 1.0, 0.25, C.yellow, 80);
// Top vessels (blue)
drawThickVessel(slide, col3X + 1.7, mc_imgY + 0.8, 1.5, 0.3, C.blue, 80);
drawThickVessel(slide, col3X + 0.5, mc_imgY + 0.95, 2.5, 0.3, C.blue, 80);

// Arrow down
slide.addShape(pres.shapes.LINE, {
  x: col3X + 1.95, y: mc_imgY + 1.7, w: 0, h: 0.3,
  line: { color: C.cyan, width: 2, endArrowType: "triangle" },
});

slide.addText("Layer separation\n& Skeletonize", {
  x: col3X + 0.3, y: mc_imgY + 1.68, w: 1.6, h: 0.35,
  fontSize: 8, fontFace: "Calibri", color: C.textSub, align: "right", margin: 0,
});

// Multi-color skeleton with layer-separated lines
const mc_skelY = mc_imgY + 2.1;
slide.addShape(pres.shapes.RECTANGLE, {
  x: col3X + 0.3, y: mc_skelY, w: 3.3, h: 1.6,
  fill: { color: "0D0D0D" },
  line: { color: C.cyan, width: 1, dashType: "dash" },
});

// Red skeleton (bottom) - horizontal
slide.addShape(pres.shapes.LINE, {
  x: col3X + 0.5, y: mc_skelY + 0.5, w: 2.8, h: 0,
  line: { color: C.red, width: 3 },
});
// Red branch
slide.addShape(pres.shapes.LINE, {
  x: col3X + 1.8, y: mc_skelY + 0.5, w: 1.2, h: 0.8,
  line: { color: C.red, width: 3 },
});

// Blue skeleton (top) - diagonal, passes through WITHOUT creating junction
slide.addShape(pres.shapes.LINE, {
  x: col3X + 0.8, y: mc_skelY + 1.3, w: 2.0, h: -1.0,
  line: { color: C.blue, width: 3 },
});

// Yellow bridge segment (connecting)
slide.addShape(pres.shapes.LINE, {
  x: col3X + 1.4, y: mc_skelY + 0.85, w: 0.5, h: -0.2,
  line: { color: C.yellow, width: 2.5, dashType: "dash" },
});

// NO false junction - show "pass-through" indicator
const passX = col3X + 1.95;
const passY = mc_skelY + 0.5;
// Crossing point with "no junction" indicator
slide.addShape(pres.shapes.OVAL, {
  x: passX - 0.15, y: passY - 0.15, w: 0.3, h: 0.3,
  fill: { color: "000000", transparency: 60 },
  line: { color: C.green, width: 2 },
});
// Check mark or "pass" indicator
slide.addText("No false\njunction", {
  x: passX + 0.18, y: passY - 0.35, w: 0.8, h: 0.4,
  fontSize: 7, fontFace: "Calibri", bold: true,
  color: C.green, align: "left", margin: 0,
});

// Layer legend inside skeleton
slide.addShape(pres.shapes.LINE, {
  x: col3X + 0.4, y: mc_skelY + 1.35, w: 0.3, h: 0,
  line: { color: C.red, width: 3 },
});
slide.addText("Bottom", {
  x: col3X + 0.73, y: mc_skelY + 1.28, w: 0.5, h: 0.15,
  fontSize: 6.5, fontFace: "Calibri", color: C.red, margin: 0,
});

slide.addShape(pres.shapes.LINE, {
  x: col3X + 1.3, y: mc_skelY + 1.35, w: 0.3, h: 0,
  line: { color: C.yellow, width: 3 },
});
slide.addText("Mid", {
  x: col3X + 1.63, y: mc_skelY + 1.28, w: 0.4, h: 0.15,
  fontSize: 6.5, fontFace: "Calibri", color: C.yellow, margin: 0,
});

slide.addShape(pres.shapes.LINE, {
  x: col3X + 2.05, y: mc_skelY + 1.35, w: 0.3, h: 0,
  line: { color: C.blue, width: 3 },
});
slide.addText("Top", {
  x: col3X + 2.38, y: mc_skelY + 1.28, w: 0.4, h: 0.15,
  fontSize: 6.5, fontFace: "Calibri", color: C.blue, margin: 0,
});


// Metrics box
const mc_metY = mc_skelY + 1.8;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: col3X + 0.3, y: mc_metY, w: 3.3, h: 1.0,
  fill: { color: "111122" }, rectRadius: 0.08,
  line: { color: C.grayDark, width: 0.5 },
});

slide.addText("Morphometric Results", {
  x: col3X + 0.3, y: mc_metY + 0.05, w: 3.3, h: 0.2,
  fontSize: 8, fontFace: "Calibri", bold: true,
  color: C.textSub, align: "center", margin: 0,
});

slide.addText([
  { text: "Junctions: ", options: { bold: true, color: C.grayLight, fontSize: 9 } },
  { text: "Accurate ", options: { color: C.green, fontSize: 9 } },
  { text: "(cross-layer crossings excluded)", options: { color: C.textSub, fontSize: 7.5, breakLine: true } },
  { text: "Length: ", options: { bold: true, color: C.grayLight, fontSize: 9 } },
  { text: "Recovered ", options: { color: C.green, fontSize: 9 } },
  { text: "(per-layer independent sum)", options: { color: C.textSub, fontSize: 7.5, breakLine: true } },
  { text: "Depth info: ", options: { bold: true, color: C.grayLight, fontSize: 9 } },
  { text: "Preserved ", options: { color: C.green, fontSize: 9 } },
  { text: "(3D morphology from 2D image)", options: { color: C.textSub, fontSize: 7.5 } },
], {
  x: col3X + 0.45, y: mc_metY + 0.28, w: 3.0, h: 0.7,
  margin: 0, paraSpaceAfter: 2,
});


// ====== CONNECTING ARROWS between columns ======
// Arrow from col1 to col2
drawArrow(slide, col1X + 3.95, topY + 2.5, 0.5, "MIP", C.green);
// Arrow from col1 to col3
drawArrow(slide, col1X + 3.95, topY + 3.8, 4.7, "Depth-encoded\nprojection", C.cyan);

// Curved arrow from col1 to col3 (use angled lines)
// Upper path to single-color
slide.addText("Max Intensity\nProjection", {
  x: col1X + 3.95, y: topY + 2.2, w: 0.55, h: 0.35,
  fontSize: 6.5, fontFace: "Calibri", color: C.green,
  align: "center", margin: 0,
});

// Lower path to multi-color
slide.addText("Z-depth Color\nEncoding", {
  x: col1X + 3.95, y: topY + 3.5, w: 0.55, h: 0.35,
  fontSize: 6.5, fontFace: "Calibri", color: C.cyan,
  align: "center", margin: 0,
});


// ====== BOTTOM INSIGHT BAR ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.3, y: 7.05, w: 12.7, h: 0.35,
  fill: { color: C.accent }, rectRadius: 0.08,
});

slide.addText([
  { text: "Key insight: ", options: { bold: true, color: C.cyan, fontSize: 10 } },
  { text: "Multi-color depth encoding enables layer-independent skeletonization, eliminating false junctions from overlapping vessels and recovering hidden vessel length — ", options: { color: C.textMain, fontSize: 10 } },
  { text: "extracting 3D morphological information from a single 2D brightfield image.", options: { color: C.yellow, fontSize: 10, bold: true } },
], {
  x: 0.5, y: 7.05, w: 12.3, h: 0.35,
  margin: 0, valign: "middle",
});


// ====== VS indicator between col2 and col3 ======
slide.addShape(pres.shapes.OVAL, {
  x: col2X + 3.9 + 0.15, y: topY + 2.7, w: 0.5, h: 0.5,
  fill: { color: C.accent },
  line: { color: C.grayDark, width: 1 },
});
slide.addText("vs", {
  x: col2X + 3.9 + 0.15, y: topY + 2.7, w: 0.5, h: 0.5,
  fontSize: 12, fontFace: "Calibri", bold: true,
  color: C.white, align: "center", valign: "middle", margin: 0,
});


// ====== SAVE ======
pres.writeFile({ fileName: "depth_aware_schematic.pptx" })
  .then(() => console.log("Created: depth_aware_schematic.pptx"))
  .catch(err => console.error(err));
