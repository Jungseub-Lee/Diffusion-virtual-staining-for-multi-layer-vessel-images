const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3" x 7.5"

const slide = pres.addSlide();
slide.background = { color: "000000" };

// ====== COLORS ======
const RED = "E94560";
const YELLOW = "F5C542";
const BLUE = "4DA8DA";
const GREEN = "3DDC84";
const WHITE = "FFFFFF";
const GRAY = "888888";
const GRAY_LIGHT = "BBBBBB";
const GRAY_DARK = "333333";
const BG_PANEL = "0C0C14";
const ACCENT_RED = "FF4444";
const ACCENT_GREEN = "44FF88";

// ====== LAYOUT ======
const row1Y = 0.55;
const row2Y = 3.95;
const rowH = 3.0;

const vesselX = 0.4;
const vesselW = 5.0;
const arrowX = 5.55;
const arrowW = 0.7;
const skelX = 6.4;
const skelW = 5.0;
const annotX = 11.6;
const annotW = 1.5;

// ====== HELPERS ======
function dot(slide, cx, cy, r, fillColor, lineColor, lineW = 2) {
  slide.addShape(pres.shapes.OVAL, {
    x: cx - r, y: cy - r, w: r * 2, h: r * 2,
    fill: { color: fillColor },
    line: { color: lineColor, width: lineW },
  });
}

// Draw a thick "vessel" line (organic-looking with round ends via thick line width)
function vLine(slide, x1, y1, x2, y2, color, width = 28, opacity = 70) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 0.001) return;
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);
  slide.addShape(pres.shapes.LINE, {
    x: x1, y: y1, w: len, h: 0,
    line: { color, width, transparency: 100 - opacity },
    rotate: angle,
  });
}

// Draw a thin skeleton line
function sLine(slide, x1, y1, x2, y2, color, width = 3.5) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 0.001) return;
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);
  slide.addShape(pres.shapes.LINE, {
    x: x1, y: y1, w: len, h: 0,
    line: { color, width },
    rotate: angle,
  });
}

// ====== DEFINE VESSEL NETWORK ======
// Natural vascular pattern - two vessel trees that cross each other
// All coordinates relative to panel origin, will be offset for each panel

// VESSEL A (will be BLUE in multi-color = top layer)
// Main trunk: enters from left, gentle curve rightward with upward bend
const vesselA = [
  // [x1, y1, x2, y2] segments
  [0.3, 1.7, 1.1, 1.3],     // enter from left, slight up
  [1.1, 1.3, 2.0, 0.9],     // continue up-right
  [2.0, 0.9, 2.9, 0.7],     // flatten out
  [2.9, 0.7, 3.8, 0.5],     // continue right-up
  [3.8, 0.5, 4.6, 0.7],     // slight down at end
  // Branch 1: from junction at ~2.0, goes down
  [2.0, 0.9, 2.4, 1.6],     // branch down-right
  [2.4, 1.6, 2.7, 2.3],     // continue down
  // Branch 2: from ~2.9, small branch up
  [2.9, 0.7, 3.2, 0.3],     // small upward sprout
];

// VESSEL B (will be RED in multi-color = bottom layer)
// Separate tree: enters from bottom-left, goes diagonally up-right, CROSSES vessel A
const vesselB = [
  [0.4, 2.5, 1.0, 2.1],     // enter from lower-left
  [1.0, 2.1, 1.7, 1.7],     // continue up-right
  [1.7, 1.7, 2.5, 1.2],     // approaching crossing zone
  [2.5, 1.2, 3.3, 0.9],     // CROSSES vessel A around here
  [3.3, 0.9, 4.2, 1.2],     // continue, curves down
  [4.2, 1.2, 4.7, 1.6],     // exit right
  // Branch: from ~1.7, goes down
  [1.7, 1.7, 1.5, 2.4],     // branch downward
  // Branch: from ~3.3
  [3.3, 0.9, 3.6, 1.6],     // branch down from crossing area
  [3.6, 1.6, 3.9, 2.2],     // continue down
];

// Crossing point is approximately at (2.7, 1.05) - where vessel A main and vessel B cross

// ====== ROW LABELS & HEADERS ======
slide.addText("a", {
  x: 0.12, y: row1Y, w: 0.25, h: 0.3,
  fontSize: 18, fontFace: "Calibri", bold: true,
  color: WHITE, margin: 0,
});
slide.addText("b", {
  x: 0.12, y: row2Y, w: 0.25, h: 0.3,
  fontSize: 18, fontFace: "Calibri", bold: true,
  color: WHITE, margin: 0,
});

slide.addText("Single-color Fluorescence", {
  x: vesselX, y: row1Y - 0.42, w: vesselW, h: 0.32,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: GREEN, align: "center", margin: 0,
});
slide.addText("Depth-encoded Multi-color Fluorescence", {
  x: vesselX, y: row2Y - 0.42, w: vesselW, h: 0.32,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: BLUE, align: "center", margin: 0,
});
slide.addText("Skeletonization", {
  x: skelX, y: row1Y - 0.42, w: skelW, h: 0.32,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: GRAY_LIGHT, align: "center", margin: 0,
});
slide.addText("Layer-separated Skeletonization", {
  x: skelX, y: row2Y - 0.42, w: skelW, h: 0.32,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: GRAY_LIGHT, align: "center", margin: 0,
});

// ====== PANEL BACKGROUNDS ======
for (const [px, py, pw] of [
  [vesselX, row1Y, vesselW],
  [skelX, row1Y, skelW],
  [vesselX, row2Y, vesselW],
  [skelX, row2Y, skelW],
]) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: px, y: py, w: pw, h: rowH,
    fill: { color: BG_PANEL }, rectRadius: 0.1,
    line: { color: "1A1A2A", width: 1 },
  });
}

// ====== ROW 1: SINGLE-COLOR VESSELS ======
// Draw ALL vessels in GREEN (same color, no depth info)
const v1ox = vesselX, v1oy = row1Y;
for (const seg of vesselA) {
  vLine(slide, v1ox + seg[0], v1oy + seg[1], v1ox + seg[2], v1oy + seg[3], GREEN, 26, 60);
}
for (const seg of vesselB) {
  vLine(slide, v1ox + seg[0], v1oy + seg[1], v1ox + seg[2], v1oy + seg[3], GREEN, 22, 55);
}

// Overlap highlight circle at crossing
dot(slide, v1ox + 2.7, v1oy + 1.0, 0.28, "FFFFFF", WHITE, 1);
// Make it transparent
slide.addShape(pres.shapes.OVAL, {
  x: v1ox + 2.7 - 0.28, y: v1oy + 1.0 - 0.28, w: 0.56, h: 0.56,
  fill: { color: "FFFFFF", transparency: 90 },
  line: { color: WHITE, width: 1, dashType: "dash" },
});
slide.addText("overlap", {
  x: v1ox + 2.35, y: v1oy + 1.3, w: 0.7, h: 0.18,
  fontSize: 7, fontFace: "Calibri", italic: true, color: GRAY_LIGHT, align: "center", margin: 0,
});

// ====== ROW 1: SINGLE-COLOR SKELETON ======
const s1ox = skelX, s1oy = row1Y;
// All skeleton lines in GREEN
for (const seg of vesselA) {
  sLine(slide, s1ox + seg[0], s1oy + seg[1], s1ox + seg[2], s1oy + seg[3], GREEN, 3);
}
for (const seg of vesselB) {
  sLine(slide, s1ox + seg[0], s1oy + seg[1], s1ox + seg[2], s1oy + seg[3], GREEN, 3);
}

// All junctions in single-color (including FALSE junction at crossing)
// Real junctions of vessel A
dot(slide, s1ox + 2.0, s1oy + 0.9, 0.09, "1A6B3A", GREEN, 2);
dot(slide, s1ox + 2.9, s1oy + 0.7, 0.09, "1A6B3A", GREEN, 2);
// Real junctions of vessel B
dot(slide, s1ox + 1.7, s1oy + 1.7, 0.09, "1A6B3A", GREEN, 2);
dot(slide, s1ox + 3.3, s1oy + 0.9, 0.09, "1A6B3A", GREEN, 2);

// FALSE JUNCTION at crossing (~2.7, 1.05)
const fj_cx = s1ox + 2.7;
const fj_cy = s1oy + 1.0;
dot(slide, fj_cx, fj_cy, 0.16, "330000", ACCENT_RED, 3);
slide.addText("✕", {
  x: fj_cx - 0.1, y: fj_cy - 0.12, w: 0.2, h: 0.24,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: ACCENT_RED, align: "center", valign: "middle", margin: 0,
});

// False junction annotation
slide.addText([
  { text: "False junction", options: { bold: true, fontSize: 8.5, color: ACCENT_RED, breakLine: true } },
  { text: "(vessels at different depths\nmisidentified as intersection)", options: { fontSize: 7, color: "CC6666" } },
], {
  x: fj_cx + 0.25, y: fj_cy - 0.7, w: 1.8, h: 0.55,
  margin: 0,
});
// Annotation line
sLine(slide, fj_cx + 0.16, fj_cy - 0.1, fj_cx + 0.35, fj_cy - 0.25, ACCENT_RED, 1.5);

// Endpoints
const epA = [[0.3, 1.7], [4.6, 0.7], [2.7, 2.3], [3.2, 0.3]];
const epB = [[0.4, 2.5], [4.7, 1.6], [1.5, 2.4], [3.9, 2.2]];
for (const ep of [...epA, ...epB]) {
  dot(slide, s1ox + ep[0], s1oy + ep[1], 0.06, "222222", WHITE, 1.5);
}


// ====== ROW 2: MULTI-COLOR VESSELS ======
const v2ox = vesselX, v2oy = row2Y;
// Vessel A = BLUE (top layer)
for (const seg of vesselA) {
  vLine(slide, v2ox + seg[0], v2oy + seg[1], v2ox + seg[2], v2oy + seg[3], BLUE, 26, 60);
}
// Vessel B = RED (bottom layer)
for (const seg of vesselB) {
  vLine(slide, v2ox + seg[0], v2oy + seg[1], v2ox + seg[2], v2oy + seg[3], RED, 22, 55);
}

// Yellow transition at crossing zone
vLine(slide, v2ox + 2.3, v2oy + 1.15, v2ox + 2.7, v2oy + 1.0, YELLOW, 14, 50);

// Crossing zone highlight
slide.addShape(pres.shapes.OVAL, {
  x: v2ox + 2.7 - 0.28, y: v2oy + 1.0 - 0.28, w: 0.56, h: 0.56,
  fill: { color: "000000", transparency: 85 },
  line: { color: YELLOW, width: 1, dashType: "dash" },
});
slide.addText("different\ndepths", {
  x: v2ox + 2.35, y: v2oy + 1.3, w: 0.7, h: 0.25,
  fontSize: 7, fontFace: "Calibri", italic: true, color: YELLOW, align: "center", margin: 0,
});

// Color legend
const clegY = v2oy + 2.55;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: v2ox + 0.3, y: clegY, w: 4.4, h: 0.35,
  fill: { color: "000000", transparency: 40 }, rectRadius: 0.06,
});
// Red
sLine(slide, v2ox + 0.5, clegY + 0.17, v2ox + 0.85, clegY + 0.17, RED, 4);
slide.addText("Bottom", { x: v2ox + 0.9, y: clegY + 0.05, w: 0.65, h: 0.25, fontSize: 8, fontFace: "Calibri", color: RED, margin: 0, valign: "middle" });
// Yellow
sLine(slide, v2ox + 1.6, clegY + 0.17, v2ox + 1.95, clegY + 0.17, YELLOW, 4);
slide.addText("Middle", { x: v2ox + 2.0, y: clegY + 0.05, w: 0.65, h: 0.25, fontSize: 8, fontFace: "Calibri", color: YELLOW, margin: 0, valign: "middle" });
// Blue
sLine(slide, v2ox + 2.7, clegY + 0.17, v2ox + 3.05, clegY + 0.17, BLUE, 4);
slide.addText("Top", { x: v2ox + 3.1, y: clegY + 0.05, w: 0.5, h: 0.25, fontSize: 8, fontFace: "Calibri", color: BLUE, margin: 0, valign: "middle" });
slide.addText("← Deep    Shallow →", { x: v2ox + 3.5, y: clegY + 0.05, w: 1.1, h: 0.25, fontSize: 7, fontFace: "Calibri", color: GRAY, margin: 0, valign: "middle" });


// ====== ROW 2: MULTI-COLOR SKELETON ======
const s2ox = skelX, s2oy = row2Y;
// BLUE skeleton (vessel A = top layer)
for (const seg of vesselA) {
  sLine(slide, s2ox + seg[0], s2oy + seg[1], s2ox + seg[2], s2oy + seg[3], BLUE, 3);
}
// RED skeleton (vessel B = bottom layer)
for (const seg of vesselB) {
  sLine(slide, s2ox + seg[0], s2oy + seg[1], s2ox + seg[2], s2oy + seg[3], RED, 3);
}
// Yellow bridge
sLine(slide, s2ox + 2.3, s2oy + 1.15, s2ox + 2.7, s2oy + 1.0, YELLOW, 2.5);

// NO FALSE JUNCTION at crossing
const nfj_cx = s2ox + 2.7;
const nfj_cy = s2oy + 1.0;
dot(slide, nfj_cx, nfj_cy, 0.16, "003300", ACCENT_GREEN, 3);
slide.addText("✓", {
  x: nfj_cx - 0.1, y: nfj_cy - 0.12, w: 0.2, h: 0.24,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0,
});

// Annotation
slide.addText([
  { text: "No false junction", options: { bold: true, fontSize: 8.5, color: ACCENT_GREEN, breakLine: true } },
  { text: "(layers separated =\nindependent skeletons)", options: { fontSize: 7, color: "66CC88" } },
], {
  x: nfj_cx + 0.25, y: nfj_cy - 0.6, w: 1.8, h: 0.5,
  margin: 0,
});
sLine(slide, nfj_cx + 0.16, nfj_cy - 0.1, nfj_cx + 0.35, nfj_cy - 0.2, ACCENT_GREEN, 1.5);

// Blue junctions
dot(slide, s2ox + 2.0, s2oy + 0.9, 0.09, "2E6685", BLUE, 2);
dot(slide, s2ox + 2.9, s2oy + 0.7, 0.09, "2E6685", BLUE, 2);
// Red junctions
dot(slide, s2ox + 1.7, s2oy + 1.7, 0.09, "8B2A3A", RED, 2);
dot(slide, s2ox + 3.3, s2oy + 0.9, 0.09, "8B2A3A", RED, 2);

// Endpoints
for (const ep of epA) {
  dot(slide, s2ox + ep[0], s2oy + ep[1], 0.06, "222222", BLUE, 1.5);
}
for (const ep of epB) {
  dot(slide, s2ox + ep[0], s2oy + ep[1], 0.06, "222222", RED, 1.5);
}


// ====== ARROWS between vessel and skeleton ======
for (const ry of [row1Y, row2Y]) {
  slide.addShape(pres.shapes.LINE, {
    x: arrowX, y: ry + rowH / 2, w: arrowW, h: 0,
    line: { color: GRAY, width: 2, endArrowType: "triangle" },
  });
}
slide.addText("Skeletonize", {
  x: arrowX - 0.15, y: row1Y + rowH / 2 - 0.22, w: arrowW + 0.3, h: 0.18,
  fontSize: 7, fontFace: "Calibri", color: GRAY, align: "center", margin: 0,
});
slide.addText("Per-layer\nSkeletonize", {
  x: arrowX - 0.15, y: row2Y + rowH / 2 - 0.28, w: arrowW + 0.3, h: 0.28,
  fontSize: 7, fontFace: "Calibri", color: GRAY, align: "center", margin: 0,
});


// ====== RIGHT ANNOTATIONS ======
// Row 1: Limitations
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: annotX, y: row1Y + 0.2, w: annotW, h: 2.6,
  fill: { color: "0F0505" }, rectRadius: 0.08,
  line: { color: "2A1111", width: 1 },
});
slide.addText("Limitations", {
  x: annotX, y: row1Y + 0.25, w: annotW, h: 0.22,
  fontSize: 10, fontFace: "Calibri", bold: true, color: ACCENT_RED, align: "center", margin: 0,
});
slide.addText([
  { text: "Junctions\n", options: { bold: true, fontSize: 9, color: WHITE } },
  { text: "Over-counted\n(false positives)\n\n", options: { fontSize: 7.5, color: "AA8888" } },
  { text: "Total Length\n", options: { bold: true, fontSize: 9, color: WHITE } },
  { text: "Under-estimated\n(merged segments)\n\n", options: { fontSize: 7.5, color: "AA8888" } },
  { text: "Depth\n", options: { bold: true, fontSize: 9, color: WHITE } },
  { text: "Lost", options: { fontSize: 7.5, color: "AA8888" } },
], {
  x: annotX + 0.08, y: row1Y + 0.55, w: annotW - 0.16, h: 2.2,
  margin: 0, paraSpaceAfter: 1,
});

// Row 2: Advantages
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: annotX, y: row2Y + 0.2, w: annotW, h: 2.6,
  fill: { color: "050F05" }, rectRadius: 0.08,
  line: { color: "112A11", width: 1 },
});
slide.addText("Advantages", {
  x: annotX, y: row2Y + 0.25, w: annotW, h: 0.22,
  fontSize: 10, fontFace: "Calibri", bold: true, color: ACCENT_GREEN, align: "center", margin: 0,
});
slide.addText([
  { text: "Junctions\n", options: { bold: true, fontSize: 9, color: WHITE } },
  { text: "Accurate\n(per-layer counting)\n\n", options: { fontSize: 7.5, color: "88AA88" } },
  { text: "Total Length\n", options: { bold: true, fontSize: 9, color: WHITE } },
  { text: "Recovered\n(hidden segments)\n\n", options: { fontSize: 7.5, color: "88AA88" } },
  { text: "Depth\n", options: { bold: true, fontSize: 9, color: WHITE } },
  { text: "Preserved", options: { fontSize: 7.5, color: "88AA88" } },
], {
  x: annotX + 0.08, y: row2Y + 0.55, w: annotW - 0.16, h: 2.2,
  margin: 0, paraSpaceAfter: 1,
});


// ====== DIVIDER ======
slide.addShape(pres.shapes.LINE, {
  x: 0.3, y: (row1Y + rowH + row2Y - 0.42) / 2, w: 12.7, h: 0,
  line: { color: GRAY_DARK, width: 0.7, dashType: "lgDash" },
});

// ====== SKELETON LEGEND (bottom center) ======
const legY = 7.1;
// Junction dot
dot(slide, skelX + 0.8, legY + 0.08, 0.07, "333333", GRAY_LIGHT, 1.5);
slide.addText("Junction", { x: skelX + 0.92, y: legY - 0.03, w: 0.65, h: 0.2, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
// Endpoint dot
dot(slide, skelX + 1.7, legY + 0.08, 0.05, "222222", WHITE, 1.5);
slide.addText("Endpoint", { x: skelX + 1.8, y: legY - 0.03, w: 0.65, h: 0.2, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
// Skeleton line
sLine(slide, skelX + 2.6, legY + 0.08, skelX + 2.9, legY + 0.08, GRAY_LIGHT, 3);
slide.addText("Skeleton", { x: skelX + 2.95, y: legY - 0.03, w: 0.65, h: 0.2, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });


// ====== SAVE ======
pres.writeFile({ fileName: "depth_schematic_v3.pptx" })
  .then(() => console.log("Created: depth_schematic_v3.pptx"))
  .catch(err => console.error(err));
