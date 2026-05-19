const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3" x 7.5"

const slide = pres.addSlide();
slide.background = { color: "000000" };

// ====== COLORS ======
const RED = "E94560";
const RED_DIM = "8B2A3A";
const YELLOW = "F5C542";
const YELLOW_DIM = "9A7D2A";
const BLUE = "4DA8DA";
const BLUE_DIM = "2E6685";
const GREEN = "3DDC84";
const GREEN_DIM = "1A6B3A";
const WHITE = "FFFFFF";
const GRAY = "888888";
const GRAY_LIGHT = "BBBBBB";
const GRAY_DARK = "333333";
const BG_PANEL = "0C0C14";
const ACCENT_RED = "FF4444";
const ACCENT_GREEN = "44FF88";

// ====== LAYOUT ======
// Two rows, each with: Vessel illustration → Arrow → Skeleton result
const row1Y = 0.6;   // Single-color row
const row2Y = 4.0;   // Multi-color row
const rowH = 3.0;

// Vessel illustration area
const vesselX = 0.4;
const vesselW = 4.8;
// Arrow area
const arrowX = 5.4;
const arrowW = 0.8;
// Skeleton area
const skelX = 6.4;
const skelW = 4.8;
// Annotation area
const annotX = 11.4;
const annotW = 1.7;

// ====== HELPER: Draw a thick vessel segment (organic look) ======
function vessel(slide, x, y, w, h, color, opacity = 100) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color, transparency: 100 - opacity },
    rectRadius: Math.min(w, h) * 0.45,
  });
}

// ====== HELPER: Draw vessel node (junction/endpoint dot) ======
function dot(slide, cx, cy, r, fillColor, lineColor, lineW = 2) {
  slide.addShape(pres.shapes.OVAL, {
    x: cx - r, y: cy - r, w: r * 2, h: r * 2,
    fill: { color: fillColor },
    line: { color: lineColor, width: lineW },
  });
}

// ====== HELPER: skeleton line ======
function skelLine(slide, x, y, w, h, color, width = 3.5) {
  slide.addShape(pres.shapes.LINE, {
    x, y, w, h,
    line: { color, width },
  });
}

// ====== ROW LABELS ======
slide.addText("a", {
  x: 0.15, y: row1Y + 0.05, w: 0.3, h: 0.35,
  fontSize: 20, fontFace: "Calibri", bold: true,
  color: WHITE, margin: 0,
});

slide.addText("b", {
  x: 0.15, y: row2Y + 0.05, w: 0.3, h: 0.35,
  fontSize: 20, fontFace: "Calibri", bold: true,
  color: WHITE, margin: 0,
});

// ====== SECTION HEADERS ======
slide.addText("Single-color Fluorescence", {
  x: vesselX, y: row1Y - 0.45, w: vesselW, h: 0.35,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: GREEN, align: "center", margin: 0,
});

slide.addText("Depth-encoded Multi-color Fluorescence", {
  x: vesselX, y: row2Y - 0.45, w: vesselW, h: 0.35,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: BLUE, align: "center", margin: 0,
});

slide.addText("Skeletonization", {
  x: skelX, y: row1Y - 0.45, w: skelW, h: 0.35,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: GRAY_LIGHT, align: "center", margin: 0,
});

slide.addText("Skeletonization (per-layer)", {
  x: skelX, y: row2Y - 0.45, w: skelW, h: 0.35,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: GRAY_LIGHT, align: "center", margin: 0,
});

// ====== ROW 1: SINGLE-COLOR ======
// Panel background
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: vesselX, y: row1Y, w: vesselW, h: rowH,
  fill: { color: BG_PANEL }, rectRadius: 0.12,
  line: { color: "1A1A2A", width: 1 },
});

// Draw vessels in single green color
// Main horizontal vessel (going left to right)
vessel(slide, vesselX + 0.3, row1Y + 0.8, 4.2, 0.55, GREEN, 65);

// Diagonal vessel crossing over (from bottom-left to upper-right)
// Segment 1: lower-left
vessel(slide, vesselX + 0.6, row1Y + 1.8, 1.5, 0.45, GREEN, 65);
// Segment 2: crossing region (overlap)
vessel(slide, vesselX + 1.7, row1Y + 1.3, 1.0, 0.5, GREEN, 75);
// Segment 3: upper-right
vessel(slide, vesselX + 2.3, row1Y + 0.6, 1.5, 0.45, GREEN, 65);

// Branching vessel from main
vessel(slide, vesselX + 3.2, row1Y + 1.15, 0.45, 1.3, GREEN, 60);

// Small branch
vessel(slide, vesselX + 1.0, row1Y + 0.55, 0.4, 0.9, GREEN, 55);

// Another small vessel
vessel(slide, vesselX + 3.8, row1Y + 1.8, 0.85, 0.4, GREEN, 55);

// "Overlap zone" highlight
slide.addShape(pres.shapes.OVAL, {
  x: vesselX + 1.8, y: row1Y + 0.85, w: 0.85, h: 0.85,
  fill: { color: "FFFFFF", transparency: 92 },
  line: { color: WHITE, width: 1.2, dashType: "dash" },
});
slide.addText("overlap", {
  x: vesselX + 1.75, y: row1Y + 1.7, w: 0.95, h: 0.2,
  fontSize: 7.5, fontFace: "Calibri", italic: true,
  color: GRAY_LIGHT, align: "center", margin: 0,
});

// Label
slide.addText("All vessels: single color\n(no depth information)", {
  x: vesselX + 0.2, y: row1Y + 2.4, w: 4.4, h: 0.45,
  fontSize: 9, fontFace: "Calibri", italic: true,
  color: GRAY, align: "center", margin: 0,
});


// ====== ROW 1: ARROW ======
slide.addShape(pres.shapes.LINE, {
  x: arrowX, y: row1Y + rowH / 2, w: arrowW, h: 0,
  line: { color: GRAY, width: 2.5, endArrowType: "triangle" },
});


// ====== ROW 1: SKELETON ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: skelX, y: row1Y, w: skelW, h: rowH,
  fill: { color: BG_PANEL }, rectRadius: 0.12,
  line: { color: "1A1A2A", width: 1 },
});

// Skeleton lines (all green)
// Main horizontal
skelLine(slide, skelX + 0.4, row1Y + 1.05, 4.0, 0, GREEN);

// Diagonal crossing
skelLine(slide, skelX + 0.8, row1Y + 2.1, 2.2, -1.5, GREEN);

// Branch down from main
skelLine(slide, skelX + 3.1, row1Y + 1.05, 0, 1.2, GREEN);

// Small branch up from main
skelLine(slide, skelX + 1.0, row1Y + 0.4, 0, 0.65, GREEN);

// Small branch right
skelLine(slide, skelX + 3.6, row1Y + 2.0, 0.8, 0, GREEN);
skelLine(slide, skelX + 3.1, row1Y + 2.0, 0.5, 0, GREEN);

// FALSE JUNCTION at crossing point
const fj_x = skelX + 2.15;
const fj_y = row1Y + 1.05;
dot(slide, fj_x, fj_y, 0.18, "440000", ACCENT_RED, 3);

// X mark on false junction
slide.addText("X", {
  x: fj_x - 0.12, y: fj_y - 0.14, w: 0.24, h: 0.28,
  fontSize: 16, fontFace: "Calibri", bold: true,
  color: ACCENT_RED, align: "center", valign: "middle", margin: 0,
});

// False junction label with arrow
slide.addShape(pres.shapes.LINE, {
  x: fj_x + 0.18, y: fj_y - 0.3, w: 0.6, h: 0.15,
  line: { color: ACCENT_RED, width: 1.5, endArrowType: "stealth" },
});
// Flip: arrow points TO the junction
slide.addShape(pres.shapes.LINE, {
  x: fj_x + 0.55, y: fj_y - 0.55, w: 0, h: 0,
  line: { color: ACCENT_RED, width: 0 },
});

slide.addText("False junction\n(vessels at different\ndepths counted as\nintersection)", {
  x: fj_x + 0.5, y: fj_y - 0.85, w: 1.6, h: 0.7,
  fontSize: 8, fontFace: "Calibri", bold: false,
  color: ACCENT_RED, align: "left", margin: 0,
});

// Real junctions (green dots)
dot(slide, skelX + 3.1, row1Y + 1.05, 0.1, GREEN_DIM, GREEN, 2);
dot(slide, skelX + 3.1, row1Y + 2.0, 0.1, GREEN_DIM, GREEN, 2);
dot(slide, skelX + 1.0, row1Y + 1.05, 0.08, GREEN_DIM, GREEN, 1.5);

// Endpoints (white dots)
dot(slide, skelX + 0.4, row1Y + 1.05, 0.07, "222222", WHITE, 1.5);
dot(slide, skelX + 4.4, row1Y + 1.05, 0.07, "222222", WHITE, 1.5);
dot(slide, skelX + 0.8, row1Y + 2.1, 0.07, "222222", WHITE, 1.5);
dot(slide, skelX + 3.0, row1Y + 0.6, 0.07, "222222", WHITE, 1.5);
dot(slide, skelX + 1.0, row1Y + 0.4, 0.07, "222222", WHITE, 1.5);
dot(slide, skelX + 4.4, row1Y + 2.0, 0.07, "222222", WHITE, 1.5);
dot(slide, skelX + 3.1, row1Y + 2.2, 0.07, "222222", WHITE, 1.5);


// ====== ROW 2: MULTI-COLOR ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: vesselX, y: row2Y, w: vesselW, h: rowH,
  fill: { color: BG_PANEL }, rectRadius: 0.12,
  line: { color: "1A1A2A", width: 1 },
});

// Draw SAME vessel shapes but in depth-encoded colors
// Main horizontal vessel = BLUE (top layer)
vessel(slide, vesselX + 0.3, row2Y + 0.8, 4.2, 0.55, BLUE, 65);

// Diagonal vessel = RED (bottom layer)
vessel(slide, vesselX + 0.6, row2Y + 1.8, 1.5, 0.45, RED, 65);
// Crossing region - show both colors
vessel(slide, vesselX + 1.85, row2Y + 1.35, 0.65, 0.45, RED, 50);
vessel(slide, vesselX + 2.3, row2Y + 0.6, 1.5, 0.45, RED, 65);

// Yellow transition zone at crossing
vessel(slide, vesselX + 1.7, row2Y + 1.55, 0.5, 0.35, YELLOW, 60);

// Branch from main (blue)
vessel(slide, vesselX + 3.2, row2Y + 1.15, 0.45, 1.3, BLUE, 60);

// Small branch (blue)
vessel(slide, vesselX + 1.0, row2Y + 0.55, 0.4, 0.9, BLUE, 55);

// Bottom vessel (red)
vessel(slide, vesselX + 3.8, row2Y + 1.8, 0.85, 0.4, RED, 55);

// "Different depths" highlight
slide.addShape(pres.shapes.OVAL, {
  x: vesselX + 1.8, y: row2Y + 0.85, w: 0.85, h: 0.85,
  fill: { color: "000000", transparency: 85 },
  line: { color: YELLOW, width: 1.2, dashType: "dash" },
});
slide.addText("different\ndepths", {
  x: vesselX + 1.75, y: row2Y + 1.7, w: 0.95, h: 0.25,
  fontSize: 7.5, fontFace: "Calibri", italic: true,
  color: YELLOW, align: "center", margin: 0,
});

// Color legend
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: vesselX + 0.2, y: row2Y + 2.35, w: 4.4, h: 0.55,
  fill: { color: "000000", transparency: 50 },
  rectRadius: 0.06,
});

// Legend items
const legY = row2Y + 2.4;
// Red
vessel(slide, vesselX + 0.4, legY + 0.08, 0.35, 0.15, RED, 80);
slide.addText("Bottom", {
  x: vesselX + 0.8, y: legY, w: 0.7, h: 0.3,
  fontSize: 8, fontFace: "Calibri", color: RED, margin: 0, valign: "middle",
});
// Yellow
vessel(slide, vesselX + 1.55, legY + 0.08, 0.35, 0.15, YELLOW, 80);
slide.addText("Middle", {
  x: vesselX + 1.95, y: legY, w: 0.7, h: 0.3,
  fontSize: 8, fontFace: "Calibri", color: YELLOW, margin: 0, valign: "middle",
});
// Blue
vessel(slide, vesselX + 2.7, legY + 0.08, 0.35, 0.15, BLUE, 80);
slide.addText("Top", {
  x: vesselX + 3.1, y: legY, w: 0.5, h: 0.3,
  fontSize: 8, fontFace: "Calibri", color: BLUE, margin: 0, valign: "middle",
});
// Depth arrow
slide.addText("← Deep    Shallow →", {
  x: vesselX + 3.4, y: legY, w: 1.1, h: 0.3,
  fontSize: 7, fontFace: "Calibri", color: GRAY, margin: 0, valign: "middle",
});


// ====== ROW 2: ARROW ======
slide.addShape(pres.shapes.LINE, {
  x: arrowX, y: row2Y + rowH / 2, w: arrowW, h: 0,
  line: { color: GRAY, width: 2.5, endArrowType: "triangle" },
});
slide.addText("per-layer", {
  x: arrowX - 0.1, y: row2Y + rowH / 2 - 0.25, w: arrowW + 0.2, h: 0.2,
  fontSize: 7.5, fontFace: "Calibri", color: GRAY, align: "center", margin: 0,
});


// ====== ROW 2: SKELETON (depth-separated) ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: skelX, y: row2Y, w: skelW, h: rowH,
  fill: { color: BG_PANEL }, rectRadius: 0.12,
  line: { color: "1A1A2A", width: 1 },
});

// BLUE skeleton (top layer) - horizontal main
skelLine(slide, skelX + 0.4, row2Y + 1.05, 4.0, 0, BLUE);
// Blue branch down
skelLine(slide, skelX + 3.1, row2Y + 1.05, 0, 1.2, BLUE);
// Blue branch up
skelLine(slide, skelX + 1.0, row2Y + 0.4, 0, 0.65, BLUE);

// RED skeleton (bottom layer) - diagonal
skelLine(slide, skelX + 0.8, row2Y + 2.1, 2.2, -1.5, RED);
// Red branch
skelLine(slide, skelX + 3.1, row2Y + 2.0, 1.3, 0, RED);

// Yellow bridge at transition
skelLine(slide, skelX + 1.85, row2Y + 1.35, 0.35, -0.2, YELLOW, 2.5);

// NO FALSE JUNCTION at crossing - the lines pass through each other
const nfj_x = skelX + 2.15;
const nfj_y = row2Y + 1.05;

// "Pass-through" indicator - green circle with checkmark
dot(slide, nfj_x, nfj_y, 0.18, "002200", ACCENT_GREEN, 3);
slide.addText("✓", {
  x: nfj_x - 0.12, y: nfj_y - 0.14, w: 0.24, h: 0.28,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0,
});

// Label
slide.addShape(pres.shapes.LINE, {
  x: nfj_x + 0.18, y: nfj_y - 0.3, w: 0.6, h: 0.15,
  line: { color: ACCENT_GREEN, width: 1.5 },
});

slide.addText("No false junction\n(different layers =\nindependent vessels)", {
  x: nfj_x + 0.5, y: nfj_y - 0.7, w: 1.6, h: 0.55,
  fontSize: 8, fontFace: "Calibri",
  color: ACCENT_GREEN, align: "left", margin: 0,
});

// Real junctions for blue layer
dot(slide, skelX + 3.1, row2Y + 1.05, 0.1, BLUE_DIM, BLUE, 2);
dot(slide, skelX + 1.0, row2Y + 1.05, 0.08, BLUE_DIM, BLUE, 1.5);

// Junction for red layer
dot(slide, skelX + 3.1, row2Y + 2.0, 0.1, RED_DIM, RED, 2);

// Endpoints
dot(slide, skelX + 0.4, row2Y + 1.05, 0.07, "222222", BLUE, 1.5);
dot(slide, skelX + 4.4, row2Y + 1.05, 0.07, "222222", BLUE, 1.5);
dot(slide, skelX + 0.8, row2Y + 2.1, 0.07, "222222", RED, 1.5);
dot(slide, skelX + 3.0, row2Y + 0.6, 0.07, "222222", RED, 1.5);
dot(slide, skelX + 1.0, row2Y + 0.4, 0.07, "222222", BLUE, 1.5);
dot(slide, skelX + 4.4, row2Y + 2.0, 0.07, "222222", RED, 1.5);
dot(slide, skelX + 3.1, row2Y + 2.2, 0.07, "222222", BLUE, 1.5);

// Skeleton legend
const skLegY = row2Y + 2.55;
slide.addShape(pres.shapes.OVAL, {
  x: skelX + 0.5, y: skLegY, w: 0.15, h: 0.15,
  fill: { color: GREEN_DIM }, line: { color: GREEN, width: 1.5 },
});
slide.addText("Junction", {
  x: skelX + 0.7, y: skLegY - 0.02, w: 0.7, h: 0.2,
  fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0,
});

dot(slide, skelX + 1.65, skLegY + 0.075, 0.06, "222222", WHITE, 1.5);
slide.addText("Endpoint", {
  x: skelX + 1.75, y: skLegY - 0.02, w: 0.7, h: 0.2,
  fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0,
});

slide.addShape(pres.shapes.LINE, {
  x: skelX + 2.6, y: skLegY + 0.075, w: 0.3, h: 0,
  line: { color: GRAY_LIGHT, width: 3 },
});
slide.addText("Skeleton", {
  x: skelX + 2.95, y: skLegY - 0.02, w: 0.7, h: 0.2,
  fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0,
});


// ====== RIGHT-SIDE COMPARISON ANNOTATIONS ======
// Row 1 annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: annotX, y: row1Y + 0.3, w: annotW, h: 2.4,
  fill: { color: "110808" }, rectRadius: 0.1,
  line: { color: "331111", width: 1 },
});

slide.addText("Limitations", {
  x: annotX, y: row1Y + 0.35, w: annotW, h: 0.25,
  fontSize: 10, fontFace: "Calibri", bold: true,
  color: ACCENT_RED, align: "center", margin: 0,
});

slide.addText([
  { text: "Junctions", options: { bold: true, fontSize: 9, color: WHITE, breakLine: true } },
  { text: "Over-counted\n(false positives at\noverlap regions)\n", options: { fontSize: 7.5, color: GRAY_LIGHT, breakLine: true } },
  { text: "Total Length", options: { bold: true, fontSize: 9, color: WHITE, breakLine: true } },
  { text: "Under-estimated\n(merged segments)\n", options: { fontSize: 7.5, color: GRAY_LIGHT, breakLine: true } },
  { text: "Depth Info", options: { bold: true, fontSize: 9, color: WHITE, breakLine: true } },
  { text: "None", options: { fontSize: 7.5, color: GRAY_LIGHT } },
], {
  x: annotX + 0.1, y: row1Y + 0.65, w: annotW - 0.2, h: 2.0,
  margin: 0, paraSpaceAfter: 1,
});


// Row 2 annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: annotX, y: row2Y + 0.3, w: annotW, h: 2.4,
  fill: { color: "081108" }, rectRadius: 0.1,
  line: { color: "113311", width: 1 },
});

slide.addText("Advantages", {
  x: annotX, y: row2Y + 0.35, w: annotW, h: 0.25,
  fontSize: 10, fontFace: "Calibri", bold: true,
  color: ACCENT_GREEN, align: "center", margin: 0,
});

slide.addText([
  { text: "Junctions", options: { bold: true, fontSize: 9, color: WHITE, breakLine: true } },
  { text: "Accurate\n(per-layer independent\ncounting)\n", options: { fontSize: 7.5, color: GRAY_LIGHT, breakLine: true } },
  { text: "Total Length", options: { bold: true, fontSize: 9, color: WHITE, breakLine: true } },
  { text: "Recovered\n(hidden segments\nrevealed)\n", options: { fontSize: 7.5, color: GRAY_LIGHT, breakLine: true } },
  { text: "Depth Info", options: { bold: true, fontSize: 9, color: WHITE, breakLine: true } },
  { text: "Preserved\n(3D from 2D)", options: { fontSize: 7.5, color: GRAY_LIGHT } },
], {
  x: annotX + 0.1, y: row2Y + 0.65, w: annotW - 0.2, h: 2.0,
  margin: 0, paraSpaceAfter: 1,
});


// ====== DIVIDER LINE between rows ======
slide.addShape(pres.shapes.LINE, {
  x: 0.4, y: (row1Y + rowH + row2Y - 0.45) / 2, w: 12.5, h: 0,
  line: { color: GRAY_DARK, width: 0.8, dashType: "lgDash" },
});


// ====== SAVE ======
pres.writeFile({ fileName: "depth_schematic_v2.pptx" })
  .then(() => console.log("Created: depth_schematic_v2.pptx"))
  .catch(err => console.error(err));
