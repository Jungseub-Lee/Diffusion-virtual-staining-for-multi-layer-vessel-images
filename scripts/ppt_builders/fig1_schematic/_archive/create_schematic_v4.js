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
const BG_PANEL = "0A0A12";
const ACCENT_RED = "FF4444";
const ACCENT_GREEN = "44FF88";

// ====== LAYOUT ======
const row1Y = 0.55;
const row2Y = 3.95;
const rowH = 3.0;

const vesselX = 0.3;
const vesselW = 5.2;
const arrowX = 5.65;
const arrowW = 0.6;
const skelX = 6.4;
const skelW = 5.2;
const annotX = 11.8;
const annotW = 1.35;

// ====== HELPERS ======
function dot(slide, cx, cy, r, fillColor, lineColor, lineW = 2) {
  slide.addShape(pres.shapes.OVAL, {
    x: cx - r, y: cy - r, w: r * 2, h: r * 2,
    fill: { color: fillColor },
    line: { color: lineColor, width: lineW },
  });
}

// Draw a smooth polyline as many small segments
function polyline(slide, points, color, width, opacity = 100) {
  for (let i = 0; i < points.length - 1; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[i + 1];
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.001) continue;
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);
    slide.addShape(pres.shapes.LINE, {
      x: x1, y: y1, w: len, h: 0,
      line: { color, width, transparency: 100 - opacity },
      rotate: angle,
    });
  }
}

// Generate smooth curve points using cubic bezier approximation
function bezierCurve(p0, p1, p2, p3, steps = 12) {
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const u = 1 - t;
    const x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0];
    const y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1];
    pts.push([x, y]);
  }
  return pts;
}

// ====== DEFINE VESSEL PATHS (smooth curves) ======
// Coordinates relative to panel, will be offset

// VESSEL GROUP A (TOP LAYER → Blue in multi-color)
// Main trunk: flows from upper-left, curves gently across, with organic bends
const vesselA_main = bezierCurve(
  [0.3, 1.0], [1.2, 0.6], [2.8, 0.5], [4.2, 0.8], 20
).concat(bezierCurve(
  [4.2, 0.8], [4.5, 0.9], [4.8, 1.1], [4.9, 1.3], 8
));

// A branch 1: branches down from ~(2.2, 0.55)
const vesselA_br1 = bezierCurve(
  [2.2, 0.55], [2.3, 0.9], [2.5, 1.4], [2.3, 1.9], 12
);

// A branch 2: small sprout up from ~(3.3, 0.6)
const vesselA_br2 = bezierCurve(
  [3.3, 0.6], [3.4, 0.35], [3.6, 0.2], [3.8, 0.15], 8
);

// A branch 3: branch from main at ~(1.2, 0.75) going down-left
const vesselA_br3 = bezierCurve(
  [1.2, 0.75], [1.0, 1.1], [0.7, 1.5], [0.4, 1.8], 10
);

// VESSEL GROUP B (BOTTOM LAYER → Red in multi-color)
// Main trunk: flows from lower-left diagonally up-right, CROSSES group A
const vesselB_main = bezierCurve(
  [0.2, 2.6], [0.8, 2.2], [1.5, 1.6], [2.2, 1.2], 12
).concat(bezierCurve(
  [2.2, 1.2], [2.8, 0.9], [3.5, 0.7], [4.0, 0.8], 10  // crosses A around (2.8-3.0, 0.7)
).concat(bezierCurve(
  [4.0, 0.8], [4.3, 0.9], [4.6, 1.2], [4.8, 1.7], 8
)));

// B branch 1: from ~(1.5, 1.6) going down-right
const vesselB_br1 = bezierCurve(
  [1.5, 1.6], [1.8, 1.9], [2.3, 2.3], [2.8, 2.5], 10
);

// B branch 2: from ~(3.0, 0.75) going down
const vesselB_br2 = bezierCurve(
  [3.0, 0.75], [3.1, 1.2], [3.3, 1.7], [3.5, 2.1], 10
);

// B branch 3: small branch from (4.0, 0.8)
const vesselB_br3 = bezierCurve(
  [4.0, 0.8], [4.2, 0.5], [4.5, 0.3], [4.8, 0.25], 8
);

const allA = [vesselA_main, vesselA_br1, vesselA_br2, vesselA_br3];
const allB = [vesselB_main, vesselB_br1, vesselB_br2, vesselB_br3];

// Crossing zone is around (2.8-3.2, 0.6-0.8) where both main trunks overlap

// ====== HEADERS ======
slide.addText("a", {
  x: 0.08, y: row1Y - 0.02, w: 0.22, h: 0.3,
  fontSize: 18, fontFace: "Calibri", bold: true, color: WHITE, margin: 0,
});
slide.addText("b", {
  x: 0.08, y: row2Y - 0.02, w: 0.22, h: 0.3,
  fontSize: 18, fontFace: "Calibri", bold: true, color: WHITE, margin: 0,
});

slide.addText("Single-color Fluorescence", {
  x: vesselX, y: row1Y - 0.4, w: vesselW, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: GREEN, align: "center", margin: 0,
});
slide.addText("Multi-color Depth-encoded Fluorescence", {
  x: vesselX, y: row2Y - 0.4, w: vesselW, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: BLUE, align: "center", margin: 0,
});
slide.addText("Skeletonization", {
  x: skelX, y: row1Y - 0.4, w: skelW, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: GRAY_LIGHT, align: "center", margin: 0,
});
slide.addText("Depth-separated Skeletonization", {
  x: skelX, y: row2Y - 0.4, w: skelW, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: GRAY_LIGHT, align: "center", margin: 0,
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

// ====== ROW 1: SINGLE-COLOR VESSEL (thick, green) ======
const v1ox = vesselX, v1oy = row1Y;
// Draw all vessels as thick green lines
for (const path of allA) {
  const offset = path.map(p => [p[0] + v1ox, p[1] + v1oy]);
  polyline(slide, offset, GREEN, 18, 55);
}
for (const path of allB) {
  const offset = path.map(p => [p[0] + v1ox, p[1] + v1oy]);
  polyline(slide, offset, GREEN, 15, 50);
}

// Overlap highlight
const crossCx = v1ox + 3.0, crossCy = v1oy + 0.72;
slide.addShape(pres.shapes.OVAL, {
  x: crossCx - 0.35, y: crossCy - 0.35, w: 0.7, h: 0.7,
  fill: { color: "FFFFFF", transparency: 92 },
  line: { color: WHITE, width: 1, dashType: "dash" },
});
slide.addText("overlap", {
  x: crossCx - 0.4, y: crossCy + 0.38, w: 0.8, h: 0.18,
  fontSize: 7, fontFace: "Calibri", italic: true, color: GRAY_LIGHT, align: "center", margin: 0,
});

// Label
slide.addText("All vessels rendered in\nsame color — no depth info", {
  x: v1ox + 0.3, y: v1oy + 2.35, w: 4.6, h: 0.4,
  fontSize: 8.5, fontFace: "Calibri", italic: true, color: GRAY, align: "center", margin: 0,
});


// ====== ROW 1: SINGLE-COLOR SKELETON ======
const s1ox = skelX, s1oy = row1Y;
// All skeleton lines in green (thin)
for (const path of allA) {
  const offset = path.map(p => [p[0] + s1ox, p[1] + s1oy]);
  polyline(slide, offset, GREEN, 3.5);
}
for (const path of allB) {
  const offset = path.map(p => [p[0] + s1ox, p[1] + s1oy]);
  polyline(slide, offset, GREEN, 3.5);
}

// Junctions (real ones from each vessel tree)
// A junctions: where branches meet main
dot(slide, s1ox + 2.2, s1oy + 0.55, 0.09, "1A6B3A", GREEN, 2);  // A br1
dot(slide, s1ox + 3.3, s1oy + 0.6, 0.09, "1A6B3A", GREEN, 2);   // A br2
dot(slide, s1ox + 1.2, s1oy + 0.75, 0.09, "1A6B3A", GREEN, 2);  // A br3
// B junctions
dot(slide, s1ox + 1.5, s1oy + 1.6, 0.09, "1A6B3A", GREEN, 2);   // B br1
dot(slide, s1ox + 3.0, s1oy + 0.75, 0.09, "1A6B3A", GREEN, 2);  // B br2
dot(slide, s1ox + 4.0, s1oy + 0.8, 0.09, "1A6B3A", GREEN, 2);   // B br3

// === FALSE JUNCTION at crossing zone ===
const fj_cx = s1ox + 2.85, fj_cy = s1oy + 0.68;
dot(slide, fj_cx, fj_cy, 0.2, "330000", ACCENT_RED, 3);
slide.addText("✕", {
  x: fj_cx - 0.12, y: fj_cy - 0.14, w: 0.24, h: 0.28,
  fontSize: 15, fontFace: "Calibri", bold: true,
  color: ACCENT_RED, align: "center", valign: "middle", margin: 0,
});

// Annotation arrow pointing to false junction
polyline(slide, [
  [fj_cx + 0.2, fj_cy - 0.15],
  [fj_cx + 0.6, fj_cy - 0.5],
], ACCENT_RED, 1.5);

slide.addText([
  { text: "False junction\n", options: { bold: true, fontSize: 9, color: ACCENT_RED } },
  { text: "Overlap of vessels at\ndifferent z-depths appears\nas a single intersection", options: { fontSize: 7.5, color: "CC6666" } },
], {
  x: fj_cx + 0.5, y: fj_cy - 1.0, w: 1.8, h: 0.65,
  margin: 0,
});

// Endpoints
const endpointsA = [[0.3, 1.0], [4.9, 1.3], [2.3, 1.9], [3.8, 0.15], [0.4, 1.8]];
const endpointsB = [[0.2, 2.6], [4.8, 1.7], [2.8, 2.5], [3.5, 2.1], [4.8, 0.25]];
for (const ep of [...endpointsA, ...endpointsB]) {
  dot(slide, s1ox + ep[0], s1oy + ep[1], 0.055, "222222", WHITE, 1.5);
}


// ====== ROW 2: MULTI-COLOR VESSEL ======
const v2ox = vesselX, v2oy = row2Y;
// Vessel A = BLUE (top layer)
for (const path of allA) {
  const offset = path.map(p => [p[0] + v2ox, p[1] + v2oy]);
  polyline(slide, offset, BLUE, 18, 55);
}
// Vessel B = RED (bottom layer)
for (const path of allB) {
  const offset = path.map(p => [p[0] + v2ox, p[1] + v2oy]);
  polyline(slide, offset, RED, 15, 50);
}

// Yellow at crossing transition
const yellowPts = bezierCurve(
  [2.5, 1.05], [2.7, 0.9], [2.9, 0.8], [3.1, 0.75], 6
);
polyline(slide, yellowPts.map(p => [p[0] + v2ox, p[1] + v2oy]), YELLOW, 10, 50);

// Crossing highlight
const cross2Cx = v2ox + 3.0, cross2Cy = v2oy + 0.72;
slide.addShape(pres.shapes.OVAL, {
  x: cross2Cx - 0.35, y: cross2Cy - 0.35, w: 0.7, h: 0.7,
  fill: { color: "000000", transparency: 88 },
  line: { color: YELLOW, width: 1, dashType: "dash" },
});
slide.addText("different\nz-depths", {
  x: cross2Cx - 0.45, y: cross2Cy + 0.38, w: 0.9, h: 0.25,
  fontSize: 7, fontFace: "Calibri", italic: true, color: YELLOW, align: "center", margin: 0,
});

// Color legend
const clegY = v2oy + 2.45;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: v2ox + 0.8, y: clegY, w: 3.6, h: 0.35,
  fill: { color: "000000", transparency: 40 }, rectRadius: 0.06,
});
polyline(slide, [[v2ox+1.0, clegY+0.17], [v2ox+1.3, clegY+0.17]], RED, 5);
slide.addText("Bottom", { x: v2ox+1.35, y: clegY+0.04, w: 0.6, h: 0.25, fontSize: 8, fontFace: "Calibri", color: RED, margin: 0, valign: "middle" });
polyline(slide, [[v2ox+2.0, clegY+0.17], [v2ox+2.3, clegY+0.17]], YELLOW, 5);
slide.addText("Middle", { x: v2ox+2.35, y: clegY+0.04, w: 0.6, h: 0.25, fontSize: 8, fontFace: "Calibri", color: YELLOW, margin: 0, valign: "middle" });
polyline(slide, [[v2ox+3.0, clegY+0.17], [v2ox+3.3, clegY+0.17]], BLUE, 5);
slide.addText("Top", { x: v2ox+3.35, y: clegY+0.04, w: 0.5, h: 0.25, fontSize: 8, fontFace: "Calibri", color: BLUE, margin: 0, valign: "middle" });
slide.addText("← Deep   Shallow →", { x: v2ox+3.7, y: clegY+0.04, w: 0.65, h: 0.25, fontSize: 6.5, fontFace: "Calibri", color: GRAY, margin: 0, valign: "middle" });


// ====== ROW 2: MULTI-COLOR SKELETON ======
const s2ox = skelX, s2oy = row2Y;
// BLUE skeleton (vessel A = top layer)
for (const path of allA) {
  const offset = path.map(p => [p[0] + s2ox, p[1] + s2oy]);
  polyline(slide, offset, BLUE, 3.5);
}
// RED skeleton (vessel B = bottom layer)
for (const path of allB) {
  const offset = path.map(p => [p[0] + s2ox, p[1] + s2oy]);
  polyline(slide, offset, RED, 3.5);
}
// Yellow bridge
const yellowSkel = bezierCurve(
  [2.5, 1.05], [2.7, 0.9], [2.9, 0.8], [3.1, 0.75], 6
);
polyline(slide, yellowSkel.map(p => [p[0] + s2ox, p[1] + s2oy]), YELLOW, 2.5);

// NO FALSE JUNCTION
const nfj_cx = s2ox + 2.85, nfj_cy = s2oy + 0.68;
dot(slide, nfj_cx, nfj_cy, 0.2, "002200", ACCENT_GREEN, 3);
slide.addText("✓", {
  x: nfj_cx - 0.1, y: nfj_cy - 0.13, w: 0.2, h: 0.26,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0,
});

// Annotation
polyline(slide, [
  [nfj_cx + 0.2, nfj_cy - 0.15],
  [nfj_cx + 0.6, nfj_cy - 0.5],
], ACCENT_GREEN, 1.5);

slide.addText([
  { text: "Correctly separated\n", options: { bold: true, fontSize: 9, color: ACCENT_GREEN } },
  { text: "Different layers pass\nthrough independently —\nno false intersection", options: { fontSize: 7.5, color: "66CC88" } },
], {
  x: nfj_cx + 0.5, y: nfj_cy - 1.0, w: 1.8, h: 0.65,
  margin: 0,
});

// Blue junctions (vessel A branch points)
dot(slide, s2ox + 2.2, s2oy + 0.55, 0.09, "2E6685", BLUE, 2);
dot(slide, s2ox + 3.3, s2oy + 0.6, 0.09, "2E6685", BLUE, 2);
dot(slide, s2ox + 1.2, s2oy + 0.75, 0.09, "2E6685", BLUE, 2);
// Red junctions (vessel B branch points)
dot(slide, s2ox + 1.5, s2oy + 1.6, 0.09, "8B2A3A", RED, 2);
dot(slide, s2ox + 3.0, s2oy + 0.75, 0.09, "8B2A3A", RED, 2);
dot(slide, s2ox + 4.0, s2oy + 0.8, 0.09, "8B2A3A", RED, 2);

// Endpoints with layer-matching colors
for (const ep of endpointsA) {
  dot(slide, s2ox + ep[0], s2oy + ep[1], 0.055, "222222", BLUE, 1.5);
}
for (const ep of endpointsB) {
  dot(slide, s2ox + ep[0], s2oy + ep[1], 0.055, "222222", RED, 1.5);
}


// ====== ARROWS ======
for (const ry of [row1Y, row2Y]) {
  slide.addShape(pres.shapes.LINE, {
    x: arrowX, y: ry + rowH / 2, w: arrowW, h: 0,
    line: { color: GRAY, width: 2, endArrowType: "triangle" },
  });
}


// ====== RIGHT ANNOTATIONS ======
// Row 1: Limitations
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: annotX, y: row1Y + 0.15, w: annotW, h: 2.7,
  fill: { color: "0D0404" }, rectRadius: 0.08,
  line: { color: "2A1010", width: 1 },
});
slide.addText("Limitations", {
  x: annotX, y: row1Y + 0.2, w: annotW, h: 0.22,
  fontSize: 9.5, fontFace: "Calibri", bold: true, color: ACCENT_RED, align: "center", margin: 0,
});
slide.addText([
  { text: "Junctions\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Over-counted\n\n", options: { fontSize: 7, color: "AA7777" } },
  { text: "Length\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Under-estimated\n\n", options: { fontSize: 7, color: "AA7777" } },
  { text: "Depth\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Lost", options: { fontSize: 7, color: "AA7777" } },
], {
  x: annotX + 0.08, y: row1Y + 0.5, w: annotW - 0.16, h: 2.3,
  margin: 0, paraSpaceAfter: 1,
});

// Row 2: Advantages
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: annotX, y: row2Y + 0.15, w: annotW, h: 2.7,
  fill: { color: "040D04" }, rectRadius: 0.08,
  line: { color: "102A10", width: 1 },
});
slide.addText("Advantages", {
  x: annotX, y: row2Y + 0.2, w: annotW, h: 0.22,
  fontSize: 9.5, fontFace: "Calibri", bold: true, color: ACCENT_GREEN, align: "center", margin: 0,
});
slide.addText([
  { text: "Junctions\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Accurate\n\n", options: { fontSize: 7, color: "77AA77" } },
  { text: "Length\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Recovered\n\n", options: { fontSize: 7, color: "77AA77" } },
  { text: "Depth\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Preserved", options: { fontSize: 7, color: "77AA77" } },
], {
  x: annotX + 0.08, y: row2Y + 0.5, w: annotW - 0.16, h: 2.3,
  margin: 0, paraSpaceAfter: 1,
});


// ====== DIVIDER ======
slide.addShape(pres.shapes.LINE, {
  x: 0.3, y: (row1Y + rowH + row2Y - 0.4) / 2, w: 12.7, h: 0,
  line: { color: GRAY_DARK, width: 0.7, dashType: "lgDash" },
});

// ====== BOTTOM LEGEND ======
const legY = 7.15;
dot(slide, skelX + 1.0, legY, 0.07, "333333", GRAY_LIGHT, 1.5);
slide.addText("Junction", { x: skelX + 1.12, y: legY - 0.08, w: 0.6, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
dot(slide, skelX + 1.85, legY, 0.05, "222222", WHITE, 1.5);
slide.addText("Endpoint", { x: skelX + 1.95, y: legY - 0.08, w: 0.6, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
polyline(slide, [[skelX + 2.7, legY], [skelX + 3.0, legY]], GRAY_LIGHT, 3);
slide.addText("Skeleton", { x: skelX + 3.05, y: legY - 0.08, w: 0.6, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });


// ====== SAVE ======
pres.writeFile({ fileName: "depth_schematic_v4.pptx" })
  .then(() => console.log("Created: depth_schematic_v4.pptx"))
  .catch(err => console.error(err));
