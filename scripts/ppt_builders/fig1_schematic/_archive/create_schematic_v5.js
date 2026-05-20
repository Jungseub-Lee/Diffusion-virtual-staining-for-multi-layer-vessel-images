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

function bezier(p0, p1, p2, p3, steps = 14) {
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

function offset(pts, ox, oy) {
  return pts.map(p => [p[0] + ox, p[1] + oy]);
}

// ====== VESSEL PATHS ======
// Design: Two vessel trees that OVERLAP for a stretch (not just cross at a point)
// This makes the "merged skeleton" vs "2 separate skeletons" length difference obvious

// VESSEL A (TOP LAYER → Blue in multi): main trunk curves across with branches
const A_main = bezier([0.25, 1.5], [0.8, 1.0], [1.5, 0.6], [2.3, 0.55], 16)
  .concat(bezier([2.3, 0.55], [3.0, 0.5], [3.7, 0.55], [4.3, 0.7], 14))  // runs parallel to B here
  .concat(bezier([4.3, 0.7], [4.6, 0.8], [4.8, 1.0], [4.9, 1.3], 8));

const A_br1 = bezier([1.8, 0.6], [1.9, 0.95], [2.1, 1.4], [1.9, 1.85], 10);
const A_br2 = bezier([3.5, 0.55], [3.6, 0.3], [3.8, 0.15], [4.1, 0.1], 8);
const A_br3 = bezier([0.8, 1.15], [0.5, 1.45], [0.35, 1.8], [0.3, 2.15], 8);

// VESSEL B (BOTTOM LAYER → Red in multi): enters from below, runs PARALLEL to A in middle, then exits
// Key: vessels A and B overlap/run together from x~2.0 to x~4.0
const B_main = bezier([0.15, 2.65], [0.7, 2.2], [1.3, 1.5], [2.0, 0.9], 14)
  .concat(bezier([2.0, 0.9], [2.5, 0.65], [3.2, 0.5], [3.8, 0.55], 14))  // parallel zone with A!
  .concat(bezier([3.8, 0.55], [4.2, 0.6], [4.5, 1.0], [4.8, 1.6], 10));

const B_br1 = bezier([1.3, 1.5], [1.6, 1.85], [2.1, 2.2], [2.7, 2.45], 10);
const B_br2 = bezier([3.2, 0.5], [3.3, 1.0], [3.5, 1.6], [3.7, 2.1], 10);

const allA = [A_main, A_br1, A_br2, A_br3];
const allB = [B_main, B_br1, B_br2];

// Overlap zone: roughly x = 2.0 to 4.0, y = 0.5 to 0.7

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


// ==============================
// ROW 1: SINGLE-COLOR
// ==============================

// === VESSEL PANEL ===
const v1ox = vesselX, v1oy = row1Y;
for (const path of allA) polyline(slide, offset(path, v1ox, v1oy), GREEN, 18, 55);
for (const path of allB) polyline(slide, offset(path, v1ox, v1oy), GREEN, 15, 50);

// Overlap zone highlight
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: v1ox + 2.0, y: v1oy + 0.25, w: 2.1, h: 0.65,
  fill: { color: "FFFFFF", transparency: 92 },
  line: { color: WHITE, width: 1, dashType: "dash" },
});
slide.addText("vessels overlap — appears merged", {
  x: v1ox + 1.9, y: v1oy + 0.92, w: 2.3, h: 0.18,
  fontSize: 7, fontFace: "Calibri", italic: true, color: GRAY_LIGHT, align: "center", margin: 0,
});


// === SKELETON PANEL ===
const s1ox = skelX, s1oy = row1Y;

// In single-color skeleton, the overlapping zone MERGES into ONE skeleton line
// Draw A and B separately OUTSIDE overlap, but ONE merged line IN overlap

// A: before overlap (x < 2.0)
const A_pre = bezier([0.25, 1.5], [0.8, 1.0], [1.5, 0.6], [2.0, 0.55], 12);
// A: after overlap (x > 4.0)
const A_post = bezier([4.3, 0.7], [4.6, 0.8], [4.8, 1.0], [4.9, 1.3], 8);

// B: before overlap (x < 2.0)
const B_pre = bezier([0.15, 2.65], [0.7, 2.2], [1.3, 1.5], [2.0, 0.9], 12);
// B: after overlap (x > 4.0)
const B_post = bezier([3.8, 0.55], [4.2, 0.6], [4.5, 1.0], [4.8, 1.6], 10);

// Merged skeleton in overlap zone (single green line — represents MERGED)
const merged_skel = bezier([2.0, 0.7], [2.5, 0.55], [3.2, 0.5], [3.8, 0.55], 14)
  .concat(bezier([3.8, 0.55], [4.0, 0.58], [4.2, 0.65], [4.3, 0.7], 6));

// Draw non-overlap parts
polyline(slide, offset(A_pre, s1ox, s1oy), GREEN, 3.5);
polyline(slide, offset(A_post, s1ox, s1oy), GREEN, 3.5);
polyline(slide, offset(B_pre, s1ox, s1oy), GREEN, 3.5);
polyline(slide, offset(B_post, s1ox, s1oy), GREEN, 3.5);

// MERGED single line through overlap zone (slightly thicker to hint merger)
polyline(slide, offset(merged_skel, s1ox, s1oy), GREEN, 4);

// Branches
polyline(slide, offset(A_br1, s1ox, s1oy), GREEN, 3.5);
polyline(slide, offset(A_br2, s1ox, s1oy), GREEN, 3.5);
polyline(slide, offset(A_br3, s1ox, s1oy), GREEN, 3.5);
polyline(slide, offset(B_br1, s1ox, s1oy), GREEN, 3.5);
polyline(slide, offset(B_br2, s1ox, s1oy), GREEN, 3.5);

// FALSE JUNCTION: where B enters the merged zone (~2.0, 0.7)
const fj1_cx = s1ox + 2.0, fj1_cy = s1oy + 0.7;
dot(slide, fj1_cx, fj1_cy, 0.17, "330000", ACCENT_RED, 3);
slide.addText("✕", {
  x: fj1_cx - 0.1, y: fj1_cy - 0.12, w: 0.2, h: 0.24,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: ACCENT_RED, align: "center", valign: "middle", margin: 0,
});

// FALSE JUNCTION: where B exits the merged zone (~4.0, 0.58)
const fj2_cx = s1ox + 4.0, fj2_cy = s1oy + 0.6;
dot(slide, fj2_cx, fj2_cy, 0.17, "330000", ACCENT_RED, 3);
slide.addText("✕", {
  x: fj2_cx - 0.1, y: fj2_cy - 0.12, w: 0.2, h: 0.24,
  fontSize: 14, fontFace: "Calibri", bold: true,
  color: ACCENT_RED, align: "center", valign: "middle", margin: 0,
});

// False junction label
slide.addText([
  { text: "False junctions\n", options: { bold: true, fontSize: 9, color: ACCENT_RED } },
  { text: "Vessels at different z-depths\nmerge → false intersections", options: { fontSize: 7.5, color: "CC6666" } },
], {
  x: s1ox + 0.35, y: s1oy + 0.1, w: 1.8, h: 0.5,
  margin: 0,
});
// Arrow from label to junction
polyline(slide, [
  [s1ox + 1.55, s1oy + 0.5],
  [fj1_cx - 0.15, fj1_cy - 0.1],
], ACCENT_RED, 1.5);

// LENGTH annotation: highlight merged zone
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: s1ox + 2.05, y: s1oy + 0.3, w: 1.9, h: 0.5,
  fill: { color: "FFFFFF", transparency: 92 },
  line: { color: YELLOW, width: 1, dashType: "dash" },
});
slide.addText("Merged = 1 skeleton\n→ length lost", {
  x: s1ox + 2.05, y: s1oy + 0.82, w: 1.9, h: 0.3,
  fontSize: 7.5, fontFace: "Calibri", bold: true, italic: true,
  color: YELLOW, align: "center", margin: 0,
});

// Real junctions
dot(slide, s1ox + 1.8, s1oy + 0.6, 0.08, "1A6B3A", GREEN, 2);  // A br1
dot(slide, s1ox + 3.5, s1oy + 0.55, 0.08, "1A6B3A", GREEN, 2); // A br2
dot(slide, s1ox + 0.8, s1oy + 1.15, 0.08, "1A6B3A", GREEN, 2); // A br3
dot(slide, s1ox + 1.3, s1oy + 1.5, 0.08, "1A6B3A", GREEN, 2);  // B br1
dot(slide, s1ox + 3.2, s1oy + 0.5, 0.08, "1A6B3A", GREEN, 2);  // B br2

// Endpoints
const ep_all = [
  [0.25, 1.5], [4.9, 1.3], [1.9, 1.85], [4.1, 0.1], [0.3, 2.15],
  [0.15, 2.65], [4.8, 1.6], [2.7, 2.45], [3.7, 2.1],
];
for (const ep of ep_all) {
  dot(slide, s1ox + ep[0], s1oy + ep[1], 0.05, "222222", WHITE, 1.5);
}


// ==============================
// ROW 2: MULTI-COLOR
// ==============================

// === VESSEL PANEL ===
const v2ox = vesselX, v2oy = row2Y;
for (const path of allA) polyline(slide, offset(path, v2ox, v2oy), BLUE, 18, 55);
for (const path of allB) polyline(slide, offset(path, v2ox, v2oy), RED, 15, 50);

// Yellow transition at overlap boundaries
const yellow1 = bezier([1.9, 0.85], [2.0, 0.75], [2.1, 0.65], [2.3, 0.6], 6);
const yellow2 = bezier([3.7, 0.55], [3.8, 0.6], [3.9, 0.7], [4.0, 0.8], 6);
polyline(slide, offset(yellow1, v2ox, v2oy), YELLOW, 10, 50);
polyline(slide, offset(yellow2, v2ox, v2oy), YELLOW, 10, 50);

// Overlap highlight
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: v2ox + 2.0, y: v2oy + 0.25, w: 2.1, h: 0.65,
  fill: { color: "000000", transparency: 88 },
  line: { color: YELLOW, width: 1, dashType: "dash" },
});
slide.addText("same region — different z-depths", {
  x: v2ox + 1.9, y: v2oy + 0.92, w: 2.3, h: 0.18,
  fontSize: 7, fontFace: "Calibri", italic: true, color: YELLOW, align: "center", margin: 0,
});

// Color legend
const clegY = v2oy + 2.5;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: v2ox + 0.8, y: clegY, w: 3.6, h: 0.32,
  fill: { color: "000000", transparency: 40 }, rectRadius: 0.06,
});
polyline(slide, [[v2ox+1.0, clegY+0.16], [v2ox+1.3, clegY+0.16]], RED, 5);
slide.addText("Bottom", { x: v2ox+1.35, y: clegY+0.04, w: 0.55, h: 0.24, fontSize: 8, fontFace: "Calibri", color: RED, margin: 0, valign: "middle" });
polyline(slide, [[v2ox+1.95, clegY+0.16], [v2ox+2.25, clegY+0.16]], YELLOW, 5);
slide.addText("Middle", { x: v2ox+2.3, y: clegY+0.04, w: 0.55, h: 0.24, fontSize: 8, fontFace: "Calibri", color: YELLOW, margin: 0, valign: "middle" });
polyline(slide, [[v2ox+2.9, clegY+0.16], [v2ox+3.2, clegY+0.16]], BLUE, 5);
slide.addText("Top", { x: v2ox+3.25, y: clegY+0.04, w: 0.4, h: 0.24, fontSize: 8, fontFace: "Calibri", color: BLUE, margin: 0, valign: "middle" });
slide.addText("← Deep    Shallow →", { x: v2ox+3.6, y: clegY+0.04, w: 0.75, h: 0.24, fontSize: 6.5, fontFace: "Calibri", color: GRAY, margin: 0, valign: "middle" });


// === SKELETON PANEL ===
const s2ox = skelX, s2oy = row2Y;

// In multi-color: TWO SEPARATE skeleton lines through the overlap zone
// Blue skeleton (vessel A = top layer) — full path including through overlap
polyline(slide, offset(A_main, s2ox, s2oy), BLUE, 3.5);
polyline(slide, offset(A_br1, s2ox, s2oy), BLUE, 3.5);
polyline(slide, offset(A_br2, s2ox, s2oy), BLUE, 3.5);
polyline(slide, offset(A_br3, s2ox, s2oy), BLUE, 3.5);

// Red skeleton (vessel B = bottom layer) — full path including through overlap
polyline(slide, offset(B_main, s2ox, s2oy), RED, 3.5);
polyline(slide, offset(B_br1, s2ox, s2oy), RED, 3.5);
polyline(slide, offset(B_br2, s2ox, s2oy), RED, 3.5);

// Yellow bridges at transitions
polyline(slide, offset(yellow1, s2ox, s2oy), YELLOW, 2.5);
polyline(slide, offset(yellow2, s2ox, s2oy), YELLOW, 2.5);

// NO FALSE JUNCTION markers at the same locations
const nf1_cx = s2ox + 2.0, nf1_cy = s2oy + 0.7;
dot(slide, nf1_cx, nf1_cy, 0.17, "002200", ACCENT_GREEN, 3);
slide.addText("✓", {
  x: nf1_cx - 0.09, y: nf1_cy - 0.12, w: 0.18, h: 0.24,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0,
});

const nf2_cx = s2ox + 4.0, nf2_cy = s2oy + 0.6;
dot(slide, nf2_cx, nf2_cy, 0.17, "002200", ACCENT_GREEN, 3);
slide.addText("✓", {
  x: nf2_cx - 0.09, y: nf2_cy - 0.12, w: 0.18, h: 0.24,
  fontSize: 13, fontFace: "Calibri", bold: true,
  color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0,
});

// Label: no false junction
slide.addText([
  { text: "No false junctions\n", options: { bold: true, fontSize: 9, color: ACCENT_GREEN } },
  { text: "Different layers pass through\nindependently", options: { fontSize: 7.5, color: "66CC88" } },
], {
  x: s2ox + 0.35, y: s2oy + 0.1, w: 1.8, h: 0.45,
  margin: 0,
});
polyline(slide, [
  [s2ox + 1.55, s2oy + 0.45],
  [nf1_cx - 0.15, nf1_cy - 0.1],
], ACCENT_GREEN, 1.5);

// LENGTH annotation: highlight overlap zone showing TWO lines
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: s2ox + 2.05, y: s2oy + 0.3, w: 1.9, h: 0.5,
  fill: { color: "000000", transparency: 88 },
  line: { color: ACCENT_GREEN, width: 1, dashType: "dash" },
});
slide.addText("2 separate skeletons\n→ length recovered", {
  x: s2ox + 2.05, y: s2oy + 0.82, w: 1.9, h: 0.3,
  fontSize: 7.5, fontFace: "Calibri", bold: true, italic: true,
  color: ACCENT_GREEN, align: "center", margin: 0,
});

// Blue junctions (A branch points only)
dot(slide, s2ox + 1.8, s2oy + 0.6, 0.08, "2E6685", BLUE, 2);
dot(slide, s2ox + 3.5, s2oy + 0.55, 0.08, "2E6685", BLUE, 2);
dot(slide, s2ox + 0.8, s2oy + 1.15, 0.08, "2E6685", BLUE, 2);

// Red junctions (B branch points only)
dot(slide, s2ox + 1.3, s2oy + 1.5, 0.08, "8B2A3A", RED, 2);
dot(slide, s2ox + 3.2, s2oy + 0.5, 0.08, "8B2A3A", RED, 2);

// Endpoints (color-matched)
const epA = [[0.25, 1.5], [4.9, 1.3], [1.9, 1.85], [4.1, 0.1], [0.3, 2.15]];
const epB = [[0.15, 2.65], [4.8, 1.6], [2.7, 2.45], [3.7, 2.1]];
for (const ep of epA) dot(slide, s2ox + ep[0], s2oy + ep[1], 0.05, "222222", BLUE, 1.5);
for (const ep of epB) dot(slide, s2ox + ep[0], s2oy + ep[1], 0.05, "222222", RED, 1.5);


// ====== ARROWS ======
for (const ry of [row1Y, row2Y]) {
  slide.addShape(pres.shapes.LINE, {
    x: arrowX, y: ry + rowH / 2, w: arrowW, h: 0,
    line: { color: GRAY, width: 2, endArrowType: "triangle" },
  });
}

// ====== RIGHT ANNOTATIONS ======
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
  { text: "Over-counted\n(false positives)\n\n", options: { fontSize: 7, color: "AA7777" } },
  { text: "Length\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Under-estimated\n(merged vessels)\n\n", options: { fontSize: 7, color: "AA7777" } },
  { text: "Depth\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Lost", options: { fontSize: 7, color: "AA7777" } },
], {
  x: annotX + 0.08, y: row1Y + 0.5, w: annotW - 0.16, h: 2.3,
  margin: 0, paraSpaceAfter: 1,
});

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
  { text: "Accurate\n(per-layer)\n\n", options: { fontSize: 7, color: "77AA77" } },
  { text: "Length\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Recovered\n(separated)\n\n", options: { fontSize: 7, color: "77AA77" } },
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
dot(slide, skelX + 0.8, legY, 0.07, "333333", GRAY_LIGHT, 1.5);
slide.addText("Junction", { x: skelX + 0.92, y: legY - 0.08, w: 0.6, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
dot(slide, skelX + 1.65, legY, 0.05, "222222", WHITE, 1.5);
slide.addText("Endpoint", { x: skelX + 1.75, y: legY - 0.08, w: 0.6, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
polyline(slide, [[skelX + 2.5, legY], [skelX + 2.8, legY]], GRAY_LIGHT, 3);
slide.addText("Skeleton", { x: skelX + 2.85, y: legY - 0.08, w: 0.6, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });

dot(slide, skelX + 3.6, legY, 0.1, "330000", ACCENT_RED, 2);
slide.addText("✕", { x: skelX + 3.52, y: legY - 0.08, w: 0.16, h: 0.16, fontSize: 10, fontFace: "Calibri", bold: true, color: ACCENT_RED, align: "center", valign: "middle", margin: 0 });
slide.addText("False junction", { x: skelX + 3.75, y: legY - 0.08, w: 0.9, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: ACCENT_RED, margin: 0 });


// ====== SAVE ======
pres.writeFile({ fileName: "depth_schematic_v5.pptx" })
  .then(() => console.log("Created: depth_schematic_v5.pptx"))
  .catch(err => console.error(err));
