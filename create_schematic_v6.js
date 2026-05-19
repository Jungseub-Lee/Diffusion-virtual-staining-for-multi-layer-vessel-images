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
function dot(s, cx, cy, r, fill, line, lw = 2) {
  s.addShape(pres.shapes.OVAL, {
    x: cx - r, y: cy - r, w: r * 2, h: r * 2,
    fill: { color: fill }, line: { color: line, width: lw },
  });
}

function polyline(s, pts, color, width, opacity = 100) {
  for (let i = 0; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.001) continue;
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);
    s.addShape(pres.shapes.LINE, {
      x: x1, y: y1, w: len, h: 0,
      line: { color, width, transparency: 100 - opacity },
      rotate: angle,
    });
  }
}

function bz(p0, p1, p2, p3, n = 14) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t;
    pts.push([
      u*u*u*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*p3[0],
      u*u*u*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*p3[1],
    ]);
  }
  return pts;
}

function off(pts, ox, oy) { return pts.map(p => [p[0]+ox, p[1]+oy]); }

// ====== ANGIOGENESIS VESSEL NETWORK ======
// Bottom = ECM baseline. Vessels sprout UPWARD from bottom.
// Y increases downward in PPT, so "upward growth" = decreasing Y
// Panel coords: x 0-5.2, y 0-3.0. Bottom baseline at y~2.7

// VESSEL GROUP A (TOP LAYER → Blue): sprouts from bottom, reaches higher
// Main stem A: rises from bottom-center, curves left
const A_stem = bz([2.0, 2.7], [1.9, 2.2], [1.6, 1.5], [1.3, 0.9], 18)
  .concat(bz([1.3, 0.9], [1.1, 0.5], [0.8, 0.25], [0.5, 0.15], 10));
// A branch right: from ~(1.6, 1.5) curves right-up
const A_br1 = bz([1.6, 1.5], [2.0, 1.2], [2.5, 0.8], [3.0, 0.5], 12)
  .concat(bz([3.0, 0.5], [3.3, 0.35], [3.6, 0.25], [3.9, 0.2], 8));
// A small sprout: from ~(1.3, 0.9) going left
const A_br2 = bz([1.3, 0.9], [1.0, 0.7], [0.7, 0.6], [0.4, 0.65], 8);
// A tip branch from A_br1: from ~(2.5, 0.8)
const A_br3 = bz([2.5, 0.8], [2.6, 0.5], [2.7, 0.3], [2.8, 0.15], 8);

// VESSEL GROUP B (BOTTOM LAYER → Red): sprouts from bottom-right, shorter
// Main stem B: rises from right, curves through same area as A
const B_stem = bz([3.5, 2.7], [3.3, 2.2], [2.8, 1.6], [2.3, 1.2], 16)
  .concat(bz([2.3, 1.2], [1.9, 0.9], [1.5, 0.7], [1.1, 0.55], 12));
// B branch left: from ~(2.8, 1.6)
const B_br1 = bz([2.8, 1.6], [2.4, 1.4], [1.8, 1.3], [1.3, 1.4], 10)
  .concat(bz([1.3, 1.4], [1.0, 1.45], [0.7, 1.5], [0.45, 1.6], 8));
// B branch right: from ~(2.3, 1.2) going right-up
const B_br2 = bz([2.3, 1.2], [2.7, 0.9], [3.2, 0.7], [3.7, 0.6], 10)
  .concat(bz([3.7, 0.6], [4.0, 0.55], [4.3, 0.5], [4.6, 0.45], 8));
// B small tip from stem near bottom
const B_br3 = bz([3.3, 2.2], [3.7, 2.0], [4.1, 1.8], [4.5, 1.65], 10);

// OVERLAP ZONE: A_stem and B_stem run close together around (1.5-2.3, 0.7-1.3)
// A goes through (1.6,1.5)→(1.3,0.9) and B goes through (2.3,1.2)→(1.5,0.7)
// They cross/overlap around (1.7-2.0, 0.9-1.1)

const allA = [A_stem, A_br1, A_br2, A_br3];
const allB = [B_stem, B_br1, B_br2, B_br3];

// Endpoints (tips of sprouts)
const epA_pts = [[0.5, 0.15], [3.9, 0.2], [0.4, 0.65], [2.8, 0.15]];
const epB_pts = [[1.1, 0.55], [0.45, 1.6], [4.6, 0.45], [4.5, 1.65]];
// Root points (at baseline)
const rootA = [2.0, 2.7];
const rootB = [3.5, 2.7];

// Crossing/overlap center
const crossCx = 1.85, crossCy = 1.0;

// ====== HEADERS ======
slide.addText("a", { x: 0.08, y: row1Y-0.02, w: 0.22, h: 0.3, fontSize: 18, fontFace: "Calibri", bold: true, color: WHITE, margin: 0 });
slide.addText("b", { x: 0.08, y: row2Y-0.02, w: 0.22, h: 0.3, fontSize: 18, fontFace: "Calibri", bold: true, color: WHITE, margin: 0 });

slide.addText("Single-color Fluorescence", {
  x: vesselX, y: row1Y-0.4, w: vesselW, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: GREEN, align: "center", margin: 0,
});
slide.addText("Multi-color Depth-encoded Fluorescence", {
  x: vesselX, y: row2Y-0.4, w: vesselW, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: BLUE, align: "center", margin: 0,
});
slide.addText("Skeletonization", {
  x: skelX, y: row1Y-0.4, w: skelW, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: GRAY_LIGHT, align: "center", margin: 0,
});
slide.addText("Depth-separated Skeletonization", {
  x: skelX, y: row2Y-0.4, w: skelW, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: GRAY_LIGHT, align: "center", margin: 0,
});

// ====== PANELS ======
for (const [px, py, pw] of [
  [vesselX, row1Y, vesselW], [skelX, row1Y, skelW],
  [vesselX, row2Y, vesselW], [skelX, row2Y, skelW],
]) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: px, y: py, w: pw, h: rowH,
    fill: { color: BG_PANEL }, rectRadius: 0.1,
    line: { color: "1A1A2A", width: 1 },
  });
}

// ====== ECM BASELINE (bottom of each panel) ======
for (const px of [vesselX, skelX]) {
  for (const py of [row1Y, row2Y]) {
    slide.addShape(pres.shapes.LINE, {
      x: px + 0.15, y: py + 2.72, w: vesselW - 0.3, h: 0,
      line: { color: GRAY_DARK, width: 1.5, dashType: "lgDash" },
    });
    slide.addText("ECM baseline", {
      x: px + vesselW - 1.4, y: py + 2.75, w: 1.3, h: 0.18,
      fontSize: 6.5, fontFace: "Calibri", italic: true, color: GRAY_DARK, align: "right", margin: 0,
    });
  }
}

// ==================================
// ROW 1: SINGLE-COLOR
// ==================================
const v1ox = vesselX, v1oy = row1Y;
// All vessels in GREEN (thick for vessel illustration)
for (const p of allA) polyline(slide, off(p, v1ox, v1oy), GREEN, 18, 55);
for (const p of allB) polyline(slide, off(p, v1ox, v1oy), GREEN, 15, 50);

// Overlap zone highlight
slide.addShape(pres.shapes.OVAL, {
  x: v1ox + crossCx - 0.45, y: v1oy + crossCy - 0.4, w: 0.9, h: 0.8,
  fill: { color: "FFFFFF", transparency: 92 },
  line: { color: WHITE, width: 1, dashType: "dash" },
});
slide.addText("overlap", {
  x: v1ox + crossCx - 0.4, y: v1oy + crossCy + 0.42, w: 0.8, h: 0.16,
  fontSize: 7, fontFace: "Calibri", italic: true, color: GRAY_LIGHT, align: "center", margin: 0,
});

// === SINGLE SKELETON ===
const s1ox = skelX, s1oy = row1Y;

// In single-color: overlapping zone merges into one skeleton
// A parts outside overlap
const A_stem_top = bz([1.3, 0.9], [1.1, 0.5], [0.8, 0.25], [0.5, 0.15], 10);
const A_stem_bot = bz([2.0, 2.7], [1.9, 2.2], [1.6, 1.5], [1.45, 1.25], 12);
// B parts outside overlap
const B_stem_top = bz([1.5, 0.7], [1.3, 0.6], [1.2, 0.57], [1.1, 0.55], 8);
const B_stem_bot = bz([3.5, 2.7], [3.3, 2.2], [2.8, 1.6], [2.4, 1.3], 12);
// Merged skeleton through overlap zone (single path)
const merged = bz([2.4, 1.3], [2.1, 1.1], [1.7, 0.95], [1.45, 1.25], 10);
const merged2 = bz([1.7, 0.95], [1.5, 0.8], [1.4, 0.75], [1.3, 0.9], 8);

// Draw skeleton
polyline(slide, off(A_stem_top, s1ox, s1oy), GREEN, 3.5);
polyline(slide, off(A_stem_bot, s1ox, s1oy), GREEN, 3.5);
polyline(slide, off(B_stem_top, s1ox, s1oy), GREEN, 3.5);
polyline(slide, off(B_stem_bot, s1ox, s1oy), GREEN, 3.5);
// Merged zone (slightly thicker)
polyline(slide, off(merged, s1ox, s1oy), GREEN, 4.5);
polyline(slide, off(merged2, s1ox, s1oy), GREEN, 4.5);

// Branches (all green)
for (const p of [A_br1, A_br2, A_br3, B_br1, B_br2, B_br3]) {
  polyline(slide, off(p, s1ox, s1oy), GREEN, 3.5);
}

// FALSE JUNCTIONS at merge entry/exit
const fj1x = s1ox + 1.45, fj1y = s1oy + 1.25;
const fj2x = s1ox + 1.7, fj2y = s1oy + 0.95;
dot(slide, fj1x, fj1y, 0.15, "330000", ACCENT_RED, 2.5);
slide.addText("✕", { x: fj1x-0.09, y: fj1y-0.11, w: 0.18, h: 0.22, fontSize: 13, fontFace: "Calibri", bold: true, color: ACCENT_RED, align: "center", valign: "middle", margin: 0 });
dot(slide, fj2x, fj2y, 0.15, "330000", ACCENT_RED, 2.5);
slide.addText("✕", { x: fj2x-0.09, y: fj2y-0.11, w: 0.18, h: 0.22, fontSize: 13, fontFace: "Calibri", bold: true, color: ACCENT_RED, align: "center", valign: "middle", margin: 0 });

// Label
slide.addText([
  { text: "False junctions\n", options: { bold: true, fontSize: 8.5, color: ACCENT_RED } },
  { text: "Overlapping vessels at\ndifferent z-depths\n→ false intersections", options: { fontSize: 7, color: "CC6666" } },
], { x: s1ox + 2.8, y: s1oy + 0.55, w: 1.8, h: 0.6, margin: 0 });
polyline(slide, [[s1ox+2.8, s1oy+0.85], [fj2x+0.15, fj2y+0.05]], ACCENT_RED, 1.2);

// Length annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: s1ox + 1.1, y: s1oy + 1.55, w: 1.5, h: 0.4,
  fill: { color: "000000", transparency: 50 }, rectRadius: 0.06,
  line: { color: YELLOW, width: 0.8, dashType: "dash" },
});
slide.addText("Merged = 1 skeleton\n→ length lost", {
  x: s1ox + 1.1, y: s1oy + 1.55, w: 1.5, h: 0.4,
  fontSize: 7, fontFace: "Calibri", bold: true, italic: true,
  color: YELLOW, align: "center", valign: "middle", margin: 0,
});
// Arrow from label to merged zone
polyline(slide, [[s1ox+1.85, s1oy+1.55], [s1ox+1.75, s1oy+1.15]], YELLOW, 1);

// Real junctions (green)
dot(slide, s1ox+1.6, s1oy+1.5, 0.07, "1A6B3A", GREEN, 1.5);   // A br1 junction
dot(slide, s1ox+1.3, s1oy+0.9, 0.07, "1A6B3A", GREEN, 1.5);    // A br2 junction
dot(slide, s1ox+2.5, s1oy+0.8, 0.07, "1A6B3A", GREEN, 1.5);    // A br3
dot(slide, s1ox+2.8, s1oy+1.6, 0.07, "1A6B3A", GREEN, 1.5);    // B br1
dot(slide, s1ox+2.3, s1oy+1.2, 0.07, "1A6B3A", GREEN, 1.5);    // B br2
dot(slide, s1ox+3.3, s1oy+2.2, 0.07, "1A6B3A", GREEN, 1.5);    // B br3

// Endpoints (white dots at tips)
for (const ep of [...epA_pts, ...epB_pts]) {
  dot(slide, s1ox + ep[0], s1oy + ep[1], 0.055, "222222", WHITE, 1.5);
}
// Root endpoints
dot(slide, s1ox + rootA[0], s1oy + rootA[1], 0.055, "222222", WHITE, 1.5);
dot(slide, s1ox + rootB[0], s1oy + rootB[1], 0.055, "222222", WHITE, 1.5);


// ==================================
// ROW 2: MULTI-COLOR
// ==================================
const v2ox = vesselX, v2oy = row2Y;
// A = Blue, B = Red
for (const p of allA) polyline(slide, off(p, v2ox, v2oy), BLUE, 18, 55);
for (const p of allB) polyline(slide, off(p, v2ox, v2oy), RED, 15, 50);

// Yellow at overlap transition
const yw1 = bz([2.1, 1.15], [1.95, 1.05], [1.85, 0.95], [1.7, 0.9], 6);
polyline(slide, off(yw1, v2ox, v2oy), YELLOW, 10, 45);

// Overlap zone highlight
slide.addShape(pres.shapes.OVAL, {
  x: v2ox + crossCx - 0.45, y: v2oy + crossCy - 0.4, w: 0.9, h: 0.8,
  fill: { color: "000000", transparency: 88 },
  line: { color: YELLOW, width: 1, dashType: "dash" },
});
slide.addText("different\nz-depths", {
  x: v2ox + crossCx - 0.45, y: v2oy + crossCy + 0.42, w: 0.9, h: 0.22,
  fontSize: 7, fontFace: "Calibri", italic: true, color: YELLOW, align: "center", margin: 0,
});

// Color legend
const cly = v2oy + 2.5;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: v2ox+0.6, y: cly, w: 3.8, h: 0.3,
  fill: { color: "000000", transparency: 40 }, rectRadius: 0.05,
});
polyline(slide, [[v2ox+0.8, cly+0.15], [v2ox+1.1, cly+0.15]], RED, 5);
slide.addText("Bottom", { x: v2ox+1.15, y: cly+0.03, w: 0.55, h: 0.24, fontSize: 8, fontFace: "Calibri", color: RED, margin: 0, valign: "middle" });
polyline(slide, [[v2ox+1.75, cly+0.15], [v2ox+2.05, cly+0.15]], YELLOW, 5);
slide.addText("Middle", { x: v2ox+2.1, y: cly+0.03, w: 0.55, h: 0.24, fontSize: 8, fontFace: "Calibri", color: YELLOW, margin: 0, valign: "middle" });
polyline(slide, [[v2ox+2.7, cly+0.15], [v2ox+3.0, cly+0.15]], BLUE, 5);
slide.addText("Top", { x: v2ox+3.05, y: cly+0.03, w: 0.4, h: 0.24, fontSize: 8, fontFace: "Calibri", color: BLUE, margin: 0, valign: "middle" });
slide.addText("← Deep    Shallow →", { x: v2ox+3.4, y: cly+0.03, w: 0.9, h: 0.24, fontSize: 6.5, fontFace: "Calibri", color: GRAY, margin: 0, valign: "middle" });

// === MULTI-COLOR SKELETON ===
const s2ox = skelX, s2oy = row2Y;

// Full independent skeletons — each layer keeps its own path through overlap
for (const p of allA) polyline(slide, off(p, s2ox, s2oy), BLUE, 3.5);
for (const p of allB) polyline(slide, off(p, s2ox, s2oy), RED, 3.5);
// Yellow bridge
polyline(slide, off(yw1, s2ox, s2oy), YELLOW, 2.5);

// NO FALSE JUNCTION markers (same positions as row 1)
const nf1x = s2ox + 1.45, nf1y = s2oy + 1.25;
const nf2x = s2ox + 1.7, nf2y = s2oy + 0.95;

// Instead of false junctions, show pass-through markers
dot(slide, nf1x, nf1y, 0.15, "002200", ACCENT_GREEN, 2.5);
slide.addText("✓", { x: nf1x-0.08, y: nf1y-0.1, w: 0.16, h: 0.2, fontSize: 12, fontFace: "Calibri", bold: true, color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0 });
dot(slide, nf2x, nf2y, 0.15, "002200", ACCENT_GREEN, 2.5);
slide.addText("✓", { x: nf2x-0.08, y: nf2y-0.1, w: 0.16, h: 0.2, fontSize: 12, fontFace: "Calibri", bold: true, color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0 });

// Label
slide.addText([
  { text: "No false junctions\n", options: { bold: true, fontSize: 8.5, color: ACCENT_GREEN } },
  { text: "Different layers pass\nthrough independently", options: { fontSize: 7, color: "66CC88" } },
], { x: s2ox + 2.8, y: s2oy + 0.55, w: 1.8, h: 0.5, margin: 0 });
polyline(slide, [[s2ox+2.8, s2oy+0.8], [nf2x+0.15, nf2y+0.05]], ACCENT_GREEN, 1.2);

// Length annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: s2ox + 1.1, y: s2oy + 1.55, w: 1.5, h: 0.4,
  fill: { color: "000000", transparency: 50 }, rectRadius: 0.06,
  line: { color: ACCENT_GREEN, width: 0.8, dashType: "dash" },
});
slide.addText("2 independent skeletons\n→ length recovered", {
  x: s2ox + 1.1, y: s2oy + 1.55, w: 1.5, h: 0.4,
  fontSize: 7, fontFace: "Calibri", bold: true, italic: true,
  color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0,
});
polyline(slide, [[s2ox+1.85, s2oy+1.55], [s2ox+1.75, s2oy+1.15]], ACCENT_GREEN, 1);

// Blue junctions
dot(slide, s2ox+1.6, s2oy+1.5, 0.07, "2E6685", BLUE, 1.5);
dot(slide, s2ox+1.3, s2oy+0.9, 0.07, "2E6685", BLUE, 1.5);
dot(slide, s2ox+2.5, s2oy+0.8, 0.07, "2E6685", BLUE, 1.5);
// Red junctions
dot(slide, s2ox+2.8, s2oy+1.6, 0.07, "8B2A3A", RED, 1.5);
dot(slide, s2ox+2.3, s2oy+1.2, 0.07, "8B2A3A", RED, 1.5);
dot(slide, s2ox+3.3, s2oy+2.2, 0.07, "8B2A3A", RED, 1.5);

// Endpoints (color-matched)
for (const ep of epA_pts) dot(slide, s2ox+ep[0], s2oy+ep[1], 0.055, "222222", BLUE, 1.5);
for (const ep of epB_pts) dot(slide, s2ox+ep[0], s2oy+ep[1], 0.055, "222222", RED, 1.5);
dot(slide, s2ox+rootA[0], s2oy+rootA[1], 0.055, "222222", BLUE, 1.5);
dot(slide, s2ox+rootB[0], s2oy+rootB[1], 0.055, "222222", RED, 1.5);


// ====== ARROWS ======
for (const ry of [row1Y, row2Y]) {
  slide.addShape(pres.shapes.LINE, {
    x: arrowX, y: ry + rowH/2, w: arrowW, h: 0,
    line: { color: GRAY, width: 2, endArrowType: "triangle" },
  });
}

// ====== RIGHT ANNOTATIONS ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: annotX, y: row1Y+0.15, w: annotW, h: 2.7,
  fill: { color: "0D0404" }, rectRadius: 0.08,
  line: { color: "2A1010", width: 1 },
});
slide.addText("Limitations", {
  x: annotX, y: row1Y+0.2, w: annotW, h: 0.22,
  fontSize: 9.5, fontFace: "Calibri", bold: true, color: ACCENT_RED, align: "center", margin: 0,
});
slide.addText([
  { text: "Junctions\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Over-counted\n(false positives)\n\n", options: { fontSize: 7, color: "AA7777" } },
  { text: "Length\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Under-estimated\n(merged vessels)\n\n", options: { fontSize: 7, color: "AA7777" } },
  { text: "Endpoints\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Same\n(tips preserved)\n\n", options: { fontSize: 7, color: "AA7777" } },
  { text: "Depth\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Lost", options: { fontSize: 7, color: "AA7777" } },
], {
  x: annotX+0.08, y: row1Y+0.5, w: annotW-0.16, h: 2.3,
  margin: 0, paraSpaceAfter: 1,
});

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: annotX, y: row2Y+0.15, w: annotW, h: 2.7,
  fill: { color: "040D04" }, rectRadius: 0.08,
  line: { color: "102A10", width: 1 },
});
slide.addText("Advantages", {
  x: annotX, y: row2Y+0.2, w: annotW, h: 0.22,
  fontSize: 9.5, fontFace: "Calibri", bold: true, color: ACCENT_GREEN, align: "center", margin: 0,
});
slide.addText([
  { text: "Junctions\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Accurate\n(per-layer)\n\n", options: { fontSize: 7, color: "77AA77" } },
  { text: "Length\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Recovered\n(separated)\n\n", options: { fontSize: 7, color: "77AA77" } },
  { text: "Endpoints\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Same\n(tips preserved)\n\n", options: { fontSize: 7, color: "77AA77" } },
  { text: "Depth\n", options: { bold: true, fontSize: 8.5, color: WHITE } },
  { text: "Preserved", options: { fontSize: 7, color: "77AA77" } },
], {
  x: annotX+0.08, y: row2Y+0.5, w: annotW-0.16, h: 2.3,
  margin: 0, paraSpaceAfter: 1,
});

// ====== DIVIDER ======
slide.addShape(pres.shapes.LINE, {
  x: 0.3, y: (row1Y+rowH+row2Y-0.4)/2, w: 12.7, h: 0,
  line: { color: GRAY_DARK, width: 0.7, dashType: "lgDash" },
});

// ====== BOTTOM LEGEND ======
const legY = 7.15;
dot(slide, skelX+0.5, legY, 0.07, "333333", GRAY_LIGHT, 1.5);
slide.addText("Junction", { x: skelX+0.62, y: legY-0.08, w: 0.55, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
dot(slide, skelX+1.3, legY, 0.05, "222222", WHITE, 1.5);
slide.addText("Endpoint", { x: skelX+1.4, y: legY-0.08, w: 0.55, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
polyline(slide, [[skelX+2.1, legY], [skelX+2.4, legY]], GRAY_LIGHT, 3);
slide.addText("Skeleton", { x: skelX+2.45, y: legY-0.08, w: 0.55, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: GRAY_LIGHT, margin: 0 });
dot(slide, skelX+3.15, legY, 0.1, "330000", ACCENT_RED, 2);
slide.addText("✕", { x: skelX+3.07, y: legY-0.08, w: 0.16, h: 0.16, fontSize: 10, fontFace: "Calibri", bold: true, color: ACCENT_RED, align: "center", valign: "middle", margin: 0 });
slide.addText("False junction", { x: skelX+3.3, y: legY-0.08, w: 0.85, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: ACCENT_RED, margin: 0 });
dot(slide, skelX+4.3, legY, 0.1, "002200", ACCENT_GREEN, 2);
slide.addText("✓", { x: skelX+4.22, y: legY-0.08, w: 0.16, h: 0.16, fontSize: 10, fontFace: "Calibri", bold: true, color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0 });
slide.addText("Correctly separated", { x: skelX+4.45, y: legY-0.08, w: 1.1, h: 0.16, fontSize: 7.5, fontFace: "Calibri", color: ACCENT_GREEN, margin: 0 });

pres.writeFile({ fileName: "depth_schematic_v6.pptx" })
  .then(() => console.log("Created: depth_schematic_v6.pptx"))
  .catch(err => console.error(err));
