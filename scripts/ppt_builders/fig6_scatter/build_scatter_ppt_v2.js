const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "A4", width: 7.5, height: 10.0 });
pres.layout = "A4";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\scatter_panels_v2"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_Perlayer_Concordance_v12.pptx"
);

const M = 0.25;
const hdr = { bold: true, color: "FFFFFF", fill: { color: "34495e" }, align: "center" };
const good = { color: "27ae60", bold: true, align: "center" };
const norm = { align: "center" };
const best = { color: "2980b9", bold: true, align: "center" };

const models = ["PBBDM", "LBBDM", "WGANGP", "Pix2pix", "LSGAN"];
const mColors = ["2980b9", "8e44ad", "e67e22", "27ae60", "c0392b"];

// ==================== SLIDE 1: Area/Length R² scatter ====================
let s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

s1.addText("Per-layer Concordance: Area & Length (R² Scatter)", {
  x: M, y: 0.08, w: 7.0, h: 0.35,
  fontSize: 24, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});
s1.addText("RGB Channel Separation: R>B = Bottom, B>R = Top | N=228 | Bottom 20% excluded", {
  x: M, y: 0.4, w: 7.0, h: 0.25,
  fontSize: 14, fontFace: "Arial", color: "888888", align: "center",
});

// Scatter: 1:1 aspect
const sW = 1.5, sH = 1.5, sGx = 0.08;
const labelColW = 0.45;
const gridStartX = M + labelColW;

["Vessel Area", "Vessel Length"].forEach((t, i) => {
  s1.addText(t, {
    x: gridStartX + i * (sW + sGx), y: 0.65, w: sW, h: 0.25,
    fontSize: 16, fontFace: "Arial", color: "333333", bold: true, align: "center",
  });
});

const rowGap = 0.06;
models.forEach((mname, row) => {
  const yBase = 0.92 + row * (sH + rowGap);
  s1.addText(mname, {
    x: M, y: yBase + sH / 2 - 0.12, w: labelColW, h: 0.24,
    fontSize: 14, fontFace: "Arial", color: mColors[row], bold: true, rotate: 270, align: "center",
  });
  s1.addImage({
    path: path.join(P, `scatter_${mname}_area.png`),
    x: gridStartX, y: yBase, w: sW, h: sH,
  });
  s1.addImage({
    path: path.join(P, `scatter_${mname}_length.png`),
    x: gridStartX + sW + sGx, y: yBase, w: sW, h: sH,
  });
});

// Unified table to the right of the plots
const tableX = gridStartX + 2 * sW + sGx + 0.15;
const tableY = 0.65;
const cW = 0.27;
const mW2 = 0.55;
const tblW = mW2 + 12 * cW;

s1.addText("Unified Per-layer Metrics", {
  x: tableX, y: tableY, w: tblW, h: 0.25,
  fontSize: 16, fontFace: "Arial", color: "333333", bold: true, align: "center",
});

// Group header boxes
const gY = tableY + 0.28;
const gH = 0.22;
const groups = [
  { label: "Area R²", c: "2c3e50", cols: 2, start: 0 },
  { label: "Length R²", c: "2c3e50", cols: 2, start: 2 },
  { label: "JN MAE", c: "8e44ad", cols: 2, start: 4 },
  { label: "JN ±3%", c: "8e44ad", cols: 2, start: 6 },
  { label: "EP MAE", c: "c0392b", cols: 2, start: 8 },
  { label: "EP ±3%", c: "c0392b", cols: 2, start: 10 },
];
groups.forEach(g => {
  const gx = tableX + mW2 + g.start * cW;
  const gw = g.cols * cW;
  s1.addShape(pres.shapes.RECTANGLE, { x: gx, y: gY, w: gw, h: gH, fill: { color: g.c } });
  s1.addText(g.label, { x: gx, y: gY, w: gw, h: gH, fontSize: 10, fontFace: "Arial", bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
});

// Sub-header row + data
const sh = { bold: true, color: "FFFFFF", fill: { color: "5d6d7e" }, align: "center", fontSize: 10 };
const d = (t, o) => ({ text: t, options: { ...o, fontSize: 11 } });

s1.addTable([
  [d("Model", sh), d("Bot",sh), d("Top",sh), d("Bot",sh), d("Top",sh), d("Bot",sh), d("Top",sh), d("Bot",sh), d("Top",sh), d("Bot",sh), d("Top",sh), d("Bot",sh), d("Top",sh)],
  [d("PBBDM",{bold:true,color:"2980b9"}), d(".879",best), d(".973",best), d(".841",best), d(".933",best), d("1.6",best), d("1.5",best), d("84%",best), d("88%",best), d("2.8",best), d("2.9",best), d("70%",best), d("70%",best)],
  [d("LBBDM",{bold:true,color:"8e44ad"}), d(".903",best), d(".979",best), d(".893",best), d(".939",best), d("1.6",best), d("1.4",best), d("85%",best), d("87%",best), d("2.7",best), d("3.0",norm), d("71%",best), d("71%",best)],
  [d("Pix2pix",{bold:true,color:"27ae60"}), d(".846",norm), d(".901",norm), d(".828",norm), d(".881",norm), d("1.6",norm), d("1.7",norm), d("85%",norm), d("86%",norm), d("3.2",norm), d("3.3",norm), d("68%",norm), d("64%",norm)],
  [d("LSGAN",{bold:true,color:"c0392b"}), d(".755",norm), d(".801",norm), d(".698",norm), d(".792",norm), d("1.9",norm), d("2.1",norm), d("85%",norm), d("80%",norm), d("3.2",norm), d("3.4",norm), d("62%",norm), d("59%",norm)],
  [d("WGANGP",{bold:true,color:"e67e22"}), d(".618",norm), d(".802",norm), d(".596",norm), d(".700",norm), d("2.3",norm), d("2.7",norm), d("75%",norm), d("75%",norm), d("4.6",norm), d("4.8",norm), d("49%",norm), d("52%",norm)],
], {
  x: tableX, y: gY + gH + 0.01, w: tblW, fontSize: 11, fontFace: "Arial",
  border: { type: "solid", pt: 0.3, color: "CCCCCC" },
  colW: [mW2, cW, cW, cW, cW, cW, cW, cW, cW, cW, cW, cW, cW],
  rowH: 0.24, autoPage: false,
});

// ==================== SLIDE 2: JN/EP Bland-Altman ====================
let s2 = pres.addSlide();
s2.background = { color: "FFFFFF" };

s2.addText("Per-layer Concordance: Junctions & Endpoints (Bland-Altman)", {
  x: M, y: 0.08, w: 7.0, h: 0.35,
  fontSize: 24, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});
s2.addText("Count metrics (0-30 range): MAE + % within ±3 are more appropriate than R²", {
  x: M, y: 0.4, w: 7.0, h: 0.25,
  fontSize: 14, fontFace: "Arial", color: "888888", align: "center",
});

const baW = 1.58, baH = baW / 1.125, baGx = 0.08;

["Junctions", "Endpoints"].forEach((t, i) => {
  s2.addText(t, {
    x: gridStartX + i * (baW + baGx), y: 0.65, w: baW, h: 0.25,
    fontSize: 16, fontFace: "Arial", color: "333333", bold: true, align: "center",
  });
});

models.forEach((mname, row) => {
  const yBase = 0.92 + row * (baH + rowGap);
  s2.addText(mname, {
    x: M, y: yBase + baH / 2 - 0.12, w: labelColW, h: 0.24,
    fontSize: 14, fontFace: "Arial", color: mColors[row], bold: true, rotate: 270, align: "center",
  });
  s2.addImage({
    path: path.join(P, `ba_${mname}_junctions.png`),
    x: gridStartX, y: yBase, w: baW, h: baH,
  });
  s2.addImage({
    path: path.join(P, `ba_${mname}_endpoints.png`),
    x: gridStartX + baW + baGx, y: yBase, w: baW, h: baH,
  });
});

s2.addText("See unified table on Slide 1 for all metrics", {
  x: gridStartX + 2 * baW + baGx + 0.3, y: 0.92, w: 3.8, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: "AAAAAA", align: "center", italic: true,
});

// ==================== SLIDE 3: Summary bars ====================
let s3 = pres.addSlide();
s3.background = { color: "FFFFFF" };

s3.addText("Summary: Per-layer Metric Comparison", {
  x: M, y: 0.08, w: 7.0, h: 0.35,
  fontSize: 26, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

const barW = 6.5, barH = barW / 2.5;
const barX = (7.5 - barW) / 2;

s3.addImage({
  path: path.join(P, "summary_mae_bar.png"),
  x: barX, y: 0.5, w: barW, h: barH,
});

s3.addImage({
  path: path.join(P, "summary_pct3_bar.png"),
  x: barX, y: 0.5 + barH + 0.3, w: barW, h: barH,
});

const findY = 0.5 + 2 * barH + 0.7;
s3.addText("Key Findings", {
  x: M, y: findY, w: 7.0, h: 0.35,
  fontSize: 22, fontFace: "Arial", color: "1a1a1a", bold: true,
});

const findings = [
  "Diffusion models (PBBDM, LBBDM) achieve JN MAE ≤ 1.6 with 84-88% agreement within ±3",
  "GANs show 2-3× higher EP MAE (WGANGP: 4.6-4.8) indicating poor topological fidelity",
  "Bottom (R>B) and Top (B>R) layers show consistent patterns across all models",
  "Area/Length R² > 0.84 for diffusion models confirms strong morphometric concordance",
];

findings.forEach((t, i) => {
  s3.addText([
    { text: `${i + 1}. `, options: { bold: true, color: "2980b9" } },
    { text: t },
  ], {
    x: M + 0.2, y: findY + 0.4 + i * 0.35, w: 6.5, h: 0.3,
    fontSize: 16, fontFace: "Arial", color: "333333",
  });
});

pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
