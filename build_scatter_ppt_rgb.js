const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "A4", width: 7.5, height: 10.0 });
pres.layout = "A4";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\scatter_panels"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_Perlayer_Concordance_RGB.pptx"
);

const M = 0.25;

// ==================== SLIDE 1: PBBDM + LBBDM (Diffusion) ====================
let s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

s1.addText("Per-layer Concordance: Diffusion Models", {
  x: M, y: 0.08, w: 7.0, h: 0.3,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

s1.addText("RGB Channel Separation (R>B = Bottom layer, B>R = Top layer) | N=228", {
  x: M, y: 0.35, w: 7.0, h: 0.2,
  fontSize: 8, fontFace: "Arial", color: "888888", align: "center",
});

const metrics = ["area", "length", "junctions", "endpoints"];
const metLabels = ["Vessel Area", "Vessel Length", "Junctions", "Endpoints"];
const iW = 1.62, iH = 1.62, gx = 0.08, gy = 0.12;

// PBBDM row
s1.addText("PBBDM", {
  x: 0.02, y: 1.0 + iH/2 - 0.1, w: 0.5, h: 0.3,
  fontSize: 9, fontFace: "Arial", color: "2980b9", bold: true, rotate: 270, align: "center",
});
metrics.forEach((met, i) => {
  const x = M + 0.15 + i * (iW + gx);
  if (i === 0 || true) {
    s1.addText(metLabels[i], {
      x, y: 0.55, w: iW, h: 0.15,
      fontSize: 7, fontFace: "Arial", color: "333333", bold: true, align: "center",
    });
  }
  s1.addImage({ path: path.join(P, `scatter_PBBDM_${met}.png`), x, y: 0.72, w: iW, h: iH });
});

// LBBDM row
s1.addText("LBBDM", {
  x: 0.02, y: 2.8 + iH/2 - 0.1, w: 0.5, h: 0.3,
  fontSize: 9, fontFace: "Arial", color: "8e44ad", bold: true, rotate: 270, align: "center",
});
metrics.forEach((met, i) => {
  const x = M + 0.15 + i * (iW + gx);
  s1.addImage({ path: path.join(P, `scatter_LBBDM_${met}.png`), x, y: 2.55, w: iW, h: iH });
});

// R² Summary table
const tY = 4.5;
s1.addText("R² Summary (Diffusion Models)", {
  x: M, y: tY, w: 7.0, h: 0.2,
  fontSize: 9, fontFace: "Arial", color: "333333", bold: true, align: "center",
});

const hdr = { bold: true, color: "FFFFFF", fill: { color: "34495e" }, align: "center" };
const good = { color: "27ae60", bold: true, align: "center" };
const norm = { align: "center" };
s1.addTable([
  [{ text: "", options: hdr },
   { text: "Area\nR", options: hdr }, { text: "Area\nB", options: hdr },
   { text: "Length\nR", options: hdr }, { text: "Length\nB", options: hdr },
   { text: "JN\nR", options: hdr }, { text: "JN\nB", options: hdr },
   { text: "EP\nR", options: hdr }, { text: "EP\nB", options: hdr }],
  [{ text: "PBBDM", options: { bold: true, color: "2980b9" } },
   { text: "0.879", options: good }, { text: "0.973", options: good },
   { text: "0.841", options: good }, { text: "0.933", options: good },
   { text: "0.472", options: norm }, { text: "0.769", options: good },
   { text: "0.564", options: norm }, { text: "0.560", options: norm }],
  [{ text: "LBBDM", options: { bold: true, color: "8e44ad" } },
   { text: "0.903", options: good }, { text: "0.979", options: good },
   { text: "0.893", options: good }, { text: "0.939", options: good },
   { text: "0.581", options: norm }, { text: "0.776", options: good },
   { text: "0.542", options: norm }, { text: "0.611", options: norm }],
], {
  x: 0.3, y: tY + 0.22, w: 6.9, fontSize: 7, fontFace: "Arial",
  border: { type: "solid", pt: 0.5, color: "CCCCCC" },
  colW: [0.7, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72],
  rowH: 0.25, autoPage: false,
});

// ==================== SLIDE 2: GANs ====================
let s2 = pres.addSlide();
s2.background = { color: "FFFFFF" };

s2.addText("Per-layer Concordance: GAN Models", {
  x: M, y: 0.08, w: 7.0, h: 0.3,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});
s2.addText("RGB Channel Separation (R>B = Bottom layer, B>R = Top layer) | N=228", {
  x: M, y: 0.35, w: 7.0, h: 0.2,
  fontSize: 8, fontFace: "Arial", color: "888888", align: "center",
});

const gans = ["WGANGP", "Pix2pix", "LSGAN"];
const ganColors = ["#e67e22", "#27ae60", "#c0392b"];
const ganColorHex = ["e67e22", "27ae60", "c0392b"];
const iW2 = 1.62, iH2 = 1.62;

gans.forEach((mname, row) => {
  const yBase = 0.55 + row * (iH2 + 0.4);
  s2.addText(mname, {
    x: 0.02, y: yBase + 0.15 + iH2/2 - 0.1, w: 0.5, h: 0.3,
    fontSize: 9, fontFace: "Arial", color: ganColorHex[row], bold: true, rotate: 270, align: "center",
  });
  metrics.forEach((met, i) => {
    const x = M + 0.15 + i * (iW2 + gx);
    if (row === 0) {
      s2.addText(metLabels[i], {
        x, y: yBase, w: iW2, h: 0.15,
        fontSize: 7, fontFace: "Arial", color: "333333", bold: true, align: "center",
      });
    }
    s2.addImage({ path: path.join(P, `scatter_${mname}_${met}.png`), x, y: yBase + 0.15, w: iW2, h: iH2 });
  });
});

// R² Summary table for GANs
const tY2 = 6.4;
s2.addText("R² Summary (GAN Models)", {
  x: M, y: tY2, w: 7.0, h: 0.2,
  fontSize: 9, fontFace: "Arial", color: "333333", bold: true, align: "center",
});

s2.addTable([
  [{ text: "", options: hdr },
   { text: "Area\nR", options: hdr }, { text: "Area\nB", options: hdr },
   { text: "Length\nR", options: hdr }, { text: "Length\nB", options: hdr },
   { text: "JN\nR", options: hdr }, { text: "JN\nB", options: hdr },
   { text: "EP\nR", options: hdr }, { text: "EP\nB", options: hdr }],
  [{ text: "WGANGP", options: { bold: true, color: "e67e22" } },
   { text: "0.618", options: norm }, { text: "0.802", options: norm },
   { text: "0.596", options: norm }, { text: "0.700", options: norm },
   { text: "0.440", options: norm }, { text: "0.368", options: norm },
   { text: "0.283", options: norm }, { text: "0.306", options: norm }],
  [{ text: "Pix2pix", options: { bold: true, color: "27ae60" } },
   { text: "0.846", options: norm }, { text: "0.901", options: norm },
   { text: "0.828", options: norm }, { text: "0.881", options: norm },
   { text: "0.555", options: norm }, { text: "0.674", options: norm },
   { text: "0.455", options: norm }, { text: "0.489", options: norm }],
  [{ text: "LSGAN", options: { bold: true, color: "c0392b" } },
   { text: "0.755", options: norm }, { text: "0.801", options: norm },
   { text: "0.698", options: norm }, { text: "0.792", options: norm },
   { text: "0.315", options: norm }, { text: "0.475", options: norm },
   { text: "0.504", options: norm }, { text: "0.489", options: norm }],
], {
  x: 0.3, y: tY2 + 0.22, w: 6.9, fontSize: 7, fontFace: "Arial",
  border: { type: "solid", pt: 0.5, color: "CCCCCC" },
  colW: [0.7, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72],
  rowH: 0.25, autoPage: false,
});

// Comparison note
s2.addText("Diffusion models (PBBDM, LBBDM) achieve R² > 0.84 for Area/Length across both layers,\n" +
           "while GANs show substantially lower concordance especially for Top layer (B>R) Junctions.", {
  x: M, y: 7.5, w: 7.0, h: 0.4,
  fontSize: 7, fontFace: "Arial", color: "555555", italic: true, align: "center",
});

// ==================== SLIDE 3: All models comparison ====================
let s3 = pres.addSlide();
s3.background = { color: "FFFFFF" };

s3.addText("R² Comparison Across All Models", {
  x: M, y: 0.08, w: 7.0, h: 0.3,
  fontSize: 13, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

// Combined figure
s3.addImage({
  path: path.join(
    "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
    "Figure_perlayer_concordance_rgb.png"
  ),
  x: 0.1, y: 0.45, w: 7.3, h: 9.1,
});

// Save
pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
