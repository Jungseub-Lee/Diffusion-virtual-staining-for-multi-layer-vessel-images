const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();

// A4 portrait
pres.defineLayout({ name: "A4", width: 7.5, height: 10.0 });
pres.layout = "A4";
pres.author = "Depth-aware Vessel Analysis";
pres.title = "Figure: Multi-color Depth-aware Vessel Analysis Pipeline";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\paper_panels_depth"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_Depth_Pipeline_v2.pptx"
);

// ==================== SLIDE 1: Pipeline ====================
let s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

// Title
s1.addText("Depth-aware Vessel Network Analysis Pipeline", {
  x: 0.25, y: 0.12, w: 7.0, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

// === Row 1: Input ===
const margin = 0.25;
const imgW = 1.58;
const imgH = 1.3;
const gap = 0.12;
const labelH = 0.18;
const lfs = 7;

function row(slide, y, panels) {
  panels.forEach((p, i) => {
    const x = margin + i * (imgW + gap);
    slide.addText(p.label, {
      x, y, w: imgW, h: labelH, fontSize: lfs, fontFace: "Arial",
      color: "333333", bold: true, margin: 0,
    });
    slide.addImage({ path: path.join(P, p.file), x, y: y + labelH, w: imgW, h: imgH });
  });
}

// Row 1: y=0.55
row(s1, 0.55, [
  { file: "a_bf.png", label: "(a) Brightfield input" },
  { file: "b_gt.png", label: "(b) Multi-color GT" },
  { file: "c_r_ch.png", label: "(c) R channel" },
  { file: "d_b_ch.png", label: "(d) B channel" },
]);

// Row separator
s1.addShape(pres.shapes.LINE, {
  x: margin, y: 2.15, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
});

// Row 2: y=2.25
row(s1, 2.25, [
  { file: "e_dom.png", label: "(e) R-B dominance" },
  { file: "f_r_sk.png", label: "(f) R skel (R>B)" },
  { file: "g_b_sk.png", label: "(g) B skel (B>R)" },
  { file: "h_rb_overlay.png", label: "(h) R+B overlay" },
]);

s1.addShape(pres.shapes.LINE, {
  x: margin, y: 3.85, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
});

// Row 3: y=3.95
row(s1, 3.95, [
  { file: "i_single.png", label: "(i) Single GT skel" },
  { file: "j_r_br.png", label: "(j) R bridged" },
  { file: "k_b_br.png", label: "(k) B bridged" },
  { file: "l_comb.png", label: "(l) R+B combined" },
]);

s1.addShape(pres.shapes.LINE, {
  x: margin, y: 5.55, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
});

// Row 4: Bridging zoom details
s1.addText("(m) Bridging Detail (yellow = bridged gap)", {
  x: margin, y: 5.65, w: 5, h: labelH,
  fontSize: lfs, fontFace: "Arial", color: "333333", bold: true, margin: 0,
});

const zoomW = 1.5;
const zoomH = 1.5;
for (let i = 0; i < 4; i++) {
  s1.addImage({
    path: path.join(P, `zoom_bridge_${i + 1}.png`),
    x: margin + i * (zoomW + gap), y: 5.85, w: zoomW, h: zoomH,
  });
}

// Legend
s1.addText([
  { text: "Red ", options: { color: "CC3333", bold: true, fontSize: 6 } },
  { text: "= R skel  ", options: { fontSize: 6 } },
  { text: "Blue ", options: { color: "3366CC", bold: true, fontSize: 6 } },
  { text: "= B skel  ", options: { fontSize: 6 } },
  { text: "Yellow ", options: { color: "CCCC00", bold: true, fontSize: 6 } },
  { text: "= Bridged gap  ", options: { fontSize: 6 } },
  { text: "Magenta ", options: { color: "CC00CC", bold: true, fontSize: 6 } },
  { text: "= Junction  ", options: { fontSize: 6 } },
  { text: "Yellow dot ", options: { color: "CC9900", bold: true, fontSize: 6 } },
  { text: "= Real EP  ", options: { fontSize: 6 } },
  { text: "Cyan dot ", options: { color: "00AAAA", bold: true, fontSize: 6 } },
  { text: "= Connected EP", options: { fontSize: 6 } },
], {
  x: margin, y: 7.4, w: 7.0, h: 0.25,
  fontFace: "Arial", color: "555555",
});

// Row labels on left side (vertical)
const rowCenters = [1.2, 2.9, 4.6, 6.5];
const rowTitles = ["Input", "Separation", "Skeleton", "Bridging"];
rowTitles.forEach((t, i) => {
  s1.addText(t, {
    x: 0.0, y: rowCenters[i] - 0.15, w: 0.22, h: 0.7,
    fontSize: 6, fontFace: "Arial", color: "999999", bold: true,
    rotate: 270, align: "center", valign: "middle",
  });
});

// ==================== SLIDE 2: Quantification ====================
let s2 = pres.addSlide();
s2.background = { color: "FFFFFF" };

s2.addText("Vessel Network Quantification", {
  x: 0.25, y: 0.12, w: 7.0, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
});

// Chart 1: Single vs Multi GT
s2.addText("(a) Single GT vs Multi GT (N=215)", {
  x: margin, y: 0.55, w: 6, h: 0.2,
  fontSize: 8, fontFace: "Arial", color: "333333", bold: true, margin: 0,
});
s2.addImage({
  path: path.join(P, "chart1_single_vs_multi.png"),
  x: 0.5, y: 0.8, w: 6.5, h: 3.2,
});

s2.addShape(pres.shapes.LINE, {
  x: margin, y: 4.15, w: 7.0, h: 0, line: { color: "E0E0E0", width: 0.5 }
});

// Chart 2: Multi GT vs all models
s2.addText("(b) Multi-color GT vs Virtual Staining Models", {
  x: margin, y: 4.25, w: 6, h: 0.2,
  fontSize: 8, fontFace: "Arial", color: "333333", bold: true, margin: 0,
});
s2.addImage({
  path: path.join(P, "chart2_multi_vs_models.png"),
  x: 0.3, y: 4.5, w: 6.9, h: 3.3,
});

// Summary table
const tblX = 0.5;
const tblY = 8.0;
s2.addText("(c) Normalized Metrics Summary (Multi GT = 1.0)", {
  x: margin, y: tblY - 0.2, w: 6, h: 0.2,
  fontSize: 8, fontFace: "Arial", color: "333333", bold: true, margin: 0,
});

const tblData = [
  [
    { text: "Model", options: { bold: true, color: "FFFFFF", fill: { color: "333333" } } },
    { text: "Length", options: { bold: true, color: "FFFFFF", fill: { color: "333333" } } },
    { text: "Endpoints", options: { bold: true, color: "FFFFFF", fill: { color: "333333" } } },
    { text: "Junctions", options: { bold: true, color: "FFFFFF", fill: { color: "333333" } } },
  ],
  [
    { text: "GT (Multi)", options: { bold: true } },
    { text: "1.00" }, { text: "1.00" }, { text: "1.00" }
  ],
  [
    { text: "PBBDM", options: { bold: true, color: "2980b9" } },
    { text: "0.95", options: { color: "27ae60" } },
    { text: "0.93", options: { color: "27ae60" } },
    { text: "0.78" }
  ],
  [
    { text: "LBBDM", options: { bold: true, color: "8e44ad" } },
    { text: "1.00", options: { color: "27ae60" } },
    { text: "0.65" }, { text: "0.77" }
  ],
  [
    { text: "WGANGP", options: {} },
    { text: "0.80" }, { text: "0.88" }, { text: "0.48" }
  ],
  [
    { text: "Pix2pix", options: {} },
    { text: "0.81" }, { text: "0.83" }, { text: "0.42" }
  ],
  [
    { text: "LSGAN", options: {} },
    { text: "0.70" }, { text: "0.77" }, { text: "0.39" }
  ],
];

s2.addTable(tblData, {
  x: tblX, y: tblY, w: 6.5,
  fontSize: 7.5, fontFace: "Arial",
  border: { type: "solid", pt: 0.5, color: "CCCCCC" },
  colW: [1.6, 1.5, 1.5, 1.5],
  rowH: 0.22,
  autoPage: false,
});

// Highlight note
s2.addText("PBBDM achieves closest match to GT across all metrics", {
  x: tblX, y: tblY + 1.65, w: 5, h: 0.2,
  fontSize: 7, fontFace: "Arial", color: "2980b9", bold: true, italic: true, margin: 0,
});

// Save
pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
