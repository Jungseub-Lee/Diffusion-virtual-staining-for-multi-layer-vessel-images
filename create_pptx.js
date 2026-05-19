const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();

// A4 portrait: 7.5" x 10" (approximate A4 in inches)
pres.defineLayout({ name: "A4", width: 7.5, height: 10 });
pres.layout = "A4";
pres.author = "Depth-aware Vessel Analysis";
pres.title = "Figure: Multi-color Depth-aware Vessel Analysis Pipeline";

const panelDir = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\paper_panels_depth"
);
const outDir = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output"
);

// Layout constants
const margin = 0.3;
const colW = (7.5 - margin * 2 - 0.15 * 3) / 4; // 4 columns with gaps
const gap = 0.15;
const labelSize = 7;
const titleSize = 8;

// Image dimensions (aspect ratio ~5:4 for panels)
const imgW = colW;
const imgH = colW * 0.82;

function addLabel(slide, text, x, y, opts = {}) {
  slide.addText(text, {
    x: x,
    y: y,
    w: imgW,
    h: 0.2,
    fontSize: opts.fontSize || labelSize,
    fontFace: "Arial",
    color: "333333",
    bold: true,
    align: "left",
    margin: 0,
  });
}

// ===== Slide 1: Main figure =====
let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// Title
slide.addText("Depth-aware Vessel Network Analysis Pipeline", {
  x: margin,
  y: 0.15,
  w: 7.5 - margin * 2,
  h: 0.35,
  fontSize: 13,
  fontFace: "Arial",
  color: "1a1a1a",
  bold: true,
  align: "center",
});

// Row labels
const rowLabels = [
  "Channel Separation",
  "Dominance Filter & Skeleton",
  "Gap Bridging & Quantification",
];

const rowY = [0.6, 3.0, 5.4];

// Row 0: (a) BF, (b) GT, (c) R channel, (d) B channel
const row0panels = [
  { file: "a_bf.png", label: "(a) Brightfield" },
  { file: "b_gt.png", label: "(b) Multi-color GT" },
  { file: "c_r_ch.png", label: "(c) R channel (bottom)" },
  { file: "d_b_ch.png", label: "(d) B channel (top)" },
];

row0panels.forEach((p, i) => {
  const x = margin + i * (imgW + gap);
  addLabel(slide, p.label, x, rowY[0]);
  slide.addImage({
    path: path.join(panelDir, p.file),
    x: x,
    y: rowY[0] + 0.2,
    w: imgW,
    h: imgH,
  });
});

// Row 1: (e) Dominance, (f) R skel, (g) B skel, (h) R+B overlay
const row1panels = [
  { file: "e_dom.png", label: "(e) Dominance map" },
  { file: "f_r_sk.png", label: "(f) R skeleton (R>B)" },
  { file: "g_b_sk.png", label: "(g) B skeleton (B>R)" },
  { file: "h_rb_overlay.png", label: "(h) R+B overlay" },
];

row1panels.forEach((p, i) => {
  const x = margin + i * (imgW + gap);
  addLabel(slide, p.label, x, rowY[1]);
  slide.addImage({
    path: path.join(panelDir, p.file),
    x: x,
    y: rowY[1] + 0.2,
    w: imgW,
    h: imgH,
  });
});

// Row 2: (i) Single GT, (j) R bridged, (k) B bridged, (l) Combined
const row2panels = [
  { file: "i_single.png", label: "(i) Single GT skel" },
  { file: "j_r_br.png", label: "(j) R bridged" },
  { file: "k_b_br.png", label: "(k) B bridged" },
  { file: "l_comb.png", label: "(l) R+B final" },
];

row2panels.forEach((p, i) => {
  const x = margin + i * (imgW + gap);
  addLabel(slide, p.label, x, rowY[2]);
  slide.addImage({
    path: path.join(panelDir, p.file),
    x: x,
    y: rowY[2] + 0.2,
    w: imgW,
    h: imgH,
  });
});

// Row 3: Bar chart (left half) + Summary table (right half)
const barY = 7.85;

slide.addImage({
  path: path.join(panelDir, "m_bar_chart.png"),
  x: margin,
  y: barY,
  w: 3.5,
  h: 2.0,
});

addLabel(slide, "(m) Normalized metrics (N=215)", margin, barY - 0.2);

// Summary table text
const tableX = 3.9;
addLabel(slide, "(n) Sample 1 Quantification (top 80%)", tableX, barY - 0.2);

const tableData = [
  ["", "Length", "EP", "JN"],
  ["Single GT", "2268", "20", "25"],
  ["R (R>B)", "1459", "16", "15"],
  ["B (B>R)", "1166", "19", "9"],
  ["R+B sum", "2625", "35", "24"],
  ["R+B real EP", "", "26", ""],
];

slide.addTable(tableData, {
  x: tableX,
  y: barY,
  w: 3.3,
  fontSize: 7,
  fontFace: "Arial",
  border: { type: "solid", pt: 0.5, color: "CCCCCC" },
  colW: [1.0, 0.7, 0.5, 0.5],
  rowH: 0.22,
  autoPage: false,
});

// Legend note
slide.addText(
  [
    {
      text: "Yellow",
      options: { color: "CC9900", bold: true },
    },
    { text: " = Real EP  " },
    {
      text: "Cyan",
      options: { color: "00AAAA", bold: true },
    },
    { text: " = Connected EP  " },
    {
      text: "Magenta",
      options: { color: "CC00CC", bold: true },
    },
    { text: " = Junction" },
  ],
  {
    x: tableX,
    y: barY + 1.45,
    w: 3.3,
    h: 0.25,
    fontSize: 6.5,
    fontFace: "Arial",
    color: "555555",
  }
);

// Row divider lines
[2.85, 5.25, 7.65].forEach((ly) => {
  slide.addShape(pres.shapes.LINE, {
    x: margin,
    y: ly,
    w: 7.5 - margin * 2,
    h: 0,
    line: { color: "DDDDDD", width: 0.5 },
  });
});

// Save
const outPath = path.join(outDir, "Figure_Depth_Pipeline.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Saved:", outPath);
});
