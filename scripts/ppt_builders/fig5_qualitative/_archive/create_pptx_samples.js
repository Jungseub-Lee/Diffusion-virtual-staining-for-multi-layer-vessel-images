const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const pres = new pptxgen();
pres.defineLayout({ name: "A4", width: 7.5, height: 10.0 });
pres.layout = "A4";

const P = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output\\paper_panels_depth\\per_sample"
);
const outFile = path.join(
  "C:\\Users\\seub1\\Desktop\\[Paper] Diffusion virtual staining paper\\analysis_output",
  "Figure_PerSample_Comparison.pptx"
);

const sids = ["1-19-716", "9-15-512", "16-18-716", "18-19-512", "1-16-512"];
const models = ["GT", "PBBDM", "LBBDM", "LSGAN", "Pix2pix", "WGANGP"];
const M = 0.2;

// Per-sample data from the generation output
const data = {
  "1-19-716": {
    GT: { rJN: 23, rEP: 17, rConn: 2, bJN: 15, bEP: 11, bConn: 2 },
    LBBDM: { rJN: 15, rEP: 14, rConn: 1, bJN: 13, bEP: 8, bConn: 1 },
    PBBDM: { rJN: 11, rEP: 8, rConn: 5, bJN: 15, bEP: 7, bConn: 4 },
    LSGAN: { rJN: 12, rEP: 18, rConn: 3, bJN: 17, bEP: 9, bConn: 3 },
    Pix2pix: { rJN: 13, rEP: 20, rConn: 2, bJN: 13, bEP: 9, bConn: 3 },
    WGANGP: { rJN: 7, rEP: 6, rConn: 5, bJN: 18, bEP: 10, bConn: 3 },
  },
  "9-15-512": {
    GT: { rJN: 5, rEP: 8, rConn: 6, bJN: 6, bEP: 4, bConn: 4 },
    LBBDM: { rJN: 2, rEP: 5, rConn: 1, bJN: 3, bEP: 4, bConn: 1 },
    PBBDM: { rJN: 2, rEP: 5, rConn: 1, bJN: 2, bEP: 3, bConn: 1 },
    LSGAN: { rJN: 5, rEP: 5, rConn: 2, bJN: 3, bEP: 9, bConn: 2 },
    Pix2pix: { rJN: 10, rEP: 7, rConn: 0, bJN: 1, bEP: 3, bConn: 0 },
    WGANGP: { rJN: 4, rEP: 3, rConn: 1, bJN: 3, bEP: 4, bConn: 1 },
  },
  "16-18-716": {
    GT: { rJN: 11, rEP: 7, rConn: 4, bJN: 28, bEP: 11, bConn: 2 },
    LBBDM: { rJN: 11, rEP: 6, rConn: 2, bJN: 18, bEP: 7, bConn: 2 },
    PBBDM: { rJN: 31, rEP: 9, rConn: 1, bJN: 24, bEP: 14, bConn: 1 },
    LSGAN: { rJN: 4, rEP: 5, rConn: 3, bJN: 24, bEP: 17, bConn: 3 },
    Pix2pix: { rJN: 23, rEP: 5, rConn: 4, bJN: 17, bEP: 8, bConn: 4 },
    WGANGP: { rJN: 12, rEP: 6, rConn: 0, bJN: 15, bEP: 14, bConn: 0 },
  },
  "18-19-512": {
    GT: { rJN: 18, rEP: 12, rConn: 4, bJN: 23, bEP: 18, bConn: 4 },
    LBBDM: { rJN: 16, rEP: 4, rConn: 1, bJN: 26, bEP: 17, bConn: 1 },
    PBBDM: { rJN: 20, rEP: 11, rConn: 2, bJN: 19, bEP: 13, bConn: 2 },
    LSGAN: { rJN: 34, rEP: 19, rConn: 2, bJN: 41, bEP: 31, bConn: 1 },
    Pix2pix: { rJN: 22, rEP: 8, rConn: 5, bJN: 38, bEP: 26, bConn: 8 },
    WGANGP: { rJN: 17, rEP: 9, rConn: 2, bJN: 18, bEP: 21, bConn: 2 },
  },
  "1-16-512": {
    GT: { rJN: 28, rEP: 20, rConn: 3, bJN: 29, bEP: 15, bConn: 2 },
    LBBDM: { rJN: 19, rEP: 13, rConn: 4, bJN: 31, bEP: 14, bConn: 4 },
    PBBDM: { rJN: 15, rEP: 16, rConn: 4, bJN: 27, bEP: 12, bConn: 4 },
    LSGAN: { rJN: 27, rEP: 12, rConn: 4, bJN: 22, bEP: 18, bConn: 4 },
    Pix2pix: { rJN: 29, rEP: 23, rConn: 9, bJN: 20, bEP: 15, bConn: 9 },
    WGANGP: { rJN: 20, rEP: 18, rConn: 6, bJN: 33, bEP: 15, bConn: 5 },
  },
};

sids.forEach((sid) => {
  let slide = pres.addSlide();
  slide.background = { color: "FFFFFF" };

  slide.addText(`Sample: ${sid}`, {
    x: M, y: 0.08, w: 7.0, h: 0.25,
    fontSize: 12, fontFace: "Arial", color: "1a1a1a", bold: true, align: "center",
  });

  // Column headers: Original | Skel R | Skel B | Combined
  const colLabels = ["Original", "Skel R", "Skel B", "R+B Combined"];
  const colColors = ["333333", "CC3333", "3366CC", "333333"];
  const imgW = 1.6, imgH = 1.32, gap = 0.1;
  const startX = (7.5 - 4 * imgW - 3 * gap) / 2;

  colLabels.forEach((l, i) => {
    slide.addText(l, {
      x: startX + i * (imgW + gap), y: 0.33, w: imgW, h: 0.15,
      fontSize: 6.5, fontFace: "Arial", color: colColors[i], bold: true, align: "center",
    });
  });

  // Rows: GT + 5 models
  models.forEach((model, row) => {
    const y = 0.5 + row * (imgH + 0.2);

    // Row label
    const isGT = model === "GT";
    slide.addText(model, {
      x: 0.02, y: y + imgH / 2 - 0.1, w: 0.4, h: 0.2,
      fontSize: 6, fontFace: "Arial", color: isGT ? "27ae60" : "666666",
      bold: isGT, align: "center",
    });

    // 4 panels
    const prefix = model === "GT" ? `${sid}_gt` : `${sid}_${model}`;
    const suffixes = ["", "_r", "_b", "_comb"];

    suffixes.forEach((suf, col) => {
      const fname = `${prefix}${suf}.png`;
      const fpath = path.join(P, fname);
      if (fs.existsSync(fpath)) {
        slide.addImage({
          path: fpath,
          x: startX + col * (imgW + gap), y, w: imgW, h: imgH,
        });
      }
    });

    // Metrics text (small, right side)
    const d = data[sid]?.[model];
    if (d) {
      const txt = `R: JN=${d.rJN} EP=${d.rEP}(+${d.rConn}c)\nB: JN=${d.bJN} EP=${d.bEP}(+${d.bConn}c)`;
      slide.addText(txt, {
        x: startX + 3 * (imgW + gap) + imgW + 0.02, y: y + 0.2,
        w: 0.7, h: 0.9,
        fontSize: 4.5, fontFace: "Consolas", color: "888888", margin: 0,
      });
    }
  });

  // Legend at bottom
  slide.addText([
    { text: "Magenta ", options: { color: "CC00CC", bold: true, fontSize: 5.5 } },
    { text: "= JN   ", options: { fontSize: 5.5 } },
    { text: "Yellow ", options: { color: "CC9900", bold: true, fontSize: 5.5 } },
    { text: "= Real EP   ", options: { fontSize: 5.5 } },
    { text: "Cyan ", options: { color: "00AAAA", bold: true, fontSize: 5.5 } },
    { text: "= Connected EP (excluded from count)", options: { fontSize: 5.5 } },
  ], {
    x: M, y: 9.65, w: 7.0, h: 0.18, fontFace: "Arial", color: "555555",
  });

  // GT highlight border
  slide.addShape(pres.shapes.RECTANGLE, {
    x: startX - 0.05, y: 0.45, w: 4 * imgW + 3 * gap + 0.1, h: imgH + 0.08,
    fill: { color: "27ae60", transparency: 95 },
    line: { color: "27ae60", width: 1.5, dashType: "dash" },
  });
});

pres.writeFile({ fileName: outFile }).then(() => {
  console.log("Saved:", outFile);
});
