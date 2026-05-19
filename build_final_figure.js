const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = "analysis_output";
let pres = new pptxgen();
pres.defineLayout({ name: "A4_PORTRAIT", width: 7.5, height: 10.0 });
pres.layout = "A4_PORTRAIT";

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

const lbl = { fontSize: 16, fontFace: "Arial", bold: true, color: "000000", margin: 0 };
const sublbl = { fontSize: 8, fontFace: "Arial", italic: true, color: "666666", margin: 0 };

// ============================================
// Panel (a): Schematic (ratio 1.792)
// ============================================
slide.addText("a", { x: 0.15, y: 0.08, w: 0.25, h: 0.25, ...lbl });
const schW = 6.9, schH = schW / 1.792;
const schPath = path.resolve(OUT, "panel_a_schematic.png");
if (fs.existsSync(schPath)) {
    slide.addImage({ path: schPath, x: (7.5 - schW) / 2, y: 0.3, w: schW, h: schH });
}

// ============================================
// Panel (b): Representative images
// ============================================
const panelBY = 0.3 + schH + 0.1;
slide.addText("b", { x: 0.15, y: panelBY - 0.2, w: 0.25, h: 0.25, ...lbl });

const sid = "18-19-512";
const imgSuffixes = ["bf", "fl", "single", "multi"];
const imgLabels = ["Brightfield", "Multi-color FL (GT)", "Single Skeleton", "Depth-sep Skeleton"];
const imgW = 1.6, imgH = 1.6;

for (let i = 0; i < 4; i++) {
    const x = 0.25 + i * (imgW + 0.12);
    const imgPath = path.resolve(OUT, `panel_b_${sid}_${imgSuffixes[i]}.png`);
    if (fs.existsSync(imgPath)) {
        slide.addImage({ path: imgPath, x: x, y: panelBY, w: imgW, h: imgH, sizing: { type: "cover", w: imgW, h: imgH } });
    }
    slide.addText(imgLabels[i], {
        x: x, y: panelBY - 0.16, w: imgW, h: 0.16,
        fontSize: 7, fontFace: "Arial", color: "333333", align: "center", margin: 0
    });
}

// ============================================
// Panel (c): Single vs Multi GT normalized bar + table
// ============================================
const panelCY = panelBY + imgH + 0.15;
slide.addText("c", { x: 0.15, y: panelCY, w: 0.25, h: 0.25, ...lbl });
slide.addText("Single vs Multi-color GT (normalized to Multi GT = 1.0)", {
    x: 0.45, y: panelCY + 0.03, w: 5.5, h: 0.16, ...sublbl
});

const pcPath = path.resolve(OUT, "panel_c_clean.png");
const pcW = 4.0, pcH = pcW / 1.382;
if (fs.existsSync(pcPath)) {
    slide.addImage({ path: pcPath, x: 0.2, y: panelCY + 0.2, w: pcW, h: pcH });
}

// Panel C table
const tblCY = panelCY + 0.45;
const tblC = [
    [
        { text: "Metric", options: { fill: { color: "E8E8E8" }, fontSize: 7, bold: true } },
        { text: "Ratio", options: { fill: { color: "E8E8E8" }, fontSize: 7, bold: true, align: "center" } },
        { text: "SEM", options: { fill: { color: "E8E8E8" }, fontSize: 7, bold: true, align: "center" } },
        { text: "p", options: { fill: { color: "E8E8E8" }, fontSize: 7, bold: true, align: "center" } },
    ],
    [{ text: "Junction", options: { fontSize: 7 } }, { text: "1.231", options: { fontSize: 7, align: "center" } }, { text: "\u00B10.088", options: { fontSize: 7, align: "center" } }, { text: "**", options: { fontSize: 7, align: "center" } }],
    [{ text: "Endpoint", options: { fontSize: 7 } }, { text: "1.154", options: { fontSize: 7, align: "center" } }, { text: "\u00B10.031", options: { fontSize: 7, align: "center" } }, { text: "***", options: { fontSize: 7, align: "center" } }],
    [{ text: "Length", options: { fontSize: 7 } }, { text: "0.954", options: { fontSize: 7, align: "center" } }, { text: "\u00B10.010", options: { fontSize: 7, align: "center" } }, { text: "***", options: { fontSize: 7, align: "center" } }],
    [{ text: "Area", options: { fontSize: 7 } }, { text: "1.043", options: { fontSize: 7, align: "center" } }, { text: "\u00B10.005", options: { fontSize: 7, align: "center" } }, { text: "***", options: { fontSize: 7, align: "center" } }],
];
slide.addTable(tblC, {
    x: 4.4, y: tblCY, w: 2.9,
    border: { pt: 0.5, color: "CCCCCC" },
    colW: [0.75, 0.65, 0.65, 0.45],
    valign: "middle",
});

// ============================================
// Panel (d): Model/GT bar + table
// ============================================
const panelDY = panelCY + pcH + 0.3;
slide.addText("d", { x: 0.15, y: panelDY, w: 0.25, h: 0.25, ...lbl });
slide.addText("Model fidelity: Model / GT Multi (dashed = 1.0)", {
    x: 0.45, y: panelDY + 0.03, w: 5.5, h: 0.16, ...sublbl
});

const pdPath = path.resolve(OUT, "panel_d_clean.png");
const pdW = 4.0, pdH = pdW / 1.382;
if (fs.existsSync(pdPath)) {
    slide.addImage({ path: pdPath, x: 0.2, y: panelDY + 0.2, w: pdW, h: pdH });
}

// Panel D table
const tblDY = panelDY + 0.35;
const tblD = [
    [
        { text: "", options: { fill: { color: "E8E8E8" }, fontSize: 6.5, bold: true } },
        { text: "LBBDM\nRatio", options: { fill: { color: "E8E8E8" }, fontSize: 6.5, bold: true, align: "center" } },
        { text: "R\u00B2", options: { fill: { color: "E8E8E8" }, fontSize: 6.5, bold: true, align: "center" } },
        { text: "PBBDM\nRatio", options: { fill: { color: "E8E8E8" }, fontSize: 6.5, bold: true, align: "center" } },
        { text: "R\u00B2", options: { fill: { color: "E8E8E8" }, fontSize: 6.5, bold: true, align: "center" } },
    ],
    [{ text: "Junction", options: { fontSize: 6.5 } }, { text: "0.973", options: { fontSize: 6.5, align: "center" } }, { text: "0.787", options: { fontSize: 6.5, align: "center" } }, { text: "0.874", options: { fontSize: 6.5, align: "center" } }, { text: "0.857", options: { fontSize: 6.5, align: "center" } }],
    [{ text: "Endpoint", options: { fontSize: 6.5 } }, { text: "0.766", options: { fontSize: 6.5, align: "center" } }, { text: "0.611", options: { fontSize: 6.5, align: "center" } }, { text: "0.953", options: { fontSize: 6.5, align: "center" } }, { text: "0.653", options: { fontSize: 6.5, align: "center" } }],
    [{ text: "Length", options: { fontSize: 6.5 } }, { text: "1.120", options: { fontSize: 6.5, align: "center" } }, { text: "0.953", options: { fontSize: 6.5, align: "center" } }, { text: "1.084", options: { fontSize: 6.5, align: "center" } }, { text: "0.952", options: { fontSize: 6.5, align: "center" } }],
    [{ text: "Area", options: { fontSize: 6.5 } }, { text: "1.131", options: { fontSize: 6.5, align: "center" } }, { text: "0.974", options: { fontSize: 6.5, align: "center" } }, { text: "1.097", options: { fontSize: 6.5, align: "center" } }, { text: "0.974", options: { fontSize: 6.5, align: "center" } }],
];
slide.addTable(tblD, {
    x: 4.4, y: tblDY, w: 2.9,
    border: { pt: 0.5, color: "CCCCCC" },
    colW: [0.6, 0.55, 0.4, 0.55, 0.4],
    valign: "middle",
});

// ============================================
// Panel (e): Per-layer scatter (ratio 1.943)
// ============================================
const panelEY = panelDY + pdH + 0.3;
slide.addText("e", { x: 0.15, y: panelEY, w: 0.25, h: 0.25, ...lbl });
slide.addText("Per-layer depth concordance: GT vs Model (Bottom/Top layer, N=215\u00D72)", {
    x: 0.45, y: panelEY + 0.03, w: 6.0, h: 0.16, ...sublbl
});

const scPath = path.resolve(OUT, "scatter_perlayer_final_rb.png");
const scW = 7.1, scH = scW / 1.943;
if (fs.existsSync(scPath)) {
    slide.addImage({ path: scPath, x: (7.5 - scW) / 2, y: panelEY + 0.2, w: scW, h: scH });
}

// Bottom note
slide.addText("Bottom 20% excluded (ECM baseline). Hole-filling applied for crossing vessel continuity. N=215 paired samples.", {
    x: 0.3, y: 9.75, w: 6.9, h: 0.18,
    fontSize: 6.5, fontFace: "Arial", italic: true, color: "999999", margin: 0
});

// ============================================
// SLIDE 2: Editable schematic (same as before)
// ============================================
let slide2 = pres.addSlide();
slide2.background = { color: "FFFFFF" };
slide2.addText("Panel (a) - Editable Schematic", { x: 0.3, y: 0.15, w: 6.0, h: 0.3, fontSize: 14, fontFace: "Arial", bold: true, color: "333333", margin: 0 });

const greenColor = "4CAF50", blueC = "4488CC", redC = "CC5555";
const bvPts = [[0.7,4.8],[1.0,4.2],[1.4,3.3],[1.7,2.6],[2.0,1.8],[2.2,1.2]];
const bbPts = [[1.4,3.3],[1.8,3.1],[2.3,2.9],[2.8,2.5],[3.1,1.8]];
const rvPts = [[3.2,4.8],[2.8,4.0],[2.3,3.2],[1.8,2.7],[1.3,2.2],[0.8,1.5]];
const rbPts = [[2.3,3.2],[1.9,3.5],[1.5,3.8],[1.2,3.7]];

function drawLines(sl, pts, color, w, ox) {
    for (let i = 0; i < pts.length - 1; i++) {
        sl.addShape(pres.shapes.LINE, {
            x: pts[i][0]+ox, y: pts[i][1],
            w: pts[i+1][0]-pts[i][0], h: pts[i+1][1]-pts[i][1],
            line: { color: color, width: w }
        });
    }
}

slide2.addText("Single-color", { x: 0.5, y: 0.6, w: 3.0, h: 0.3, fontSize: 12, fontFace: "Arial", bold: true, color: greenColor, align: "center", margin: 0 });
drawLines(slide2, bvPts, greenColor, 8, 0);
drawLines(slide2, bbPts, greenColor, 6, 0);
drawLines(slide2, rvPts, greenColor, 8, 0);
drawLines(slide2, rbPts, greenColor, 6, 0);
const crossPts = [[1.85, 2.85], [1.55, 2.45]];
for (const [cx, cy] of crossPts) {
    slide2.addShape(pres.shapes.OVAL, { x: cx-0.15, y: cy-0.15, w: 0.3, h: 0.3, line: { color: "FF0000", width: 2, dashType: "dash" } });
}
slide2.addText("False junction", { x: 2.1, y: 2.55, w: 1.0, h: 0.2, fontSize: 7, fontFace: "Arial", bold: true, color: "CC0000", margin: 0 });

const ox = 3.7;
slide2.addText("Multi-color (Depth-separated)", { x: 4.0, y: 0.6, w: 3.0, h: 0.3, fontSize: 12, fontFace: "Arial", bold: true, color: "336699", align: "center", margin: 0 });
drawLines(slide2, bvPts, blueC, 8, ox);
drawLines(slide2, bbPts, blueC, 6, ox);
drawLines(slide2, rvPts, redC, 8, ox);
drawLines(slide2, rbPts, redC, 6, ox);
for (const [cx, cy] of crossPts) {
    slide2.addShape(pres.shapes.OVAL, { x: cx+ox-0.15, y: cy-0.15, w: 0.3, h: 0.3, line: { color: "00AA00", width: 2, dashType: "dash" } });
}
slide2.addText("No junction\n(different depth)", { x: 5.8, y: 2.55, w: 1.2, h: 0.3, fontSize: 7, fontFace: "Arial", bold: true, color: "006600", margin: 0 });

const ly = 4.9;
slide2.addShape(pres.shapes.RECTANGLE, { x: 1.0, y: ly, w: 0.3, h: 0.12, fill: { color: redC } });
slide2.addText("Bottom (Red)", { x: 1.4, y: ly-0.02, w: 1.1, h: 0.18, fontSize: 8, fontFace: "Arial", color: "666666", margin: 0 });
slide2.addShape(pres.shapes.RECTANGLE, { x: 2.8, y: ly, w: 0.3, h: 0.12, fill: { color: blueC } });
slide2.addText("Top (Blue)", { x: 3.2, y: ly-0.02, w: 1.0, h: 0.18, fontSize: 8, fontFace: "Arial", color: "666666", margin: 0 });
slide2.addShape(pres.shapes.RECTANGLE, { x: 4.5, y: ly, w: 0.3, h: 0.12, fill: { color: greenColor } });
slide2.addText("Single-color", { x: 4.9, y: ly-0.02, w: 1.0, h: 0.18, fontSize: 8, fontFace: "Arial", color: "666666", margin: 0 });

const outputPath = path.resolve("analysis_output", "Figure_Complete.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
    console.log("Saved:", outputPath);
}).catch(err => console.error(err));
