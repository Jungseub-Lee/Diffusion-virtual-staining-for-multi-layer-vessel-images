const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = path.resolve("analysis_output");
let pres = new pptxgen();

// A4 portrait
pres.defineLayout({ name: "A4P", width: 7.5, height: 10.63 });
pres.layout = "A4P";

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

const M = 0.25; // margin
const W = 7.5 - 2 * M; // usable width = 7.0

let y = 0.15;

// ===== Panel A: Schematic =====
slide.addText("a", { x: M, y: y, w: 0.25, h: 0.25, fontSize: 14, bold: true, fontFace: "Arial" });
// Schematic placeholder - insert manually from depth_schematic_v9.pptx
const schH = 1.6;
const schW = W;
slide.addShape(pres.shapes.RECTANGLE, {
    x: M, y: y + 0.2, w: schW, h: schH,
    fill: { color: "F5F5F5" }, line: { color: "CCCCCC", width: 1, dashType: "dash" }
});
slide.addText("Panel A: Schematic\n(Insert from depth_schematic_v9.pptx)", {
    x: M, y: y + 0.2, w: schW, h: schH,
    fontSize: 10, fontFace: "Arial", align: "center", valign: "middle", color: "999999", margin: 0
});

y += schH + 0.35;

// ===== Panel B: Representative images =====
slide.addText("b", { x: M, y: y, w: 0.25, h: 0.25, fontSize: 14, bold: true, fontFace: "Arial" });

const imgLabels = ["Brightfield", "Multi-color FL (GT)", "Single Skeleton\non BF", "Depth-sep Skeleton\non BF"];
const imgFiles = [
    path.join(OUT, "panel_b_18-19-512_bf.png"),
    path.join(OUT, "panel_b_18-19-512_fl.png"),
    path.join(OUT, "panel_b_18-19-512_single.png"),
    path.join(OUT, "panel_b_18-19-512_multi.png"),
];
const imgW = (W - 0.15 * 3) / 4;  // 4 images with gaps
const imgH = imgW;  // square

for (let i = 0; i < 4; i++) {
    const ix = M + i * (imgW + 0.15);
    slide.addText(imgLabels[i], {
        x: ix, y: y + 0.15, w: imgW, h: 0.3,
        fontSize: 6.5, bold: true, fontFace: "Arial", align: "center", color: "333333", margin: 0
    });
    slide.addImage({ path: imgFiles[i], x: ix, y: y + 0.45, w: imgW, h: imgH });
}

y += imgH + 0.65;

// ===== Panel C: Single vs Multi GT bar chart + table =====
slide.addText("c", { x: M, y: y, w: 0.25, h: 0.25, fontSize: 14, bold: true, fontFace: "Arial" });
slide.addText("Single-color vs Multi-color GT (Normalized to Single-color GT = 1.0)", {
    x: M + 0.25, y: y, w: W - 0.3, h: 0.2, fontSize: 8, italic: true, fontFace: "Arial", color: "555555", margin: 0
});

const chartCW = 3.6;
const chartCH = 2.3;
slide.addImage({ path: path.join(OUT, "panel_c_norm.png"), x: M, y: y + 0.25, w: chartCW, h: chartCH });

// Panel C table
const tableC_x = M + chartCW + 0.15;
const tableC_w = W - chartCW - 0.15;
const cellH = 0.22;
const hdrStyle = { fontSize: 7, bold: true, fontFace: "Arial", align: "center", color: "FFFFFF", fill: { color: "444444" }, margin: [1,2,1,2] };
const cellStyle = { fontSize: 7, fontFace: "Arial", align: "center", margin: [1,2,1,2] };
const cellStyleBold = { fontSize: 7, bold: true, fontFace: "Arial", align: "center", margin: [1,2,1,2] };

const tableC_rows = [
    [
        { text: "Metric", options: hdrStyle },
        { text: "Multi / Single", options: hdrStyle },
        { text: "p-value", options: hdrStyle },
    ],
    [
        { text: "Junctions", options: cellStyleBold },
        { text: "1.071", options: cellStyle },
        { text: "ns", options: cellStyle },
    ],
    [
        { text: "Endpoints", options: cellStyleBold },
        { text: "0.935", options: cellStyle },
        { text: "**", options: cellStyle },
    ],
    [
        { text: "Length", options: cellStyleBold },
        { text: "1.063", options: cellStyle },
        { text: "***", options: cellStyle },
    ],
    [
        { text: "Area", options: cellStyleBold },
        { text: "0.949", options: cellStyle },
        { text: "***", options: cellStyle },
    ],
];
slide.addTable(tableC_rows, {
    x: tableC_x, y: y + 0.45, w: tableC_w,
    rowH: [cellH, cellH, cellH, cellH, cellH],
    border: { type: "solid", pt: 0.5, color: "CCCCCC" },
    colW: [tableC_w * 0.4, tableC_w * 0.35, tableC_w * 0.25],
});

y += chartCH + 0.5;

// ===== Panel D: Model/GT ratio bar chart + table =====
slide.addText("d", { x: M, y: y, w: 0.25, h: 0.25, fontSize: 14, bold: true, fontFace: "Arial" });
slide.addText("Model Multi / GT Multi (Normalized to GT Multi-color = 1.0)", {
    x: M + 0.25, y: y, w: W - 0.3, h: 0.2, fontSize: 8, italic: true, fontFace: "Arial", color: "555555", margin: 0
});

const chartDW = 3.6;
const chartDH = 2.3;
slide.addImage({ path: path.join(OUT, "panel_d_norm.png"), x: M, y: y + 0.25, w: chartDW, h: chartDH });

// Panel D table
const tableD_x = M + chartDW + 0.15;
const tableD_w = W - chartDW - 0.15;
const hdrStyleD = { fontSize: 6.5, bold: true, fontFace: "Arial", align: "center", color: "FFFFFF", fill: { color: "444444" }, margin: [1,1,1,1] };
const cellStyleD = { fontSize: 6.5, fontFace: "Arial", align: "center", margin: [1,1,1,1] };
const cellBoldD = { fontSize: 6.5, bold: true, fontFace: "Arial", align: "center", margin: [1,1,1,1] };
const lbbdmColor = "D4762C";
const pbbdmColor = "8B6AAE";

const tableD_rows = [
    [
        { text: "Metric", options: hdrStyleD },
        { text: "LBBDM\nratio", options: { ...hdrStyleD, fill: { color: lbbdmColor } } },
        { text: "R\u00B2", options: { ...hdrStyleD, fill: { color: lbbdmColor } } },
        { text: "p", options: { ...hdrStyleD, fill: { color: lbbdmColor } } },
        { text: "PBBDM\nratio", options: { ...hdrStyleD, fill: { color: pbbdmColor } } },
        { text: "R\u00B2", options: { ...hdrStyleD, fill: { color: pbbdmColor } } },
        { text: "p", options: { ...hdrStyleD, fill: { color: pbbdmColor } } },
    ],
    [
        { text: "Junction", options: cellBoldD },
        { text: "0.973", options: cellStyleD },
        { text: "0.787", options: cellStyleD },
        { text: "ns", options: { ...cellStyleD, bold: true, color: "2E7D32" } },
        { text: "0.874", options: cellStyleD },
        { text: "0.857", options: cellStyleD },
        { text: "**", options: cellStyleD },
    ],
    [
        { text: "Endpoint", options: cellBoldD },
        { text: "0.766", options: cellStyleD },
        { text: "0.611", options: cellStyleD },
        { text: "***", options: cellStyleD },
        { text: "0.953", options: cellStyleD },
        { text: "0.653", options: cellStyleD },
        { text: "ns", options: { ...cellStyleD, bold: true, color: "2E7D32" } },
    ],
    [
        { text: "Length", options: cellBoldD },
        { text: "1.120", options: cellStyleD },
        { text: "0.953", options: { ...cellStyleD, bold: true, color: "2E7D32" } },
        { text: "***", options: cellStyleD },
        { text: "1.084", options: cellStyleD },
        { text: "0.952", options: { ...cellStyleD, bold: true, color: "2E7D32" } },
        { text: "ns", options: { ...cellStyleD, bold: true, color: "2E7D32" } },
    ],
    [
        { text: "Area", options: cellBoldD },
        { text: "1.131", options: cellStyleD },
        { text: "0.974", options: { ...cellStyleD, bold: true, color: "2E7D32" } },
        { text: "***", options: cellStyleD },
        { text: "1.097", options: cellStyleD },
        { text: "0.974", options: { ...cellStyleD, bold: true, color: "2E7D32" } },
        { text: "ns", options: { ...cellStyleD, bold: true, color: "2E7D32" } },
    ],
];

const cw = tableD_w / 7;
slide.addTable(tableD_rows, {
    x: tableD_x, y: y + 0.35, w: tableD_w,
    rowH: [0.3, cellH, cellH, cellH, cellH],
    border: { type: "solid", pt: 0.5, color: "CCCCCC" },
    colW: [cw * 1.2, cw, cw, cw * 0.7, cw, cw, cw * 0.7],
});

// ===== Save =====
const outputPath = path.join(OUT, "Figure_Complete_v3.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
    console.log("Saved:", outputPath);
}).catch(err => console.error(err));
