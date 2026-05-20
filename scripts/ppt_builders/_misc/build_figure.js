const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = "analysis_output";
let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Depth-aware Analysis";
pres.title = "Figure - Depth-aware Morphometric Analysis";

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// Panel labels
const lbl = { fontSize: 18, fontFace: "Arial", bold: true, color: "000000", margin: 0 };
slide.addText("a", { x: 0.15, y: 0.1, w: 0.3, h: 0.3, ...lbl });
slide.addText("b", { x: 4.5, y: 0.1, w: 0.3, h: 0.3, ...lbl });
slide.addText("c", { x: 0.15, y: 3.85, w: 0.3, h: 0.3, ...lbl });
slide.addText("d", { x: 0.15, y: 5.6, w: 0.3, h: 0.3, ...lbl });

// Panel (a): Schematic placeholder
slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.4, w: 4.0, h: 3.3,
    fill: { color: "F8F9FA" },
    line: { color: "DEE2E6", width: 1 }
});
slide.addText("Schematic: Single-color vs Multi-color\nDepth-separated Skeleton Analysis\n\n(Insert refined schematic illustration)", {
    x: 0.4, y: 0.8, w: 3.8, h: 2.5,
    fontSize: 11, fontFace: "Arial", color: "555555", align: "center", valign: "middle", margin: 0
});

// Legend
slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.15, w: 0.3, h: 0.12, fill: { color: "CC6666" } });
slide.addText("Bottom (Red)", { x: 0.95, y: 3.1, w: 1.0, h: 0.2, fontSize: 7, fontFace: "Arial", color: "666666", margin: 0 });
slide.addShape(pres.shapes.RECTANGLE, { x: 2.0, y: 3.15, w: 0.3, h: 0.12, fill: { color: "DDCC44" } });
slide.addText("Middle (Yellow)", { x: 2.35, y: 3.1, w: 1.0, h: 0.2, fontSize: 7, fontFace: "Arial", color: "666666", margin: 0 });
slide.addShape(pres.shapes.RECTANGLE, { x: 3.3, y: 3.15, w: 0.3, h: 0.12, fill: { color: "6699CC" } });
slide.addText("Top (Blue)", { x: 3.65, y: 3.1, w: 0.8, h: 0.2, fontSize: 7, fontFace: "Arial", color: "666666", margin: 0 });

// Panel (b): Representative images (2 rows x 4 cols)
const imgW = 2.0, imgH = 1.35;
const imgLabels = ["Brightfield", "Multi-color FL (GT)", "Single Skeleton", "Depth-sep Skeleton"];
const imgStartX = 4.7;
const sampleIds = ["16-18-716", "1-16-512"];
const imgSuffixes = ["bf", "fl", "single", "multi"];

for (let row = 0; row < 2; row++) {
    const imgY = 0.5 + row * (imgH + 0.35);
    const sid = sampleIds[row];
    for (let i = 0; i < 4; i++) {
        const x = imgStartX + i * (imgW + 0.12);
        const imgPath = path.resolve(OUT, `panel_b_${sid}_${imgSuffixes[i]}.png`);

        if (fs.existsSync(imgPath)) {
            slide.addImage({
                path: imgPath,
                x: x, y: imgY, w: imgW, h: imgH,
                sizing: { type: "cover", w: imgW, h: imgH }
            });
        } else {
            slide.addShape(pres.shapes.RECTANGLE, {
                x: x, y: imgY, w: imgW, h: imgH,
                fill: { color: "1A1A1A" },
                line: { color: "666666", width: 0.5 }
            });
        }
        slide.addText(row === 0 ? imgLabels[i] : "", {
            x: x, y: imgY - 0.18, w: imgW, h: 0.18,
            fontSize: 7, fontFace: "Arial", color: "555555", align: "center", margin: 0
        });
    }
}

// Panel (c): Charts
const panelCPath = path.resolve(OUT, "panel_c.png");
if (fs.existsSync(panelCPath)) {
    slide.addImage({ path: panelCPath, x: 0.3, y: 4.05, w: 6.5, h: 1.45 });
}
slide.addText("Single-color GT vs Multi-color GT (N=215, paired t-test)", {
    x: 0.5, y: 3.9, w: 5.5, h: 0.18,
    fontSize: 8, fontFace: "Arial", italic: true, color: "666666", margin: 0
});

// Panel (d): Charts
const panelDPath = path.resolve(OUT, "panel_d.png");
if (fs.existsSync(panelDPath)) {
    slide.addImage({ path: panelDPath, x: 0.3, y: 5.8, w: 6.5, h: 1.45 });
}
slide.addText("Model fidelity: Multi-color Model / Multi-color GT (dashed line = 1.0)", {
    x: 0.5, y: 5.65, w: 6.0, h: 0.18,
    fontSize: 8, fontFace: "Arial", italic: true, color: "666666", margin: 0
});

// Summary table
slide.addText("Quantitative Summary", {
    x: 7.2, y: 3.9, w: 3.0, h: 0.25,
    fontSize: 11, fontFace: "Arial", bold: true, color: "333333", margin: 0
});

const tbl = [
    [
        { text: "", options: { fill: { color: "E0E0E0" }, fontSize: 7, bold: true } },
        { text: "S vs M\nRatio", options: { fill: { color: "E0E0E0" }, fontSize: 7, bold: true, align: "center" } },
        { text: "LBBDM\nRatio", options: { fill: { color: "E0E0E0" }, fontSize: 7, bold: true, align: "center" } },
        { text: "R\u00B2", options: { fill: { color: "E0E0E0" }, fontSize: 7, bold: true, align: "center" } },
        { text: "PBBDM\nRatio", options: { fill: { color: "E0E0E0" }, fontSize: 7, bold: true, align: "center" } },
        { text: "R\u00B2", options: { fill: { color: "E0E0E0" }, fontSize: 7, bold: true, align: "center" } },
    ],
    [
        { text: "Junction", options: { fontSize: 7 } },
        { text: "1.11**", options: { fontSize: 7, align: "center" } },
        { text: "0.973", options: { fontSize: 7, align: "center" } },
        { text: "0.787", options: { fontSize: 7, align: "center" } },
        { text: "0.874", options: { fontSize: 7, align: "center" } },
        { text: "0.857", options: { fontSize: 7, align: "center" } },
    ],
    [
        { text: "Endpoint", options: { fontSize: 7 } },
        { text: "0.93**", options: { fontSize: 7, align: "center" } },
        { text: "0.766", options: { fontSize: 7, align: "center", color: "CC0000" } },
        { text: "0.611", options: { fontSize: 7, align: "center", color: "CC0000" } },
        { text: "0.953", options: { fontSize: 7, align: "center" } },
        { text: "0.653", options: { fontSize: 7, align: "center" } },
    ],
    [
        { text: "Length", options: { fontSize: 7, bold: true } },
        { text: "1.15***", options: { fontSize: 7, bold: true, align: "center", color: "006600" } },
        { text: "1.120", options: { fontSize: 7, bold: true, align: "center", color: "006600" } },
        { text: "0.953", options: { fontSize: 7, bold: true, align: "center", color: "006600" } },
        { text: "1.084", options: { fontSize: 7, bold: true, align: "center", color: "006600" } },
        { text: "0.952", options: { fontSize: 7, bold: true, align: "center", color: "006600" } },
    ],
    [
        { text: "Area", options: { fontSize: 7 } },
        { text: "1.00 ns", options: { fontSize: 7, align: "center" } },
        { text: "1.131", options: { fontSize: 7, align: "center", color: "006600" } },
        { text: "0.974", options: { fontSize: 7, align: "center", color: "006600" } },
        { text: "1.097", options: { fontSize: 7, align: "center", color: "006600" } },
        { text: "0.974", options: { fontSize: 7, align: "center", color: "006600" } },
    ],
];

slide.addTable(tbl, {
    x: 7.2, y: 4.2, w: 5.8,
    border: { pt: 0.5, color: "CCCCCC" },
    colW: [0.7, 0.65, 0.65, 0.5, 0.65, 0.5],
    align: "center",
    valign: "middle",
});

slide.addText([
    { text: "Green", options: { fontSize: 7, bold: true, color: "006600" } },
    { text: " = ratio near 1.0 & R\u00B2>0.9   ", options: { fontSize: 7, color: "666666" } },
    { text: "Red", options: { fontSize: 7, bold: true, color: "CC0000" } },
    { text: " = significant deviation from GT", options: { fontSize: 7, color: "666666" } },
], {
    x: 7.2, y: 5.35, w: 5.5, h: 0.2,
    fontFace: "Arial", margin: 0
});

// Method note
slide.addText([
    { text: "Methods: ", options: { fontSize: 7, bold: true, color: "666666" } },
    { text: "Bottom 20% excluded (ECM baseline). Hole-filling for crossing continuity. Per-layer independent metrics. N=215.", options: { fontSize: 7, color: "999999" } },
], {
    x: 0.3, y: 7.15, w: 12.5, h: 0.25,
    fontFace: "Arial", margin: 0
});

const outputPath = path.resolve("analysis_output", "Figure_DepthAware_v2.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
    console.log("Saved:", outputPath);
}).catch(err => console.error(err));
