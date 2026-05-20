const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = "analysis_output";
let pres = new pptxgen();

// A4 portrait: 7.5" x 10" (approximate A4 in inches)
pres.defineLayout({ name: "A4_PORTRAIT", width: 7.5, height: 10.0 });
pres.layout = "A4_PORTRAIT";

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// Panel labels style
const lbl = { fontSize: 16, fontFace: "Arial", bold: true, color: "000000", margin: 0 };

// ============================================
// Panel (a): Schematic - top section
// ============================================
slide.addText("a", { x: 0.2, y: 0.15, w: 0.25, h: 0.25, ...lbl });
slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.4, w: 6.9, h: 2.4,
    fill: { color: "F8F9FA" },
    line: { color: "DEE2E6", width: 1 }
});
slide.addText("Insert Schematic: Single-color vs Multi-color Depth Separation\n(Replace with refined illustration)", {
    x: 0.5, y: 0.8, w: 6.5, h: 1.5,
    fontSize: 12, fontFace: "Arial", color: "888888", align: "center", valign: "middle", margin: 0
});
// Color legend
const legY = 2.5;
slide.addShape(pres.shapes.RECTANGLE, { x: 1.0, y: legY, w: 0.25, h: 0.1, fill: { color: "CC6666" } });
slide.addText("Bottom (Red)", { x: 1.3, y: legY-0.03, w: 1.0, h: 0.18, fontSize: 7, fontFace: "Arial", color: "666666", margin: 0 });
slide.addShape(pres.shapes.RECTANGLE, { x: 2.6, y: legY, w: 0.25, h: 0.1, fill: { color: "DDCC44" } });
slide.addText("Middle (Yellow)", { x: 2.9, y: legY-0.03, w: 1.0, h: 0.18, fontSize: 7, fontFace: "Arial", color: "666666", margin: 0 });
slide.addShape(pres.shapes.RECTANGLE, { x: 4.2, y: legY, w: 0.25, h: 0.1, fill: { color: "6699CC" } });
slide.addText("Top (Blue)", { x: 4.5, y: legY-0.03, w: 0.8, h: 0.18, fontSize: 7, fontFace: "Arial", color: "666666", margin: 0 });

// ============================================
// Panel (b): Representative images
// ============================================
slide.addText("b", { x: 0.2, y: 2.85, w: 0.25, h: 0.25, ...lbl });

const sid = "18-19-512";
const imgSuffixes = ["bf", "fl", "single", "multi"];
const imgLabels = ["Brightfield", "Multi-color FL (GT)", "Single Skeleton on BF", "Depth-sep Skeleton on BF"];
const imgW = 1.65;
const imgH = 1.65;
const imgStartX = 0.3;
const imgY = 3.25;

for (let i = 0; i < 4; i++) {
    const x = imgStartX + i * (imgW + 0.1);
    const imgPath = path.resolve(OUT, `panel_b_${sid}_${imgSuffixes[i]}.png`);

    if (fs.existsSync(imgPath)) {
        slide.addImage({
            path: imgPath, x: x, y: imgY, w: imgW, h: imgH,
            sizing: { type: "cover", w: imgW, h: imgH }
        });
    }
    slide.addText(imgLabels[i], {
        x: x, y: imgY - 0.18, w: imgW, h: 0.18,
        fontSize: 7, fontFace: "Arial", color: "333333", align: "center", margin: 0
    });
}

// ============================================
// Panel (c): Normalized Single/Multi GT chart
// ============================================
slide.addText("c", { x: 0.2, y: 5.0, w: 0.25, h: 0.25, ...lbl });
slide.addText("Single-color GT vs Multi-color GT (normalized, N=215)", {
    x: 0.5, y: 5.05, w: 5.5, h: 0.18,
    fontSize: 8, fontFace: "Arial", italic: true, color: "666666", margin: 0
});

const panelCPath = path.resolve(OUT, "panel_c_norm.png");
if (fs.existsSync(panelCPath)) {
    slide.addImage({ path: panelCPath, x: 0.3, y: 5.25, w: 6.9, h: 2.1 });
}

// ============================================
// Panel (d): Model/GT ratio chart
// ============================================
slide.addText("d", { x: 0.2, y: 7.4, w: 0.25, h: 0.25, ...lbl });
slide.addText("Model fidelity: Model Multi / GT Multi (dashed = 1.0)", {
    x: 0.5, y: 7.45, w: 5.5, h: 0.18,
    fontSize: 8, fontFace: "Arial", italic: true, color: "666666", margin: 0
});

const panelDPath = path.resolve(OUT, "panel_d_norm.png");
if (fs.existsSync(panelDPath)) {
    slide.addImage({ path: panelDPath, x: 0.3, y: 7.65, w: 6.9, h: 2.1 });
}

// Bottom note
slide.addText("Bottom 20% excluded (ECM baseline). Hole-filling for crossing vessel continuity. Per-layer independent metrics. N=215 paired samples.", {
    x: 0.3, y: 9.7, w: 6.9, h: 0.2,
    fontSize: 6.5, fontFace: "Arial", italic: true, color: "999999", margin: 0
});

const outputPath = path.resolve("analysis_output", "Figure_A4_portrait.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
    console.log("Saved:", outputPath);
}).catch(err => console.error(err));
