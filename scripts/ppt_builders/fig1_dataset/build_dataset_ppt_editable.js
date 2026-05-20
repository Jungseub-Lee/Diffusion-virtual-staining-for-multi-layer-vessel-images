const pptxgen = require("pptxgenjs");
const path = require("path");

const P = path.resolve("analysis_output/panels");
let pres = new pptxgen();

// A4 portrait
pres.defineLayout({ name: "A4P", width: 7.5, height: 10.0 });
pres.layout = "A4P";

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// ===== Layout constants =====
const margin = 0.3;
const usableW = 7.5 - 2 * margin;  // 6.9
const imgSize_a = 2.6;  // BF and GT
const layerSize = 0.78;  // each layer image
const groupSize = 1.4;   // merged group images
const chartW = 3.2;
const chartH = 2.2;
const depthW = 3.0;
const depthH = 2.2;

let y = 0.15;

// ===== Panel a: BF → GT =====
slide.addText("a", { x: margin - 0.05, y: y, w: 0.3, h: 0.3, fontSize: 14, bold: true, fontFace: "Arial", color: "000000", margin: 0 });

const bfX = margin + 0.2;
const gtX = 7.5 - margin - imgSize_a - 0.2;
const arrowX = bfX + imgSize_a + 0.1;
const arrowW = gtX - arrowX - 0.1;

slide.addImage({ path: path.join(P, "panel_a_bf.png"), x: bfX, y: y + 0.15, w: imgSize_a, h: imgSize_a });
slide.addText("Brightfield (BF)", { x: bfX, y: y, w: imgSize_a, h: 0.2, fontSize: 8, bold: true, fontFace: "Arial", align: "center", color: "333333", margin: 0 });

// Arrow
slide.addShape(pres.shapes.RIGHT_ARROW, {
    x: arrowX + arrowW * 0.2, y: y + 0.15 + imgSize_a / 2 - 0.12, w: arrowW * 0.6, h: 0.24,
    fill: { color: "333333" }
});

slide.addImage({ path: path.join(P, "panel_a_gt.png"), x: gtX, y: y + 0.15, w: imgSize_a, h: imgSize_a });
slide.addText("Depth-encoded Fluorescence (GT)", { x: gtX, y: y, w: imgSize_a, h: 0.2, fontSize: 8, bold: true, fontFace: "Arial", align: "center", color: "333333", margin: 0 });

y += imgSize_a + 0.45;

// ===== Panel b: 8 layers =====
slide.addText("b", { x: margin - 0.05, y: y, w: 0.3, h: 0.3, fontSize: 14, bold: true, fontFace: "Arial", color: "000000", margin: 0 });

const layerGap = 0.02;
const totalLayerW = 8 * layerSize + 7 * layerGap;
const layerStartX = (7.5 - totalLayerW) / 2;

// Group bracket colors and positions
const groupDefs = [
    { start: 0, end: 2, color: "4488FF", label: "Top (Blue)\nL1-L3" },
    { start: 3, end: 4, color: "DDAA00", label: "Middle (Yellow)\nL4-L5" },
    { start: 5, end: 7, color: "FF4444", label: "Bottom (Red)\nL6-L8" },
];

for (let i = 0; i < 8; i++) {
    const lx = layerStartX + i * (layerSize + layerGap);
    slide.addImage({ path: path.join(P, `panel_b_L${i + 1}.png`), x: lx, y: y + 0.15, w: layerSize, h: layerSize });
    slide.addText(`L${i + 1}`, { x: lx, y: y, w: layerSize, h: 0.15, fontSize: 7, bold: true, fontFace: "Arial", align: "center", color: "333333", margin: 0 });

    // Color border (top/bottom lines for group)
    let groupColor = "CCCCCC";
    for (const gd of groupDefs) {
        if (i >= gd.start && i <= gd.end) { groupColor = gd.color; break; }
    }
    // Top border
    slide.addShape(pres.shapes.LINE, { x: lx, y: y + 0.15, w: layerSize, h: 0, line: { color: groupColor, width: 2.5 } });
    // Bottom border
    slide.addShape(pres.shapes.LINE, { x: lx, y: y + 0.15 + layerSize, w: layerSize, h: 0, line: { color: groupColor, width: 2.5 } });
}

// Group labels below
for (const gd of groupDefs) {
    const gx = layerStartX + gd.start * (layerSize + layerGap);
    const gw = (gd.end - gd.start + 1) * layerSize + (gd.end - gd.start) * layerGap;
    slide.addText(gd.label, {
        x: gx, y: y + 0.15 + layerSize + 0.05, w: gw, h: 0.3,
        fontSize: 6.5, bold: true, fontFace: "Arial", align: "center", color: gd.color, margin: 0
    });
    // Bracket line
    slide.addShape(pres.shapes.LINE, { x: gx, y: y + 0.15 + layerSize + 0.03, w: gw, h: 0, line: { color: gd.color, width: 1.5 } });
}

y += layerSize + 0.7;

// ===== Panel c: Merged groups → Projection =====
slide.addText("c", { x: margin - 0.05, y: y, w: 0.3, h: 0.3, fontSize: 14, bold: true, fontFace: "Arial", color: "000000", margin: 0 });

const groupImages = ["panel_c_top.png", "panel_c_middle.png", "panel_c_bottom.png"];
const groupLabels = ["Top (Blue)\n0-30 \u00B5m", "Middle (Yellow)\n30-50 \u00B5m", "Bottom (Red)\n50-80 \u00B5m"];
const groupLabelColors = ["4488FF", "DDAA00", "FF4444"];
const gGap = 0.15;
const projSize = groupSize;
const totalGroupW = 3 * groupSize + 2 * gGap + 0.4 + projSize;
const groupStartX = (7.5 - totalGroupW) / 2;

for (let i = 0; i < 3; i++) {
    const gx = groupStartX + i * (groupSize + gGap);
    slide.addImage({ path: path.join(P, groupImages[i]), x: gx, y: y + 0.25, w: groupSize, h: groupSize });
    slide.addText(groupLabels[i], { x: gx, y: y, w: groupSize, h: 0.25, fontSize: 6.5, bold: true, fontFace: "Arial", align: "center", color: groupLabelColors[i], margin: 0 });
}

// Arrow to projection
const arrX2 = groupStartX + 3 * (groupSize + gGap) - gGap + 0.05;
slide.addShape(pres.shapes.RIGHT_ARROW, {
    x: arrX2, y: y + 0.25 + groupSize / 2 - 0.1, w: 0.35, h: 0.2,
    fill: { color: "333333" }
});

const projX = arrX2 + 0.4;
slide.addImage({ path: path.join(P, "panel_a_gt.png"), x: projX, y: y + 0.25, w: projSize, h: projSize });
slide.addText("Multi-color\nProjection", { x: projX, y: y, w: projSize, h: 0.25, fontSize: 6.5, bold: true, fontFace: "Arial", align: "center", color: "333333", margin: 0 });

y += groupSize + 0.6;

// ===== Panel d: Hue histogram + Intensity =====
slide.addText("d", { x: margin - 0.05, y: y, w: 0.3, h: 0.3, fontSize: 14, bold: true, fontFace: "Arial", color: "000000", margin: 0 });

slide.addImage({ path: path.join(P, "panel_d_hue.png"), x: margin + 0.1, y: y + 0.05, w: chartW, h: chartH });
slide.addImage({ path: path.join(P, "panel_d_intensity.png"), x: 7.5 - margin - chartW - 0.1, y: y + 0.05, w: chartW, h: chartH });

y += chartH + 0.25;

// ===== Panel e: Depth maps =====
slide.addText("e", { x: margin - 0.05, y: y, w: 0.3, h: 0.3, fontSize: 14, bold: true, fontFace: "Arial", color: "000000", margin: 0 });

slide.addImage({ path: path.join(P, "panel_e_depth_s1.png"), x: margin + 0.3, y: y + 0.05, w: depthW, h: depthH });
slide.addText("Sample 1", { x: margin + 0.3, y: y + depthH + 0.05, w: depthW, h: 0.2, fontSize: 7, fontFace: "Arial", align: "center", color: "666666", margin: 0 });

slide.addImage({ path: path.join(P, "panel_e_depth_s2.png"), x: 7.5 - margin - depthW - 0.3, y: y + 0.05, w: depthW, h: depthH });
slide.addText("Sample 2", { x: 7.5 - margin - depthW - 0.3, y: y + depthH + 0.05, w: depthW, h: 0.2, fontSize: 7, fontFace: "Arial", align: "center", color: "666666", margin: 0 });

// ===== Save =====
const outputPath = path.resolve("analysis_output", "Figure_Dataset_DepthEncoding_v2.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
    console.log("Saved:", outputPath);
}).catch(err => console.error(err));
