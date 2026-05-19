const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = "analysis_output";
let pres = new pptxgen();
pres.defineLayout({ name: "A4_PORTRAIT", width: 7.5, height: 10.0 });
pres.layout = "A4_PORTRAIT";

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// Image ratio: 1.826
const imgPath = path.resolve(OUT, "Figure_perlayer_concordance.png");
const imgW = 7.1, imgH = imgW / 1.826;

if (fs.existsSync(imgPath)) {
    slide.addImage({ path: imgPath, x: (7.5 - imgW) / 2, y: 0.3, w: imgW, h: imgH });
}

// Bottom note
slide.addText("Per-layer depth-aware concordance between GT and model-predicted multi-color virtual staining. " +
    "Bottom layer (Red, \u25CF) and Top layer (Blue, \u25A0) are independently compared. " +
    "Bottom 20% excluded (ECM baseline). Hole-filling applied. N=215 per layer.", {
    x: 0.4, y: imgH + 0.5, w: 6.7, h: 0.6,
    fontSize: 7.5, fontFace: "Arial", italic: true, color: "666666", margin: 0
});

const outputPath = path.resolve("analysis_output", "Figure_PerLayer_Concordance.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
    console.log("Saved:", outputPath);
}).catch(err => console.error(err));
