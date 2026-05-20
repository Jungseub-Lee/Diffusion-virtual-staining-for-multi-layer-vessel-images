const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = path.resolve("analysis_output");
let pres = new pptxgen();

// A4 portrait
pres.defineLayout({ name: "A4_PORTRAIT", width: 7.5, height: 10.0 });
pres.layout = "A4_PORTRAIT";

let slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// Image: 8.27 x 11.69 aspect → ratio ~0.707
// Fit to slide width with margins
const imgW = 7.0;
const imgH = imgW * (11.69 / 8.27);  // ~9.89, but image is cropped by bbox
// Actual saved image ratio from matplotlib (10x14 figure, but bbox_inches='tight' crops it)
// Let's use a reasonable ratio
const actualH = imgW * 1.35;  // approximate from the rendered image

slide.addImage({
    path: path.join(OUT, "dataset_figure_final.png"),
    x: (7.5 - imgW) / 2,
    y: 0.15,
    w: imgW,
    h: actualH
});

// Caption at bottom
slide.addText(
    "Multi-color depth-encoded fluorescence dataset. " +
    "(a) BF to depth-encoded GT showing vessel network with color-coded depth information. " +
    "(b) 8 individual z-layers with distinct color assignments spanning 0-80 \u00B5m MPS height. " +
    "L1-L3: Top layer (Blue, 0-30 \u00B5m), L4-L5: Middle layer (Yellow, 30-50 \u00B5m), L6-L8: Bottom layer (Red, 50-80 \u00B5m). " +
    "(c) Layer group merging to multi-color maximum intensity projection. " +
    "(d) Hue distribution confirming separable depth-color encoding, and per-layer fluorescence intensity profile. " +
    "(e) Mean depth maps showing spatial depth distribution across two representative samples.",
    {
        x: 0.3,
        y: actualH + 0.3,
        w: 6.9,
        h: 0.7,
        fontSize: 6.5,
        fontFace: "Arial",
        italic: true,
        color: "777777",
        valign: "top",
        margin: 0
    }
);

const outputPath = path.join(OUT, "Figure_Dataset_DepthEncoding.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
    console.log("Saved:", outputPath);
}).catch(err => console.error(err));
