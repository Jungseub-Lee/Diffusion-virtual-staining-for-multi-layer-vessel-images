const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";

const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// ====== COLORS (for white background) ======
const RED = "C0392B";
const RED_LIGHT = "E74C3C";
const YELLOW = "D4A017";
const YELLOW_LIGHT = "F0C040";
const BLUE = "2980B9";
const BLUE_LIGHT = "3498DB";
const GREEN = "27AE60";
const GREEN_DARK = "1E8449";
const BLACK = "2C3E50";
const GRAY = "7F8C8D";
const GRAY_LIGHT = "BDC3C7";
const GRAY_DARK = "5D6D7E";
const PANEL_BG = "F7F9FA";
const PANEL_BORDER = "D5DBDB";
const ACCENT_RED = "E74C3C";
const ACCENT_GREEN = "27AE60";

const row1Y = 0.55, row2Y = 3.95, rowH = 3.0;
const vesselX = 0.3, vesselW = 5.2;
const arrowX = 5.65, arrowW = 0.6;
const skelX = 6.4, skelW = 5.2;
const annotX = 11.8, annotW = 1.35;

// ====== HELPERS ======
// Junction marker: filled circle with white center ring (target-like)
function junctionMark(s, cx, cy, color, r = 0.09) {
  // Outer filled circle
  s.addShape(pres.shapes.OVAL, {
    x: cx-r, y: cy-r, w: r*2, h: r*2,
    fill: { color }, line: { color, width: 1 },
  });
  // Inner white ring
  const ri = r * 0.45;
  s.addShape(pres.shapes.OVAL, {
    x: cx-ri, y: cy-ri, w: ri*2, h: ri*2,
    fill: { color: "FFFFFF" }, line: { color: "FFFFFF", width: 0.5 },
  });
}

// Endpoint marker: open triangle/arrowhead pointing outward (or small open circle)
function endpointMark(s, cx, cy, color, r = 0.065) {
  // Open circle with thick border
  s.addShape(pres.shapes.OVAL, {
    x: cx-r, y: cy-r, w: r*2, h: r*2,
    fill: { color: "FFFFFF" },
    line: { color, width: 2.5 },
  });
}

// False junction: circle with X
function falseJunctionMark(s, cx, cy, r = 0.16) {
  s.addShape(pres.shapes.OVAL, {
    x: cx-r, y: cy-r, w: r*2, h: r*2,
    fill: { color: "FDEDEC" },
    line: { color: ACCENT_RED, width: 2.5 },
  });
  // X lines
  const d = r * 0.55;
  s.addShape(pres.shapes.LINE, {
    x: cx-d, y: cy-d, w: d*2, h: 0,
    line: { color: ACCENT_RED, width: 2 },
    rotate: 45,
  });
  s.addShape(pres.shapes.LINE, {
    x: cx-d, y: cy+d, w: d*2, h: 0,
    line: { color: ACCENT_RED, width: 2 },
    rotate: -45,
  });
}

// Correct pass-through: circle with checkmark
function correctMark(s, cx, cy, r = 0.16) {
  s.addShape(pres.shapes.OVAL, {
    x: cx-r, y: cy-r, w: r*2, h: r*2,
    fill: { color: "EAFAF1" },
    line: { color: ACCENT_GREEN, width: 2.5 },
  });
  s.addText("✓", {
    x: cx-r, y: cy-r-0.02, w: r*2, h: r*2+0.04,
    fontSize: 11, fontFace: "Calibri", bold: true,
    color: ACCENT_GREEN, align: "center", valign: "middle", margin: 0,
  });
}

function taperedLine(s, pts, color, startW, endW, opacity = 100) {
  const n = pts.length - 1;
  for (let i = 0; i < n; i++) {
    const [x1,y1] = pts[i], [x2,y2] = pts[i+1];
    const dx=x2-x1, dy=y2-y1, len=Math.sqrt(dx*dx+dy*dy);
    if (len<0.001) continue;
    const angle = Math.atan2(dy,dx)*(180/Math.PI);
    const w = startW + (endW-startW)*(i/n);
    s.addShape(pres.shapes.LINE, {
      x:x1, y:y1, w:len, h:0,
      line:{color, width:w, transparency:100-opacity}, rotate:angle,
    });
  }
}

function skelLine(s, pts, color, w=3) {
  for (let i=0; i<pts.length-1; i++) {
    const [x1,y1]=pts[i], [x2,y2]=pts[i+1];
    const dx=x2-x1, dy=y2-y1, len=Math.sqrt(dx*dx+dy*dy);
    if (len<0.001) continue;
    const angle=Math.atan2(dy,dx)*(180/Math.PI);
    s.addShape(pres.shapes.LINE, {
      x:x1, y:y1, w:len, h:0, line:{color, width:w}, rotate:angle,
    });
  }
}

function bz(p0,p1,p2,p3,n=14) {
  const pts=[];
  for (let i=0;i<=n;i++) {
    const t=i/n, u=1-t;
    pts.push([u*u*u*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*p3[0], u*u*u*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*p3[1]]);
  }
  return pts;
}

function off(pts,ox,oy){return pts.map(p=>[p[0]+ox,p[1]+oy]);}

// ====== VESSEL PATHS ======
const A_stem = bz([2.0,2.7],[1.9,2.1],[1.7,1.4],[1.4,0.9],20).concat(bz([1.4,0.9],[1.2,0.55],[0.9,0.3],[0.6,0.18],12));
const A_br1 = bz([1.7,1.4],[2.1,1.1],[2.6,0.7],[3.1,0.45],14).concat(bz([3.1,0.45],[3.4,0.3],[3.7,0.2],[4.0,0.15],8));
const A_br2 = bz([1.4,0.9],[1.1,0.7],[0.8,0.6],[0.5,0.6],8);
const A_br3 = bz([2.6,0.7],[2.7,0.45],[2.8,0.25],[2.9,0.12],8);

const B_stem = bz([3.5,2.7],[3.3,2.1],[2.8,1.5],[2.3,1.1],18).concat(bz([2.3,1.1],[1.9,0.85],[1.55,0.65],[1.2,0.5],12));
const B_br1 = bz([2.8,1.5],[2.4,1.3],[1.8,1.2],[1.3,1.3],12).concat(bz([1.3,1.3],[1.0,1.35],[0.7,1.45],[0.45,1.55],8));
const B_br2 = bz([2.3,1.1],[2.7,0.8],[3.2,0.6],[3.7,0.5],12).concat(bz([3.7,0.5],[4.1,0.45],[4.4,0.4],[4.7,0.35],8));
const B_br3 = bz([3.3,2.1],[3.7,1.9],[4.1,1.7],[4.5,1.55],10);

const allA=[A_stem,A_br1,A_br2,A_br3], allB=[B_stem,B_br1,B_br2,B_br3];
const taperA=[[24,8],[14,5],[8,4],[8,4]], taperB=[[22,7],[12,4],[12,5],[14,5]];
const epA=[[0.6,0.18],[4.0,0.15],[0.5,0.6],[2.9,0.12]];
const epB=[[1.2,0.5],[0.45,1.55],[4.7,0.35],[4.5,1.55]];
const rootA=[2.0,2.7], rootB=[3.5,2.7];

// ====== HEADERS ======
slide.addText("a", {x:0.08,y:row1Y-0.02,w:0.22,h:0.3,fontSize:18,fontFace:"Arial",bold:true,color:BLACK,margin:0});
slide.addText("b", {x:0.08,y:row2Y-0.02,w:0.22,h:0.3,fontSize:18,fontFace:"Arial",bold:true,color:BLACK,margin:0});
slide.addText("Single-color Fluorescence",{x:vesselX,y:row1Y-0.4,w:vesselW,h:0.3,fontSize:12,fontFace:"Arial",bold:true,color:GREEN_DARK,align:"center",margin:0});
slide.addText("Multi-color Depth-encoded",{x:vesselX,y:row2Y-0.4,w:vesselW,h:0.3,fontSize:12,fontFace:"Arial",bold:true,color:BLUE,align:"center",margin:0});
slide.addText("Skeletonization",{x:skelX,y:row1Y-0.4,w:skelW,h:0.3,fontSize:12,fontFace:"Arial",bold:true,color:GRAY_DARK,align:"center",margin:0});
slide.addText("Depth-separated Skeletonization",{x:skelX,y:row2Y-0.4,w:skelW,h:0.3,fontSize:12,fontFace:"Arial",bold:true,color:GRAY_DARK,align:"center",margin:0});

// ====== PANELS ======
for (const [px,py,pw] of [[vesselX,row1Y,vesselW],[skelX,row1Y,skelW],[vesselX,row2Y,vesselW],[skelX,row2Y,skelW]]) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:px,y:py,w:pw,h:rowH,fill:{color:PANEL_BG},rectRadius:0.08,line:{color:PANEL_BORDER,width:1},
  });
}

// ====== ECM BASELINE ======
for (const px of [vesselX,skelX]) {
  for (const py of [row1Y,row2Y]) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x:px+0.08, y:py+2.63, w:vesselW-0.16, h:0.28,
      fill:{color:"ECF0F1"},
    });
    slide.addShape(pres.shapes.LINE, {
      x:px+0.08, y:py+2.63, w:vesselW-0.16, h:0,
      line:{color:GRAY_LIGHT, width:1.5},
    });
    slide.addText("ECM",{x:px+vesselW-0.7,y:py+2.67,w:0.55,h:0.18,fontSize:7,fontFace:"Arial",italic:true,color:GRAY,align:"right",margin:0});
  }
}

// ==================================
// ROW 1: SINGLE-COLOR
// ==================================
const v1ox=vesselX, v1oy=row1Y;
for (let i=0;i<allA.length;i++) taperedLine(slide,off(allA[i],v1ox,v1oy),GREEN,taperA[i][0],taperA[i][1],70);
for (let i=0;i<allB.length;i++) taperedLine(slide,off(allB[i],v1ox,v1oy),GREEN,taperB[i][0],taperB[i][1],65);

// Overlap zone
slide.addShape(pres.shapes.OVAL, {
  x:v1ox+1.4, y:v1oy+0.55, w:0.9, h:0.75,
  fill:{color:"F9E79F",transparency:70}, line:{color:GRAY,width:1,dashType:"dash"},
});
slide.addText("overlap",{x:v1ox+1.35,y:v1oy+1.32,w:1.0,h:0.16,fontSize:7,fontFace:"Arial",italic:true,color:GRAY,align:"center",margin:0});

// === SINGLE SKELETON ===
const s1ox=skelX, s1oy=row1Y;
const A_stem_top=bz([1.4,0.9],[1.2,0.55],[0.9,0.3],[0.6,0.18],12);
const A_stem_bot=bz([2.0,2.7],[1.9,2.1],[1.7,1.4],[1.5,1.15],14);
const B_stem_top=bz([1.6,0.7],[1.4,0.6],[1.3,0.55],[1.2,0.5],8);
const B_stem_bot=bz([3.5,2.7],[3.3,2.1],[2.8,1.5],[2.4,1.2],14);
const merged1=bz([2.4,1.2],[2.1,1.0],[1.8,0.9],[1.5,1.15],10);
const merged2=bz([1.8,0.9],[1.6,0.8],[1.5,0.75],[1.4,0.9],8);

skelLine(slide,off(A_stem_top,s1ox,s1oy),GREEN_DARK);
skelLine(slide,off(A_stem_bot,s1ox,s1oy),GREEN_DARK);
skelLine(slide,off(B_stem_top,s1ox,s1oy),GREEN_DARK);
skelLine(slide,off(B_stem_bot,s1ox,s1oy),GREEN_DARK);
skelLine(slide,off(merged1,s1ox,s1oy),GREEN_DARK,4);
skelLine(slide,off(merged2,s1ox,s1oy),GREEN_DARK,4);
for (const p of [A_br1,A_br2,A_br3,B_br1,B_br2,B_br3]) skelLine(slide,off(p,s1ox,s1oy),GREEN_DARK);

// FALSE JUNCTIONS
const fj1x=s1ox+1.5, fj1y=s1oy+1.15;
const fj2x=s1ox+1.8, fj2y=s1oy+0.9;
falseJunctionMark(slide, fj1x, fj1y);
falseJunctionMark(slide, fj2x, fj2y);

// Annotation with bracket line
slide.addText([
  {text:"False junctions\n",options:{bold:true,fontSize:8.5,color:ACCENT_RED}},
  {text:"Overlapping vessels at different\nz-depths → false intersections",options:{fontSize:7,color:"922B21"}},
],{x:s1ox+2.8,y:s1oy+0.35,w:1.85,h:0.6,margin:0});
skelLine(slide,[[s1ox+2.8,s1oy+0.72],[fj2x+0.16,fj2y]],ACCENT_RED,1);

// Length annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{
  x:s1ox+0.85,y:s1oy+1.48,w:1.65,h:0.35,
  fill:{color:"FEF9E7"}, rectRadius:0.05, line:{color:YELLOW,width:1,dashType:"dash"},
});
slide.addText("Merged → 1 skeleton, length lost",{
  x:s1ox+0.85,y:s1oy+1.48,w:1.65,h:0.35,
  fontSize:6.5,fontFace:"Arial",bold:true,italic:true,color:"7D6608",align:"center",valign:"middle",margin:0,
});
skelLine(slide,[[s1ox+1.68,s1oy+1.48],[s1ox+1.65,s1oy+1.1]],YELLOW,1);

// Real junctions
junctionMark(slide,s1ox+1.7,s1oy+1.4,GREEN_DARK);
junctionMark(slide,s1ox+1.4,s1oy+0.9,GREEN_DARK);
junctionMark(slide,s1ox+2.6,s1oy+0.7,GREEN_DARK);
junctionMark(slide,s1ox+2.8,s1oy+1.5,GREEN_DARK);
junctionMark(slide,s1ox+2.3,s1oy+1.1,GREEN_DARK);
junctionMark(slide,s1ox+3.3,s1oy+2.1,GREEN_DARK);

// Endpoints
for (const ep of [...epA,...epB]) endpointMark(slide,s1ox+ep[0],s1oy+ep[1],GREEN_DARK);
endpointMark(slide,s1ox+rootA[0],s1oy+rootA[1],GREEN_DARK);
endpointMark(slide,s1ox+rootB[0],s1oy+rootB[1],GREEN_DARK);


// ==================================
// ROW 2: MULTI-COLOR
// ==================================
const v2ox=vesselX, v2oy=row2Y;
for (let i=0;i<allA.length;i++) taperedLine(slide,off(allA[i],v2ox,v2oy),BLUE_LIGHT,taperA[i][0],taperA[i][1],70);
for (let i=0;i<allB.length;i++) taperedLine(slide,off(allB[i],v2ox,v2oy),RED_LIGHT,taperB[i][0],taperB[i][1],65);

// Yellow transition
const yw=bz([2.1,1.05],[1.95,0.95],[1.85,0.88],[1.75,0.85],6);
taperedLine(slide,off(yw,v2ox,v2oy),YELLOW_LIGHT,10,6,55);

slide.addShape(pres.shapes.OVAL,{
  x:v2ox+1.4,y:v2oy+0.55,w:0.9,h:0.75,
  fill:{color:"FDEBD0",transparency:60},line:{color:YELLOW,width:1,dashType:"dash"},
});
slide.addText("different\nz-depths",{x:v2ox+1.35,y:v2oy+1.32,w:1.0,h:0.22,fontSize:7,fontFace:"Arial",italic:true,color:YELLOW,align:"center",margin:0});

// Color legend
const cly=v2oy+2.42;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:v2ox+0.6,y:cly,w:3.8,h:0.3,fill:{color:"FFFFFF",transparency:20},rectRadius:0.05,line:{color:PANEL_BORDER,width:0.5}});
skelLine(slide,[[v2ox+0.8,cly+0.15],[v2ox+1.1,cly+0.15]],RED,5);
slide.addText("Bottom",{x:v2ox+1.15,y:cly+0.03,w:0.55,h:0.24,fontSize:8,fontFace:"Arial",color:RED,margin:0,valign:"middle"});
skelLine(slide,[[v2ox+1.75,cly+0.15],[v2ox+2.05,cly+0.15]],YELLOW,5);
slide.addText("Middle",{x:v2ox+2.1,y:cly+0.03,w:0.55,h:0.24,fontSize:8,fontFace:"Arial",color:YELLOW,margin:0,valign:"middle"});
skelLine(slide,[[v2ox+2.7,cly+0.15],[v2ox+3.0,cly+0.15]],BLUE,5);
slide.addText("Top",{x:v2ox+3.05,y:cly+0.03,w:0.4,h:0.24,fontSize:8,fontFace:"Arial",color:BLUE,margin:0,valign:"middle"});
slide.addText("← Deep    Shallow →",{x:v2ox+3.4,y:cly+0.03,w:0.9,h:0.24,fontSize:6.5,fontFace:"Arial",color:GRAY,margin:0,valign:"middle"});

// === MULTI SKELETON ===
const s2ox=skelX, s2oy=row2Y;
for (const p of allA) skelLine(slide,off(p,s2ox,s2oy),BLUE);
for (const p of allB) skelLine(slide,off(p,s2ox,s2oy),RED);
skelLine(slide,off(yw,s2ox,s2oy),YELLOW,2.5);

// CORRECT marks
const nf1x=s2ox+1.5,nf1y=s2oy+1.15;
const nf2x=s2ox+1.8,nf2y=s2oy+0.9;
correctMark(slide,nf1x,nf1y);
correctMark(slide,nf2x,nf2y);

slide.addText([
  {text:"No false junctions\n",options:{bold:true,fontSize:8.5,color:ACCENT_GREEN}},
  {text:"Different layers pass through\nindependently",options:{fontSize:7,color:"1A5276"}},
],{x:s2ox+2.8,y:s2oy+0.35,w:1.85,h:0.5,margin:0});
skelLine(slide,[[s2ox+2.8,s2oy+0.65],[nf2x+0.16,nf2y]],ACCENT_GREEN,1);

// Length annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{
  x:s2ox+0.85,y:s2oy+1.48,w:1.65,h:0.35,
  fill:{color:"EAFAF1"}, rectRadius:0.05, line:{color:ACCENT_GREEN,width:1,dashType:"dash"},
});
slide.addText("2 skeletons → length recovered",{
  x:s2ox+0.85,y:s2oy+1.48,w:1.65,h:0.35,
  fontSize:6.5,fontFace:"Arial",bold:true,italic:true,color:"196F3D",align:"center",valign:"middle",margin:0,
});
skelLine(slide,[[s2ox+1.68,s2oy+1.48],[s2ox+1.65,s2oy+1.1]],ACCENT_GREEN,1);

// Blue junctions
junctionMark(slide,s2ox+1.7,s2oy+1.4,BLUE);
junctionMark(slide,s2ox+1.4,s2oy+0.9,BLUE);
junctionMark(slide,s2ox+2.6,s2oy+0.7,BLUE);
// Red junctions
junctionMark(slide,s2ox+2.8,s2oy+1.5,RED);
junctionMark(slide,s2ox+2.3,s2oy+1.1,RED);
junctionMark(slide,s2ox+3.3,s2oy+2.1,RED);

// Endpoints color-matched
for (const ep of epA) endpointMark(slide,s2ox+ep[0],s2oy+ep[1],BLUE);
for (const ep of epB) endpointMark(slide,s2ox+ep[0],s2oy+ep[1],RED);
endpointMark(slide,s2ox+rootA[0],s2oy+rootA[1],BLUE);
endpointMark(slide,s2ox+rootB[0],s2oy+rootB[1],RED);

// ====== ARROWS ======
for (const ry of [row1Y,row2Y]) {
  slide.addShape(pres.shapes.LINE, {
    x:arrowX, y:ry+rowH/2, w:arrowW, h:0,
    line:{color:GRAY_DARK, width:2, endArrowType:"triangle"},
  });
}

// ====== RIGHT ANNOTATIONS ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:annotX,y:row1Y+0.15,w:annotW,h:2.7,fill:{color:"FDEDEC"},rectRadius:0.08,line:{color:"F5B7B1",width:1}});
slide.addText("Limitations",{x:annotX,y:row1Y+0.2,w:annotW,h:0.22,fontSize:9.5,fontFace:"Arial",bold:true,color:ACCENT_RED,align:"center",margin:0});
slide.addText([
  {text:"Junctions\n",options:{bold:true,fontSize:8.5,color:BLACK}},
  {text:"Over-counted\n(false positives)\n\n",options:{fontSize:7,color:"922B21"}},
  {text:"Length\n",options:{bold:true,fontSize:8.5,color:BLACK}},
  {text:"Under-estimated\n(merged vessels)\n\n",options:{fontSize:7,color:"922B21"}},
  {text:"Endpoints\n",options:{bold:true,fontSize:8.5,color:BLACK}},
  {text:"Preserved\n\n",options:{fontSize:7,color:"922B21"}},
  {text:"Depth\n",options:{bold:true,fontSize:8.5,color:BLACK}},
  {text:"Lost",options:{fontSize:7,color:"922B21"}},
],{x:annotX+0.08,y:row1Y+0.5,w:annotW-0.16,h:2.3,margin:0,paraSpaceAfter:1});

slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:annotX,y:row2Y+0.15,w:annotW,h:2.7,fill:{color:"EAFAF1"},rectRadius:0.08,line:{color:"A9DFBF",width:1}});
slide.addText("Advantages",{x:annotX,y:row2Y+0.2,w:annotW,h:0.22,fontSize:9.5,fontFace:"Arial",bold:true,color:ACCENT_GREEN,align:"center",margin:0});
slide.addText([
  {text:"Junctions\n",options:{bold:true,fontSize:8.5,color:BLACK}},
  {text:"Accurate\n(per-layer)\n\n",options:{fontSize:7,color:"196F3D"}},
  {text:"Length\n",options:{bold:true,fontSize:8.5,color:BLACK}},
  {text:"Recovered\n(separated)\n\n",options:{fontSize:7,color:"196F3D"}},
  {text:"Endpoints\n",options:{bold:true,fontSize:8.5,color:BLACK}},
  {text:"Preserved\n\n",options:{fontSize:7,color:"196F3D"}},
  {text:"Depth\n",options:{bold:true,fontSize:8.5,color:BLACK}},
  {text:"Preserved",options:{fontSize:7,color:"196F3D"}},
],{x:annotX+0.08,y:row2Y+0.5,w:annotW-0.16,h:2.3,margin:0,paraSpaceAfter:1});

// ====== DIVIDER ======
slide.addShape(pres.shapes.LINE,{x:0.3,y:(row1Y+rowH+row2Y-0.4)/2,w:12.7,h:0,line:{color:GRAY_LIGHT,width:0.7,dashType:"lgDash"}});

// ====== BOTTOM LEGEND ======
const legY=7.15;
junctionMark(slide,skelX+0.5,legY,GRAY_DARK,0.07);
slide.addText("Junction",{x:skelX+0.62,y:legY-0.08,w:0.55,h:0.16,fontSize:7.5,fontFace:"Arial",color:GRAY_DARK,margin:0});
endpointMark(slide,skelX+1.35,legY,GRAY_DARK,0.055);
slide.addText("Endpoint",{x:skelX+1.45,y:legY-0.08,w:0.55,h:0.16,fontSize:7.5,fontFace:"Arial",color:GRAY_DARK,margin:0});
skelLine(slide,[[skelX+2.15,legY],[skelX+2.45,legY]],GRAY_DARK,3);
slide.addText("Skeleton",{x:skelX+2.5,y:legY-0.08,w:0.55,h:0.16,fontSize:7.5,fontFace:"Arial",color:GRAY_DARK,margin:0});
falseJunctionMark(slide,skelX+3.25,legY,0.08);
slide.addText("False junction",{x:skelX+3.38,y:legY-0.08,w:0.85,h:0.16,fontSize:7.5,fontFace:"Arial",color:ACCENT_RED,margin:0});
correctMark(slide,skelX+4.4,legY,0.08);
slide.addText("Correctly separated",{x:skelX+4.53,y:legY-0.08,w:1.1,h:0.16,fontSize:7.5,fontFace:"Arial",color:ACCENT_GREEN,margin:0});

pres.writeFile({fileName:"depth_schematic_v8.pptx"})
  .then(()=>console.log("Created: depth_schematic_v8.pptx"))
  .catch(err=>console.error(err));
