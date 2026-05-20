const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";

const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// ====== COLORS ======
const RED = "C0392B";
const RED_L = "E74C3C";
const YEL = "D4A017";
const YEL_L = "F0C040";
const BLUE = "2471A3";
const BLUE_L = "3498DB";
const GRN = "1E8449";
const GRN_L = "27AE60";
const BLK = "2C3E50";
const GRY = "7F8C8D";
const GRY_L = "BDC3C7";
const GRY_D = "5D6D7E";
const PNL = "F8F9F9";
const PNL_B = "D5DBDB";

const row1Y = 0.55, row2Y = 3.95, rowH = 3.0;
const vX = 0.3, vW = 5.2;
const aX = 5.65, aW = 0.6;
const sX = 6.4, sW = 5.2;
const anX = 11.8, anW = 1.35;

// ====== MARKERS ======
// Junction: solid filled circle
function junc(s, cx, cy, col, r = 0.08) {
  s.addShape(pres.shapes.OVAL, {
    x: cx-r, y: cy-r, w: r*2, h: r*2,
    fill: { color: col }, line: { color: "FFFFFF", width: 1.5 },
  });
}

// Endpoint: open diamond shape using rotated square
function endpt(s, cx, cy, col, r = 0.065) {
  s.addShape(pres.shapes.OVAL, {
    x: cx-r, y: cy-r, w: r*2, h: r*2,
    fill: { color: "FFFFFF" }, line: { color: col, width: 2 },
  });
  // Inner dot
  const ri = r * 0.3;
  s.addShape(pres.shapes.OVAL, {
    x: cx-ri, y: cy-ri, w: ri*2, h: ri*2,
    fill: { color: col },
  });
}

// False junction: red dashed circle
function falseJ(s, cx, cy, r = 0.18) {
  s.addShape(pres.shapes.OVAL, {
    x: cx-r, y: cy-r, w: r*2, h: r*2,
    fill: { color: "FDEDEC", transparency: 30 },
    line: { color: RED, width: 2, dashType: "dash" },
  });
}

// Correct pass: green dashed circle
function correctJ(s, cx, cy, r = 0.18) {
  s.addShape(pres.shapes.OVAL, {
    x: cx-r, y: cy-r, w: r*2, h: r*2,
    fill: { color: "EAFAF1", transparency: 30 },
    line: { color: GRN, width: 2, dashType: "dash" },
  });
}

function tapered(s, pts, col, sw, ew, op = 100) {
  const n = pts.length - 1;
  for (let i = 0; i < n; i++) {
    const [x1,y1]=pts[i],[x2,y2]=pts[i+1];
    const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy);
    if(len<0.001)continue;
    const ang=Math.atan2(dy,dx)*(180/Math.PI);
    const w=sw+(ew-sw)*(i/n);
    s.addShape(pres.shapes.LINE,{x:x1,y:y1,w:len,h:0,line:{color:col,width:w,transparency:100-op},rotate:ang});
  }
}

function skel(s, pts, col, w=3) {
  for(let i=0;i<pts.length-1;i++){
    const [x1,y1]=pts[i],[x2,y2]=pts[i+1];
    const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy);
    if(len<0.001)continue;
    const ang=Math.atan2(dy,dx)*(180/Math.PI);
    s.addShape(pres.shapes.LINE,{x:x1,y:y1,w:len,h:0,line:{color:col,width:w},rotate:ang});
  }
}

function bz(p0,p1,p2,p3,n=14){
  const r=[];
  for(let i=0;i<=n;i++){const t=i/n,u=1-t;r.push([u*u*u*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*p3[0],u*u*u*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*p3[1]]);}
  return r;
}
function O(p,ox,oy){return p.map(v=>[v[0]+ox,v[1]+oy]);}

// ====== VESSEL NETWORK ======
// Sprouting angiogenesis: ECM at bottom (y=2.7), vessels grow upward

// GROUP A (Top layer → Blue): main stem rises from center-left
const A0 = bz([2.0,2.7],[1.95,2.3],[1.8,1.7],[1.65,1.35],20)
  .concat(bz([1.65,1.35],[1.5,1.0],[1.3,0.65],[1.0,0.4],14))
  .concat(bz([1.0,0.4],[0.8,0.28],[0.6,0.2],[0.45,0.18],8));
// A branch 1: from (1.65,1.35) sweeping right
const A1 = bz([1.65,1.35],[2.0,1.1],[2.5,0.75],[3.0,0.5],14)
  .concat(bz([3.0,0.5],[3.3,0.38],[3.6,0.28],[3.9,0.22],8));
// A branch 2: small left sprout from (1.0,0.4)
const A2 = bz([1.0,0.4],[0.8,0.55],[0.6,0.65],[0.4,0.7],8);
// A branch 3: tip from A1 at (2.5,0.75)
const A3 = bz([2.5,0.75],[2.55,0.5],[2.6,0.3],[2.65,0.15],8);

// GROUP B (Bottom layer → Red): main stem rises from right
const B0 = bz([3.6,2.7],[3.45,2.3],[3.1,1.8],[2.7,1.45],18)
  .concat(bz([2.7,1.45],[2.3,1.15],[1.9,0.9],[1.55,0.72],14))
  .concat(bz([1.55,0.72],[1.35,0.62],[1.2,0.55],[1.05,0.5],8));
// B branch 1: from (2.7,1.45) going left-down
const B1 = bz([2.7,1.45],[2.3,1.3],[1.8,1.25],[1.4,1.35],10)
  .concat(bz([1.4,1.35],[1.1,1.42],[0.8,1.5],[0.5,1.58],8));
// B branch 2: from (2.3,1.15) going right-up
const B2 = bz([2.3,1.15],[2.7,0.9],[3.2,0.7],[3.65,0.58],12)
  .concat(bz([3.65,0.58],[4.0,0.5],[4.3,0.43],[4.6,0.38],8));
// B branch 3: from (3.1,1.8) going right
const B3 = bz([3.1,1.8],[3.5,1.65],[3.9,1.5],[4.3,1.42],10);

const allA=[A0,A1,A2,A3], allB=[B0,B1,B2,B3];
const tA=[[26,5],[16,5],[10,4],[9,4]], tB=[[24,5],[14,4],[14,5],[16,5]];

// Exact junction coordinates (where branches split from parent)
const jA = [[1.65,1.35],[1.0,0.4],[2.5,0.75]]; // A0→A1, A0→A2, A1→A3
const jB = [[2.7,1.45],[2.3,1.15],[3.1,1.8]];   // B0→B1, B0→B2, B0→B3

// Exact endpoint coordinates (tips)
const eA = [[0.45,0.18],[3.9,0.22],[0.4,0.7],[2.65,0.15]];
const eB = [[1.05,0.5],[0.5,1.58],[4.6,0.38],[4.3,1.42]];
const rA=[2.0,2.7], rB=[3.6,2.7]; // roots at ECM

// Overlap zone: A0 passes through (1.65,1.35)→(1.0,0.4) while B0 passes through (2.3,1.15)→(1.55,0.72)
// They cross around (1.6-1.8, 0.8-1.0)
const crossX = 1.72, crossY = 0.92;

// ====== HEADERS ======
slide.addText("a",{x:0.08,y:row1Y-0.02,w:0.22,h:0.3,fontSize:18,fontFace:"Arial",bold:true,color:BLK,margin:0});
slide.addText("b",{x:0.08,y:row2Y-0.02,w:0.22,h:0.3,fontSize:18,fontFace:"Arial",bold:true,color:BLK,margin:0});
slide.addText("Single-color Fluorescence",{x:vX,y:row1Y-0.38,w:vW,h:0.28,fontSize:12,fontFace:"Arial",bold:true,color:GRN,align:"center",margin:0});
slide.addText("Multi-color Depth-encoded",{x:vX,y:row2Y-0.38,w:vW,h:0.28,fontSize:12,fontFace:"Arial",bold:true,color:BLUE,align:"center",margin:0});
slide.addText("Skeletonization",{x:sX,y:row1Y-0.38,w:sW,h:0.28,fontSize:12,fontFace:"Arial",bold:true,color:GRY_D,align:"center",margin:0});
slide.addText("Depth-separated Skeletonization",{x:sX,y:row2Y-0.38,w:sW,h:0.28,fontSize:12,fontFace:"Arial",bold:true,color:GRY_D,align:"center",margin:0});

// ====== PANELS ======
for(const[px,py,pw] of [[vX,row1Y,vW],[sX,row1Y,sW],[vX,row2Y,vW],[sX,row2Y,sW]]){
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:px,y:py,w:pw,h:rowH,fill:{color:PNL},rectRadius:0.06,line:{color:PNL_B,width:0.8}});
}

// ====== ECM BASELINE ======
for(const px of [vX,sX]){for(const py of [row1Y,row2Y]){
  slide.addShape(pres.shapes.RECTANGLE,{x:px+0.06,y:py+2.62,w:vW-0.12,h:0.3,fill:{color:"ECF0F1"}});
  slide.addShape(pres.shapes.LINE,{x:px+0.06,y:py+2.62,w:vW-0.12,h:0,line:{color:GRY_L,width:1}});
  slide.addText("ECM",{x:px+vW-0.65,y:py+2.67,w:0.5,h:0.18,fontSize:6.5,fontFace:"Arial",italic:true,color:GRY,align:"right",margin:0});
}}

// ==================================
// ROW 1: SINGLE-COLOR
// ==================================
const v1=vX, o1=row1Y;
for(let i=0;i<allA.length;i++) tapered(slide,O(allA[i],v1,o1),GRN_L,tA[i][0],tA[i][1],65);
for(let i=0;i<allB.length;i++) tapered(slide,O(allB[i],v1,o1),GRN_L,tB[i][0],tB[i][1],60);

// Overlap highlight
slide.addShape(pres.shapes.OVAL,{x:v1+crossX-0.42,y:o1+crossY-0.35,w:0.84,h:0.7,
  fill:{color:"FCF3CF",transparency:40},line:{color:GRY,width:0.8,dashType:"dash"}});
slide.addText("overlap",{x:v1+crossX-0.35,y:o1+crossY+0.37,w:0.7,h:0.15,fontSize:6.5,fontFace:"Arial",italic:true,color:GRY,align:"center",margin:0});

// === SINGLE SKELETON ===
const s1=sX, q1=row1Y;
// Merged skeleton in overlap zone
const A0_bot=bz([2.0,2.7],[1.95,2.3],[1.8,1.7],[1.65,1.35],16);
const A0_top=bz([1.0,0.4],[0.8,0.28],[0.6,0.2],[0.45,0.18],8);
const B0_bot=bz([3.6,2.7],[3.45,2.3],[3.1,1.8],[2.7,1.45],16);
const B0_top=bz([1.55,0.72],[1.35,0.62],[1.2,0.55],[1.05,0.5],8);

// Merge: from where B enters A zone to where they separate
// A goes: (1.65,1.35)→(1.5,1.0)→(1.3,0.65)→(1.0,0.4)
// B goes: (2.3,1.15)→(1.9,0.9)→(1.55,0.72)
// They merge around (1.65,1.0) and separate around (1.3,0.7)
const mergeIn = bz([2.7,1.45],[2.3,1.15],[1.9,0.95],[1.65,1.35],10);  // B enters from right
const mergeThru = bz([1.65,1.35],[1.5,1.0],[1.35,0.78],[1.0,0.4],12); // merged path
const mergeOut = bz([1.35,0.78],[1.45,0.72],[1.5,0.7],[1.55,0.72],6); // B exits

skel(slide,O(A0_bot,s1,q1),GRN);
skel(slide,O(mergeThru,s1,q1),GRN,4); // merged (thicker)
skel(slide,O(A0_top,s1,q1),GRN);
skel(slide,O(B0_bot,s1,q1),GRN);
skel(slide,O(mergeIn,s1,q1),GRN);
skel(slide,O(mergeOut,s1,q1),GRN);
skel(slide,O(B0_top,s1,q1),GRN);
for(const p of [A1,A2,A3,B1,B2,B3]) skel(slide,O(p,s1,q1),GRN);

// FALSE JUNCTIONS at merge entry (1.65,1.35) and merge split (1.35,0.78)
const fj1=[1.65,1.35], fj2=[1.35,0.78];
falseJ(slide, s1+fj1[0], q1+fj1[1]);
falseJ(slide, s1+fj2[0], q1+fj2[1]);

// Annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{
  x:s1+2.7,y:q1+0.25,w:2.0,h:0.7,
  fill:{color:"FDEDEC"},rectRadius:0.06,line:{color:"F5B7B1",width:0.8},
});
slide.addText([
  {text:"False junctions\n",options:{bold:true,fontSize:9,color:RED}},
  {text:"Vessels at different z-depths\noverlap and create false\nbranch points in skeleton",options:{fontSize:7,color:"922B21"}},
],{x:s1+2.75,y:q1+0.28,w:1.9,h:0.65,margin:0});
// Lines to false junctions
skel(slide,[[s1+2.7,q1+0.55],[s1+fj1[0]+0.18,q1+fj1[1]]],RED,1);
skel(slide,[[s1+2.7,q1+0.7],[s1+fj2[0]+0.18,q1+fj2[1]]],RED,1);

// Length annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{
  x:s1+0.65,y:q1+1.65,w:1.7,h:0.35,
  fill:{color:"FEF9E7"},rectRadius:0.05,line:{color:YEL,width:0.8,dashType:"dash"},
});
slide.addText("Merged → single skeleton path, length under-estimated",{
  x:s1+0.65,y:q1+1.65,w:1.7,h:0.35,
  fontSize:5.5,fontFace:"Arial",bold:true,color:"7D6608",align:"center",valign:"middle",margin:0,
});
skel(slide,[[s1+1.5,q1+1.65],[s1+1.5,q1+1.3]],YEL,0.8);

// Real junctions (exact branch points)
for(const j of jA) junc(slide,s1+j[0],q1+j[1],GRN);
for(const j of jB) junc(slide,s1+j[0],q1+j[1],GRN);

// Endpoints
for(const e of eA) endpt(slide,s1+e[0],q1+e[1],GRN);
for(const e of eB) endpt(slide,s1+e[0],q1+e[1],GRN);
endpt(slide,s1+rA[0],q1+rA[1],GRN);
endpt(slide,s1+rB[0],q1+rB[1],GRN);


// ==================================
// ROW 2: MULTI-COLOR
// ==================================
const v2=vX, o2=row2Y;
for(let i=0;i<allA.length;i++) tapered(slide,O(allA[i],v2,o2),BLUE_L,tA[i][0],tA[i][1],65);
for(let i=0;i<allB.length;i++) tapered(slide,O(allB[i],v2,o2),RED_L,tB[i][0],tB[i][1],60);

// Yellow at transitions
const yw1=bz([2.0,1.1],[1.85,1.0],[1.75,0.92],[1.65,0.85],6);
tapered(slide,O(yw1,v2,o2),YEL_L,10,5,50);

slide.addShape(pres.shapes.OVAL,{x:v2+crossX-0.42,y:o2+crossY-0.35,w:0.84,h:0.7,
  fill:{color:"FDEBD0",transparency:40},line:{color:YEL,width:0.8,dashType:"dash"}});
slide.addText("different\nz-depths",{x:v2+crossX-0.38,y:o2+crossY+0.37,w:0.76,h:0.2,fontSize:6.5,fontFace:"Arial",italic:true,color:YEL,align:"center",margin:0});

// Legend
const cl=o2+2.42;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:v2+0.5,y:cl,w:4.0,h:0.28,fill:{color:"FFFFFF"},rectRadius:0.04,line:{color:PNL_B,width:0.5}});
skel(slide,[[v2+0.7,cl+0.14],[v2+1.0,cl+0.14]],RED,4);
slide.addText("Bottom",{x:v2+1.05,y:cl+0.02,w:0.55,h:0.24,fontSize:7.5,fontFace:"Arial",color:RED,margin:0,valign:"middle"});
skel(slide,[[v2+1.65,cl+0.14],[v2+1.95,cl+0.14]],YEL,4);
slide.addText("Middle",{x:v2+2.0,y:cl+0.02,w:0.55,h:0.24,fontSize:7.5,fontFace:"Arial",color:YEL,margin:0,valign:"middle"});
skel(slide,[[v2+2.6,cl+0.14],[v2+2.9,cl+0.14]],BLUE,4);
slide.addText("Top",{x:v2+2.95,y:cl+0.02,w:0.35,h:0.24,fontSize:7.5,fontFace:"Arial",color:BLUE,margin:0,valign:"middle"});
slide.addText("← Deep    Shallow →",{x:v2+3.3,y:cl+0.02,w:1.0,h:0.24,fontSize:6,fontFace:"Arial",color:GRY,margin:0,valign:"middle"});

// === MULTI SKELETON ===
const s2=sX, q2=row2Y;
for(const p of allA) skel(slide,O(p,s2,q2),BLUE);
for(const p of allB) skel(slide,O(p,s2,q2),RED);
skel(slide,O(yw1,s2,q2),YEL,2.5);

// CORRECT marks at same positions
correctJ(slide, s2+fj1[0], q2+fj1[1]);
correctJ(slide, s2+fj2[0], q2+fj2[1]);

// Annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{
  x:s2+2.7,y:q2+0.25,w:2.0,h:0.7,
  fill:{color:"EAFAF1"},rectRadius:0.06,line:{color:"A9DFBF",width:0.8},
});
slide.addText([
  {text:"No false junctions\n",options:{bold:true,fontSize:9,color:GRN}},
  {text:"Each layer skeletonized\nindependently — cross-layer\noverlaps correctly ignored",options:{fontSize:7,color:"196F3D"}},
],{x:s2+2.75,y:q2+0.28,w:1.9,h:0.65,margin:0});
skel(slide,[[s2+2.7,q2+0.55],[s2+fj1[0]+0.18,q2+fj1[1]]],GRN,1);
skel(slide,[[s2+2.7,q2+0.7],[s2+fj2[0]+0.18,q2+fj2[1]]],GRN,1);

// Length annotation
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{
  x:s2+0.65,y:q2+1.65,w:1.7,h:0.35,
  fill:{color:"EAFAF1"},rectRadius:0.05,line:{color:GRN,width:0.8,dashType:"dash"},
});
slide.addText("2 independent paths → total length recovered",{
  x:s2+0.65,y:q2+1.65,w:1.7,h:0.35,
  fontSize:5.5,fontFace:"Arial",bold:true,color:"196F3D",align:"center",valign:"middle",margin:0,
});
skel(slide,[[s2+1.5,q2+1.65],[s2+1.5,q2+1.3]],GRN,0.8);

// Blue junctions (exact branch points)
for(const j of jA) junc(slide,s2+j[0],q2+j[1],BLUE);
// Red junctions
for(const j of jB) junc(slide,s2+j[0],q2+j[1],RED);

// Endpoints color-matched
for(const e of eA) endpt(slide,s2+e[0],q2+e[1],BLUE);
for(const e of eB) endpt(slide,s2+e[0],q2+e[1],RED);
endpt(slide,s2+rA[0],q2+rA[1],BLUE);
endpt(slide,s2+rB[0],q2+rB[1],RED);

// ====== ARROWS ======
for(const ry of [row1Y,row2Y]){
  slide.addShape(pres.shapes.LINE,{x:aX,y:ry+rowH/2,w:aW,h:0,line:{color:GRY_D,width:1.8,endArrowType:"triangle"}});
}

// ====== RIGHT ANNOTATIONS ======
slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:anX,y:row1Y+0.15,w:anW,h:2.7,fill:{color:"FDEDEC"},rectRadius:0.06,line:{color:"F5B7B1",width:0.8}});
slide.addText("Limitations",{x:anX,y:row1Y+0.22,w:anW,h:0.2,fontSize:9,fontFace:"Arial",bold:true,color:RED,align:"center",margin:0});
slide.addText([
  {text:"Junctions\n",options:{bold:true,fontSize:8,color:BLK}},
  {text:"Over-counted\n\n",options:{fontSize:6.5,color:"922B21"}},
  {text:"Length\n",options:{bold:true,fontSize:8,color:BLK}},
  {text:"Under-estimated\n\n",options:{fontSize:6.5,color:"922B21"}},
  {text:"Endpoints\n",options:{bold:true,fontSize:8,color:BLK}},
  {text:"Preserved\n\n",options:{fontSize:6.5,color:"922B21"}},
  {text:"Depth info\n",options:{bold:true,fontSize:8,color:BLK}},
  {text:"Lost",options:{fontSize:6.5,color:"922B21"}},
],{x:anX+0.06,y:row1Y+0.48,w:anW-0.12,h:2.3,margin:0,paraSpaceAfter:1});

slide.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:anX,y:row2Y+0.15,w:anW,h:2.7,fill:{color:"EAFAF1"},rectRadius:0.06,line:{color:"A9DFBF",width:0.8}});
slide.addText("Advantages",{x:anX,y:row2Y+0.22,w:anW,h:0.2,fontSize:9,fontFace:"Arial",bold:true,color:GRN,align:"center",margin:0});
slide.addText([
  {text:"Junctions\n",options:{bold:true,fontSize:8,color:BLK}},
  {text:"Accurate\n\n",options:{fontSize:6.5,color:"196F3D"}},
  {text:"Length\n",options:{bold:true,fontSize:8,color:BLK}},
  {text:"Recovered\n\n",options:{fontSize:6.5,color:"196F3D"}},
  {text:"Endpoints\n",options:{bold:true,fontSize:8,color:BLK}},
  {text:"Preserved\n\n",options:{fontSize:6.5,color:"196F3D"}},
  {text:"Depth info\n",options:{bold:true,fontSize:8,color:BLK}},
  {text:"Preserved",options:{fontSize:6.5,color:"196F3D"}},
],{x:anX+0.06,y:row2Y+0.48,w:anW-0.12,h:2.3,margin:0,paraSpaceAfter:1});

// ====== DIVIDER ======
slide.addShape(pres.shapes.LINE,{x:0.3,y:(row1Y+rowH+row2Y-0.38)/2,w:12.7,h:0,line:{color:GRY_L,width:0.6,dashType:"lgDash"}});

// ====== BOTTOM LEGEND ======
const ly=7.15;
junc(slide,sX+0.5,ly,GRY_D,0.065);
slide.addText("Junction",{x:sX+0.6,y:ly-0.07,w:0.55,h:0.14,fontSize:7,fontFace:"Arial",color:GRY_D,margin:0});
endpt(slide,sX+1.3,ly,GRY_D,0.05);
slide.addText("Endpoint",{x:sX+1.4,y:ly-0.07,w:0.55,h:0.14,fontSize:7,fontFace:"Arial",color:GRY_D,margin:0});
skel(slide,[[sX+2.1,ly],[sX+2.35,ly]],GRY_D,3);
slide.addText("Skeleton",{x:sX+2.4,y:ly-0.07,w:0.55,h:0.14,fontSize:7,fontFace:"Arial",color:GRY_D,margin:0});
falseJ(slide,sX+3.15,ly,0.07);
slide.addText("False junction",{x:sX+3.27,y:ly-0.07,w:0.8,h:0.14,fontSize:7,fontFace:"Arial",color:RED,margin:0});
correctJ(slide,sX+4.25,ly,0.07);
slide.addText("Correctly separated",{x:sX+4.37,y:ly-0.07,w:1.05,h:0.14,fontSize:7,fontFace:"Arial",color:GRN,margin:0});

pres.writeFile({fileName:"depth_schematic_v9.pptx"}).then(()=>console.log("Created: depth_schematic_v9.pptx"));
