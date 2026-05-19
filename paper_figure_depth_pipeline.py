# Paper Figure: Depth-aware vessel analysis pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage
import json, warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size':9,'font.family':'Arial','axes.titlesize':10})

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

s1 = np.array(Image.open(DATA/'Original color stack data'/'Sample 1_BF and GT(512x512 two images).png'))
bf_raw, gt_raw = s1[:512], s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7,7), 0)
h, w = gt_raw.shape[:2]
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff_BR = B - R
by = int(h * 0.8)
tm = np.zeros((h,w), dtype=bool); tm[:by,:] = True

# --- Skeletons ---
rm = cv2.morphologyEx((R>30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
rm = remove_small_objects(rm, min_size=50)
rs = skeletonize(rm)

bs_ = B > 50; bl = B > 20
bll, nl = ndimage.label(bl)
bh = np.zeros_like(bl, dtype=bool)
for i in range(1, nl+1):
    reg = bll == i
    if np.any(bs_[reg]): bh |= reg
bh = cv2.morphologyEx(bh.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
bh = remove_small_objects(bh, min_size=50)
bsk = skeletonize(bh)

va = remove_small_objects(cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
base = skeletonize(va)
vml = cv2.dilate(va.astype(np.uint8)*255, k5) > 0
rf = rs & (diff_BR < 0)
bf = bsk & (diff_BR > 0)

# --- Bridging ---
def gept(sk, tl=20):
    su = sk.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    eps = np.argwhere(nb == 1); res = []
    for ep in eps:
        y0, x0 = ep; path = [(y0,x0)]; vis = {(y0,x0)}; cy, cx = y0, x0
        for _ in range(tl):
            fd = False
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dy==0 and dx==0: continue
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and sk[ny,nx] and (ny,nx) not in vis:
                        path.append((ny,nx)); vis.add((ny,nx)); cy,cx=ny,nx; fd=True; break
                if fd: break
            if not fd: break
        if len(path) >= 5:
            na = min(len(path), 10)
            t = np.array(path[0], dtype=float) - np.array(path[na-1], dtype=float)
            n = np.linalg.norm(t)
            if n > 0: t /= n
            res.append({'pos':(y0,x0), 'tangent':t})
    return res

def bezier(e1, e2):
    p1 = np.array(e1['pos'], dtype=float); p2 = np.array(e2['pos'], dtype=float)
    t1, t2 = e1['tangent'], e2['tangent']
    d = np.linalg.norm(p2-p1)
    c1 = p1 + t1*d*0.4; c2 = p2 + t2*d*0.4
    pts = []
    for t in np.linspace(0, 1, max(int(d*1.5), 10)):
        p = (1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

def dobr(sk, vm, md=80, ma=60):
    eps = gept(sk); cands = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1 = np.array(eps[i]['pos'], dtype=float)
            p2 = np.array(eps[j]['pos'], dtype=float)
            d = np.linalg.norm(p2-p1)
            if d < 5 or d > md: continue
            d12 = (p2-p1)/d
            c1 = np.dot(eps[i]['tangent'], d12)
            c2 = np.dot(eps[j]['tangent'], -d12)
            th = np.cos(np.radians(ma))
            if c1 < th or c2 < th: continue
            pts = bezier(eps[i], eps[j])
            ov = sum(1 for(y,x) in pts if 0<=y<h and 0<=x<w and vm[y,x])
            vr = ov / max(len(pts), 1)
            if vr < 0.7: continue
            cands.append((i, j, (c1+c2)/2 - d/md*0.2 + vr*0.2))
    cands.sort(key=lambda x: -x[2])
    used = set(); br = sk.copy(); nb = 0
    for i, j, sc in cands:
        if i not in used and j not in used:
            for (y,x) in bezier(eps[i], eps[j]):
                if 0<=y<h and 0<=x<w: br[y,x] = True
            used.add(i); used.add(j); nb += 1
    return skeletonize(br), nb

rb, rnb = dobr(rf, vml)
bb, bnb = dobr(bf, vml)

# --- EP/JN ---
def epjn(sk, mask):
    s = sk & mask; su = s.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    L = int(np.sum(s))
    el, ne = ndimage.label(nb == 1); epc = []
    for i in range(1, ne+1):
        pts = np.argwhere(el==i); cy,cx = pts.mean(axis=0)
        y, x = int(round(cy)), int(round(cx))
        if y < by - 5: epc.append((y,x))
    jd = cv2.dilate((nb>=3).astype(np.uint8), k3)
    jl, nj = ndimage.label(jd); jnc = []
    for i in range(1, nj+1):
        cm = (jl==i) & (nb>=3); pts = np.argwhere(cm)
        if len(pts)==0: pts = np.argwhere(jl==i)
        cy, cx = pts.mean(axis=0); jy,jx = int(round(cy)), int(round(cx))
        ca = (jl==i); ys,xs = np.where(ca); pad = 3
        l0 = max(0,ys.min()-pad); l1 = min(h,ys.max()+pad+1)
        x0 = max(0,xs.min()-pad); x1 = min(w,xs.max()+pad+1)
        ls = s[l0:l1,x0:x1].copy(); ls[ca[l0:l1,x0:x1]] = False
        _, nb2 = ndimage.label(ls)
        if nb2 >= 3: jnc.append((jy,jx))
    return epc, jnc, L

se, sj, sL = epjn(base, tm)
re, rjn, rL = epjn(rb, tm)
be, bjn, bL = epjn(bb, tm)

rdl = cv2.dilate(rb.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))) > 0
bdl = cv2.dilate(bb.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))) > 0
rr = [(y,x) for y,x in re if not bdl[y,x]]
rcn = [(y,x) for y,x in re if bdl[y,x]]
brl = [(y,x) for y,x in be if not rdl[y,x]]
bcn = [(y,x) for y,x in be if rdl[y,x]]

# --- Load quant data ---
with open(OUT/'final_results_clean_crop10.json') as f:
    qd = json.load(f)

def am(dl):
    return {k: np.mean([d[k] for d in dl]) for k in ['junctions','endpoints','length']}

gsa = am(qd['GT']['single']); gma = am(qd['GT']['multi'])
mods = {}
for mn in ['LBBDM','PBBDM']:
    if mn in qd: mods[mn] = {'s': am(qd[mn]['single']), 'm': am(qd[mn]['multi'])}

# ===== FIGURE =====
fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.25, height_ratios=[1,1,1,0.9])
ct = lambda img: img[:by]

# Row 0
ax=fig.add_subplot(gs[0,0]); ax.imshow(ct(bf_raw)); ax.set_title('(a) Brightfield',fontweight='bold'); ax.axis('off')
ax=fig.add_subplot(gs[0,1]); ax.imshow(ct(gt_raw)); ax.set_title('(b) Multi-color GT',fontweight='bold'); ax.axis('off')

ax=fig.add_subplot(gs[0,2])
rv=np.zeros((by,w,3),dtype=np.uint8); rc=np.clip(R[:by]*1.5,0,255).astype(np.uint8)
rv[:,:,0]=rc; rv[:,:,1]=(rc*0.2).astype(np.uint8)
ax.imshow(rv); ax.set_title('(c) R channel (bottom)',fontweight='bold'); ax.axis('off')

ax=fig.add_subplot(gs[0,3])
bvis=np.zeros((by,w,3),dtype=np.uint8); bcc=np.clip(B[:by]*1.5,0,255).astype(np.uint8)
bvis[:,:,2]=bcc; bvis[:,:,1]=(bcc*0.2).astype(np.uint8)
ax.imshow(bvis); ax.set_title('(d) B channel (top)',fontweight='bold'); ax.axis('off')

# Row 1
ax=fig.add_subplot(gs[1,0])
dv=np.zeros((by,w,3),dtype=np.uint8); dv[diff_BR[:by]<0]=[180,50,50]; dv[diff_BR[:by]>0]=[50,80,180]; dv[~va[:by]]=[0,0,0]
ax.imshow(dv); ax.set_title('(e) Dominance map',fontweight='bold'); ax.axis('off')

ax=fig.add_subplot(gs[1,1]); v=ct(gt_raw).copy().astype(float)*0.2; v[cv2.dilate(rf[:by].astype(np.uint8),dk4)>0]=[255,80,80]
ax.imshow(v.astype(np.uint8)); ax.set_title('(f) R skeleton (R>B)',fontweight='bold'); ax.axis('off')

ax=fig.add_subplot(gs[1,2]); v=ct(gt_raw).copy().astype(float)*0.2; v[cv2.dilate(bf[:by].astype(np.uint8),dk4)>0]=[80,150,255]
ax.imshow(v.astype(np.uint8)); ax.set_title('(g) B skeleton (B>R)',fontweight='bold'); ax.axis('off')

ax=fig.add_subplot(gs[1,3]); v=ct(gt_raw).copy().astype(float)*0.2
v[cv2.dilate(rf[:by].astype(np.uint8),dk4)>0]=[255,80,80]; v[cv2.dilate(bf[:by].astype(np.uint8),dk4)>0]=[80,150,255]
ax.imshow(v.astype(np.uint8)); ax.set_title('(h) R+B overlay',fontweight='bold'); ax.axis('off')

# Row 2
ax=fig.add_subplot(gs[2,0]); v=ct(gt_raw).copy().astype(float)*0.2; v[cv2.dilate(base[:by].astype(np.uint8),dk4)>0]=[0,200,0]
for y,x in se: cv2.circle(v,(x,y),6,(255,255,0),2)
for y,x in sj: cv2.circle(v,(x,y),6,(255,0,255),2)
ax.imshow(v.astype(np.uint8)); ax.set_title(f'(i) Single GT\nL={sL} EP={len(se)} JN={len(sj)}',fontweight='bold'); ax.axis('off')

ax=fig.add_subplot(gs[2,1]); v=ct(gt_raw).copy().astype(float)*0.2; v[cv2.dilate(rb[:by].astype(np.uint8),dk4)>0]=[255,80,80]
for y,x in rr: cv2.circle(v,(x,y),6,(255,255,0),2)
for y,x in rcn: cv2.circle(v,(x,y),6,(0,255,255),2)
for y,x in rjn: cv2.circle(v,(x,y),5,(255,0,255),2)
ax.imshow(v.astype(np.uint8)); ax.set_title(f'(j) R bridged (+{rnb})\nL={rL} EP={len(rr)}+{len(rcn)} JN={len(rjn)}',fontweight='bold',fontsize=9); ax.axis('off')

ax=fig.add_subplot(gs[2,2]); v=ct(gt_raw).copy().astype(float)*0.2; v[cv2.dilate(bb[:by].astype(np.uint8),dk4)>0]=[80,150,255]
for y,x in brl: cv2.circle(v,(x,y),6,(255,255,0),2)
for y,x in bcn: cv2.circle(v,(x,y),6,(0,255,255),2)
for y,x in bjn: cv2.circle(v,(x,y),5,(255,0,255),2)
ax.imshow(v.astype(np.uint8)); ax.set_title(f'(k) B bridged (+{bnb})\nL={bL} EP={len(brl)}+{len(bcn)} JN={len(bjn)}',fontweight='bold',fontsize=9); ax.axis('off')

ax=fig.add_subplot(gs[2,3]); v=np.zeros((by,w,3),dtype=np.uint8)
v[cv2.dilate(rb[:by].astype(np.uint8),dk4)>0]=[220,60,60]; v[cv2.dilate(bb[:by].astype(np.uint8),dk4)>0]=[60,130,220]
for y,x in rr+brl: cv2.circle(v,(x,y),6,(255,255,0),2)
for y,x in rcn+bcn: cv2.circle(v,(x,y),6,(0,255,255),2)
for y,x in rjn+bjn: cv2.circle(v,(x,y),5,(255,0,255),2)
ax.imshow(v); ax.set_title(f'(l) R+B final\nL={rL+bL} EP(real)={len(rr)+len(brl)} JN={len(rjn)+len(bjn)}',fontweight='bold',fontsize=9); ax.axis('off')

# Row 3: Quantification
ax=fig.add_subplot(gs[3,0:2])
grps=['GT\nSingle','GT\nMulti']; gv={'L':[gsa['length'],gma['length']],'EP':[gsa['endpoints'],gma['endpoints']],'JN':[gsa['junctions'],gma['junctions']]}
for mn in ['LBBDM','PBBDM']:
    if mn in mods:
        grps.extend([f'{mn}\nSingle',f'{mn}\nMulti'])
        for k,mk in [('L','length'),('EP','endpoints'),('JN','junctions')]:
            gv[k].extend([mods[mn]['s'][mk], mods[mn]['m'][mk]])
x=np.arange(len(grps)); wd=0.25
for idx,(m,c) in enumerate([('L','#f39c12'),('EP','#e74c3c'),('JN','#9b59b6')]):
    nv=[v/gsa[{'L':'length','EP':'endpoints','JN':'junctions'}[m]] for v in gv[m]]
    ax.bar(x+(idx-1)*wd, nv, wd, label={'L':'Length','EP':'Endpoints','JN':'Junctions'}[m], color=c, alpha=0.75)
ax.set_xticks(x); ax.set_xticklabels(grps,fontsize=8)
ax.axhline(1.0,color='gray',ls='--',alpha=0.5)
ax.set_ylabel('Normalized (GT Single=1.0)')
ax.set_title('(m) Normalized metrics (N=215)',fontweight='bold')
ax.legend(fontsize=8); ax.set_ylim(0,2.5)

ax=fig.add_subplot(gs[3,2:4]); ax.axis('off')
txt=(f"Sample 1 Quantification (top 80%)\n{'='*48}\n"
     f"{'':22s} {'L':>6s} {'EP':>5s} {'JN':>5s}\n{'-'*48}\n"
     f"{'Single GT (base)':22s} {sL:6d} {len(se):5d} {len(sj):5d}\n\n"
     f"{'R (R>B, bridged)':22s} {rL:6d} {len(re):5d} {len(rjn):5d}\n"
     f"{'  EP: real':22s} {'':6s} {len(rr):5d}\n{'  EP: connected':22s} {'':6s} {len(rcn):5d}\n\n"
     f"{'B (B>R, bridged)':22s} {bL:6d} {len(be):5d} {len(bjn):5d}\n"
     f"{'  EP: real':22s} {'':6s} {len(brl):5d}\n{'  EP: connected':22s} {'':6s} {len(bcn):5d}\n\n"
     f"{'-'*48}\n{'R + B sum':22s} {rL+bL:6d} {len(re)+len(be):5d} {len(rjn)+len(bjn):5d}\n"
     f"{'R + B (real EP)':22s} {'':6s} {len(rr)+len(brl):5d}\n\n"
     f"  Yellow = Real EP | Cyan = Connected EP\n  Magenta = Junction")
ax.text(0.05,0.95,txt,fontsize=10,transform=ax.transAxes,fontfamily='monospace',color='black',va='top',
        bbox=dict(boxstyle='round,pad=0.5',facecolor='#f8f8f8',edgecolor='#cccccc'))
ax.set_title('(n) Quantification summary',fontweight='bold')

plt.savefig(OUT/'paper_figure_depth_pipeline.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.close()
print('Saved paper_figure_depth_pipeline.png')

# --- Individual panels ---
pd = OUT/'paper_panels_depth'; pd.mkdir(exist_ok=True)
def sp(img, name):
    f,a = plt.subplots(figsize=(5,4)); a.imshow(img); a.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(pd/f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='black')
    plt.close()

sp(ct(bf_raw),'a_bf'); sp(ct(gt_raw),'b_gt'); sp(rv,'c_r_ch'); sp(bvis,'d_b_ch'); sp(dv,'e_dom')
for nm,sk,c in [('f_r_sk',rf,[255,80,80]),('g_b_sk',bf,[80,150,255])]:
    v=ct(gt_raw).copy().astype(float)*0.2; v[cv2.dilate(sk[:by].astype(np.uint8),dk4)>0]=c; sp(v.astype(np.uint8),nm)
v=ct(gt_raw).copy().astype(float)*0.2; v[cv2.dilate(base[:by].astype(np.uint8),dk4)>0]=[0,200,0]
for y,x in se: cv2.circle(v,(x,y),6,(255,255,0),2)
for y,x in sj: cv2.circle(v,(x,y),6,(255,0,255),2)
sp(v.astype(np.uint8),'i_single')
for nm,sk,epr,epc,jns,c in [('j_r_br',rb,rr,rcn,rjn,[255,80,80]),('k_b_br',bb,brl,bcn,bjn,[80,150,255])]:
    v=ct(gt_raw).copy().astype(float)*0.2; v[cv2.dilate(sk[:by].astype(np.uint8),dk4)>0]=c
    for y,x in epr: cv2.circle(v,(x,y),6,(255,255,0),2)
    for y,x in epc: cv2.circle(v,(x,y),6,(0,255,255),2)
    for y,x in jns: cv2.circle(v,(x,y),5,(255,0,255),2)
    sp(v.astype(np.uint8),nm)
v=np.zeros((by,w,3),dtype=np.uint8)
v[cv2.dilate(rb[:by].astype(np.uint8),dk4)>0]=[220,60,60]; v[cv2.dilate(bb[:by].astype(np.uint8),dk4)>0]=[60,130,220]
for y,x in rr+brl: cv2.circle(v,(x,y),6,(255,255,0),2)
for y,x in rcn+bcn: cv2.circle(v,(x,y),6,(0,255,255),2)
for y,x in rjn+bjn: cv2.circle(v,(x,y),5,(255,0,255),2)
sp(v,'l_comb')
print(f'Saved {len(list(pd.glob("*.png")))} panels')
print('Done!')
