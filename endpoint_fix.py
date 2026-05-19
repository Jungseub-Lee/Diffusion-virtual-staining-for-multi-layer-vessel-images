"""
Fixed endpoint calculation:
- Per-layer independent endpoint detection
- R↔Y boundary endpoints: remove (same vessel transitioning)
- Y↔B boundary endpoints: remove (same vessel transitioning)
- R↔B meeting: keep as independent endpoints (different depth vessels)
"""
import numpy as np
from PIL import Image
from pathlib import Path
import cv2, json, warnings
from skimage.morphology import skeletonize, remove_small_objects
from scipy.spatial.distance import cdist
from scipy import stats
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data")
OUT = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")

def load(p): return np.array(Image.open(p).convert("RGB"))

def separate_layers(img, min_area=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[:,:,0], hsv[:,:,1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)
    b = ((h<=12)|(h>=155))&fg; m = ((h>=13)&(h<=55))&fg; t = ((h>=85)&(h<=140))&fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    def c(mask):
        u = mask.astype(np.uint8)*255
        u = cv2.morphologyEx(u,cv2.MORPH_CLOSE,k,iterations=2)
        u = cv2.morphologyEx(u,cv2.MORPH_OPEN,k,iterations=1)
        return remove_small_objects(u>0, min_size=min_area)
    return c(b),c(m),c(t)

def get_vessel_mask(img, min_area=50):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    u = v.astype(np.uint8)*255
    u = cv2.morphologyEx(u,cv2.MORPH_CLOSE,k,iterations=2)
    u = cv2.morphologyEx(u,cv2.MORPH_OPEN,k,iterations=1)
    return remove_small_objects(u>0, min_size=min_area)

def get_baseline_y(img, thr=0.35):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15; h, w = v.shape
    rd = np.sum(v, axis=1) / w
    for y in range(h-1,-1,-1):
        if rd[y] < thr: return y
    return h

def make_valid_mask(img):
    h, w = img.shape[:2]
    by = get_baseline_y(img)
    vm = np.ones((h,w), dtype=bool); vm[by:,:] = False
    return vm, by

def find_features(skeleton):
    su = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su,-1,ks)*su
    return np.argwhere(nb==1), np.argwhere(nb>=3)

def count_in_mask(coords, mask):
    if len(coords) == 0: return 0
    return sum(1 for y,x in coords if mask[y,x])

def count_true_endpoints(skels, masks, vm, radius=15):
    """
    Per-layer independent endpoint calculation.
    - Find endpoints on each layer's skeleton independently
    - Remove boundary endpoints: those near ADJACENT layer's vessel
      (R↔Y, Y↔B = same vessel transitioning depth)
    - Keep R↔B endpoints: different depth, independent vessels

    Layer 0=Bottom(R), 1=Middle(Y), 2=Top(B)
    Adjacent pairs: (0,1) and (1,2)
    Non-adjacent: (0,2) = R↔B → keep endpoints
    """
    adjacent_pairs = [(0,1), (1,2)]  # R-Y, Y-B

    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))

    # Dilated masks for proximity check
    dilated = []
    for m in masks:
        d = cv2.dilate(m.astype(np.uint8)*255, dk, iterations=1) > 0
        dilated.append(d)

    total_true_ep = 0

    for li in range(3):
        skel = skels[li]
        if np.sum(skel) == 0:
            continue

        ep_coords, _ = find_features(skel)

        for ey, ex in ep_coords:
            if not vm[ey, ex]:
                continue  # outside valid region

            # Check if this endpoint is near an ADJACENT layer's vessel
            is_boundary = False
            for la, lb in adjacent_pairs:
                if li == la:
                    # This layer's endpoint near adjacent layer lb?
                    if dilated[lb][ey, ex]:
                        is_boundary = True
                        break
                elif li == lb:
                    # This layer's endpoint near adjacent layer la?
                    if dilated[la][ey, ex]:
                        is_boundary = True
                        break

            if not is_boundary:
                total_true_ep += 1
                # R(0) endpoint near B(2) or B endpoint near R → NOT boundary, keep it

    return total_true_ep


# ====== RUN ON FULL DATASET ======
gt_multi_dir = BASE/"Multi-color"/"Pix2pix"
gt_single_dir = BASE/"Single-color"/"Pix2pix"
all_sets = [
    {f.stem.replace("_real_B","") for f in gt_multi_dir.glob("*_real_B.png")},
    {f.stem.replace("_real_B","") for f in gt_single_dir.glob("*_real_B.png")},
    {d.name for d in (BASE/"Multi-color"/"LBBDM").iterdir() if d.is_dir()},
    {d.name for d in (BASE/"Single-color"/"LBBDM").iterdir() if d.is_dir()},
    {d.name for d in (BASE/"Multi-color"/"PBBDM").iterdir() if d.is_dir()},
    {d.name for d in (BASE/"Single-color"/"PBBDM").iterdir() if d.is_dir()},
]
common = sorted(set.intersection(*all_sets))
print(f"Common: {len(common)}")

metrics = ["junctions", "endpoints", "length"]
results = {src: {"single":[], "multi":[]} for src in ["GT","LBBDM","PBBDM"]}

for i, sid in enumerate(common):
    if (i+1)%50==0: print(f"  {i+1}/{len(common)}...")
    gt_m = load(gt_multi_dir/f"{sid}_real_B.png")
    vm, by = make_valid_mask(gt_m)

    paths = {
        "GT": (gt_multi_dir/f"{sid}_real_B.png", gt_single_dir/f"{sid}_real_B.png"),
        "LBBDM": (BASE/"Multi-color"/"LBBDM"/sid/"output_0.png", BASE/"Single-color"/"LBBDM"/sid/"output_0.png"),
        "PBBDM": (BASE/"Multi-color"/"PBBDM"/sid/"output_0.png", BASE/"Single-color"/"PBBDM"/sid/"output_0.png"),
    }

    for src, (mp, sp) in paths.items():
        # --- Multi-color analysis ---
        img_m = load(mp)
        bottom, middle, top = separate_layers(img_m)
        layer_masks = [bottom, middle, top]
        layer_skels = [skeletonize(m) for m in layer_masks]

        # Junctions: per-layer independent
        m_j = 0
        for sk in layer_skels:
            _, jn = find_features(sk)
            m_j += count_in_mask(jn, vm)

        # Length: per-layer independent
        m_l = sum(int(np.sum(sk & vm)) for sk in layer_skels)

        # Endpoints: per-layer independent with boundary removal
        m_e = count_true_endpoints(layer_skels, layer_masks, vm)

        results[src]["multi"].append({"junctions": m_j, "endpoints": m_e, "length": m_l})

        # --- Single-color analysis ---
        img_s = load(sp)
        mask_s = get_vessel_mask(img_s)
        skel_s = skeletonize(mask_s)
        ep_s, jn_s = find_features(skel_s)

        results[src]["single"].append({
            "junctions": count_in_mask(jn_s, vm),
            "endpoints": count_in_mask(ep_s, vm),
            "length": int(np.sum(skel_s & vm)),
        })

n = len(common)
print(f"Processed: {n}")

# Stats
def pval_stars(p):
    if p<0.001: return "***"
    elif p<0.01: return "**"
    elif p<0.05: return "*"
    return "n.s."

output = {}
for src in ["GT","LBBDM","PBBDM"]:
    s, m = results[src]["single"], results[src]["multi"]
    d = {}
    for key in metrics:
        sv = [x[key] for x in s]; mv = [x[key] for x in m]
        _, p = stats.ttest_rel(sv, mv)
        _, wp = stats.wilcoxon(sv, mv)
        d[key] = {"sv": sv, "mv": mv, "p": p, "wp": wp,
                  "single_mean": float(np.mean(sv)), "single_sem": float(np.std(sv)/np.sqrt(n)),
                  "multi_mean": float(np.mean(mv)), "multi_sem": float(np.std(mv)/np.sqrt(n))}
    output[src] = d

print("\n--- Results (Fixed Endpoints) ---")
for src in ["GT","LBBDM","PBBDM"]:
    print(f"\n{src}:")
    for key in metrics:
        d = output[src][key]
        sm, mm = d["single_mean"], d["multi_mean"]
        ratio = mm/sm if sm > 0 else 0
        print(f"  {key:<11s} S:{sm:>7.1f}  M:{mm:>7.1f}  ratio:{ratio:.2f}  p={d['p']:.2e} {pval_stars(d['p'])}")

# Save
with open(OUT / "metric_results_fixed.json", "w") as f:
    json.dump({src: {key: {"single_vals": d[key]["sv"], "multi_vals": d[key]["mv"],
                           "single_mean": d[key]["single_mean"], "single_sem": d[key]["single_sem"],
                           "multi_mean": d[key]["multi_mean"], "multi_sem": d[key]["multi_sem"],
                           "p_ttest": float(d[key]["p"]), "p_wilcoxon": float(d[key]["wp"])}
                     for key in metrics} for src, d in output.items()}, f)
print("\nSaved: metric_results_fixed.json")
