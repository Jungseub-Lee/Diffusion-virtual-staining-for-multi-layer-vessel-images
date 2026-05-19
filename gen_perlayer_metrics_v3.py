"""
Per-layer concordance v3:
  - Area/Length: R² scatter (NO text annotations inside plots)
  - JN/EP: Bland-Altman (NO text annotations inside plots)
  - Unified summary table with ALL metrics
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from scipy import stats

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
PANEL = OUT / 'scatter_panels_v2'
PANEL.mkdir(exist_ok=True)

with open(OUT / 'perlayer_rgb_scatter_data.json') as f:
    data = json.load(f)

N = len(data['GT']['R']['area'])
print(f'N = {N} samples')

models = ['PBBDM', 'LBBDM', 'WGANGP', 'Pix2pix', 'LSGAN']
layer_colors = {'R': '#CC4444', 'B': '#4477CC'}
layer_markers = {'R': 'o', 'B': 's'}
layer_labels = {'R': 'Bottom (R>B)', 'B': 'Top (B>R)'}

# Collect all stats
all_r2 = {}  # model → {area_R, area_B, length_R, length_B}
all_ba = {}  # model → {junctions: {R:{mae,pct3}, B:{...}}, endpoints: ...}

# ============================================================
# 1. Area/Length: R² scatter (CLEAN — no text inside)
# ============================================================
for mname in models:
    all_r2[mname] = {}
    for met, mtitle in [('area', 'Vessel Area'), ('length', 'Vessel Length')]:
        fig, ax = plt.subplots(figsize=(4, 4))
        for layer in ['R', 'B']:
            gt_v = np.array(data['GT'][layer][met])
            mod_v = np.array(data[mname][layer][met])
            ax.scatter(gt_v, mod_v, c=layer_colors[layer], marker=layer_markers[layer],
                      s=20, alpha=0.35, edgecolors='none', zorder=2, label=layer_labels[layer])
            valid = gt_v > 0
            if np.sum(valid) > 2:
                _, _, r_val, _, _ = stats.linregress(gt_v[valid], mod_v[valid])
                all_r2[mname][f'{met}_{layer}'] = r_val**2

        all_vals = np.concatenate([data['GT']['R'][met], data['GT']['B'][met],
                                   data[mname]['R'][met], data[mname]['B'][met]])
        maxval = np.max(all_vals) * 1.08 if len(all_vals) > 0 else 1
        ax.plot([0, maxval], [0, maxval], color='#888888', linewidth=1, linestyle='--', zorder=1)
        ax.set_xlim(0, maxval); ax.set_ylim(0, maxval)
        ax.set_aspect('equal')
        ax.set_xlabel('Ground Truth', fontsize=10)
        ax.set_ylabel(f'{mname}', fontsize=10)
        ax.set_title(f'{mname} — {mtitle}', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        if met == 'area':
            ax.legend(fontsize=7, loc='lower right')
        plt.tight_layout()
        plt.savefig(PANEL / f'scatter_{mname}_{met}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

print('Saved Area/Length scatter plots (clean)')

# ============================================================
# 2. JN/EP: Bland-Altman (CLEAN — no text inside)
# ============================================================
for mname in models:
    all_ba[mname] = {}
    for met, mtitle in [('junctions', 'Junctions'), ('endpoints', 'Endpoints')]:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        all_ba[mname][met] = {}

        for layer in ['R', 'B']:
            gt_v = np.array(data['GT'][layer][met], dtype=float)
            mod_v = np.array(data[mname][layer][met], dtype=float)
            mean_vals = (gt_v + mod_v) / 2
            diff_vals = mod_v - gt_v

            ax.scatter(mean_vals, diff_vals, c=layer_colors[layer], marker=layer_markers[layer],
                      s=20, alpha=0.35, edgecolors='none', zorder=2, label=layer_labels[layer])

            mae = np.mean(np.abs(diff_vals))
            pct_3 = np.mean(np.abs(diff_vals) <= 3) * 100
            mean_diff = np.mean(diff_vals)
            all_ba[mname][met][layer] = {'mae': mae, 'pct_3': pct_3, 'bias': mean_diff}

        # Combined LoA lines
        gt_all = np.concatenate([data['GT']['R'][met], data['GT']['B'][met]]).astype(float)
        mod_all = np.concatenate([data[mname]['R'][met], data[mname]['B'][met]]).astype(float)
        diff_all = mod_all - gt_all
        mean_d = np.mean(diff_all)
        std_d = np.std(diff_all, ddof=1)
        loa_u = mean_d + 1.96 * std_d
        loa_l = mean_d - 1.96 * std_d

        xmin, xmax = ax.get_xlim()
        ax.axhline(mean_d, color='#333333', linewidth=1.5, linestyle='-', zorder=1)
        ax.axhline(loa_u, color='#999999', linewidth=1, linestyle='--', zorder=1)
        ax.axhline(loa_l, color='#999999', linewidth=1, linestyle='--', zorder=1)
        ax.axhline(0, color='#CCCCCC', linewidth=0.5, linestyle='-', zorder=0)
        ax.fill_between([xmin-5, xmax+5], loa_l, loa_u, alpha=0.08, color='#888888', zorder=0)

        ax.set_xlabel(f'Mean of GT and {mname}', fontsize=9)
        ax.set_ylabel(f'Difference ({mname} - GT)', fontsize=9)
        ax.set_title(f'{mname} — {mtitle}', fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)
        if met == 'junctions':
            ax.legend(fontsize=7, loc='lower left')
        plt.tight_layout()
        plt.savefig(PANEL / f'ba_{mname}_{met}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

print('Saved JN/EP Bland-Altman plots (clean)')

# ============================================================
# 3. Summary bar charts (same as before)
# ============================================================
for chart_type, ylabel, title_suffix, get_val, ylim_factor in [
    ('mae', 'MAE (count)', 'Mean Absolute Error', lambda s: s['mae'], 1.3),
    ('pct3', '% within ±3', 'Agreement within ±3', lambda s: s['pct_3'], None)]:

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_idx, (met, mtitle) in enumerate([('junctions', 'Junctions'), ('endpoints', 'Endpoints')]):
        ax = axes[ax_idx]
        x = np.arange(len(models)); width = 0.35
        r_vals = [get_val(all_ba[m][met]['R']) for m in models]
        b_vals = [get_val(all_ba[m][met]['B']) for m in models]
        bars_r = ax.bar(x - width/2, r_vals, width, color='#CC4444', alpha=0.75, label='Bottom (R>B)')
        bars_b = ax.bar(x + width/2, b_vals, width, color='#4477CC', alpha=0.75, label='Top (B>R)')
        fmt = '{:.1f}' if chart_type == 'mae' else '{:.0f}%'
        for bar in list(bars_r) + list(bars_b):
            label_text = f'{bar.get_height():.1f}' if chart_type == 'mae' else f'{bar.get_height():.0f}%'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if chart_type=='mae' else 0.5),
                   label_text, ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{mtitle} — {title_suffix}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        if ylim_factor:
            ax.set_ylim(0, max(max(r_vals), max(b_vals)) * ylim_factor)
        else:
            ax.set_ylim(0, 105)
            ax.axhline(80, color='#888888', linewidth=0.5, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PANEL / f'summary_{chart_type}_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

print('Saved summary bar charts')

# ============================================================
# 4. Print unified table
# ============================================================
print('\n' + '='*110)
print(f'{"Model":>10} | {"Area R²":>8} {"Area R²":>8} | {"Len R²":>8} {"Len R²":>8} | '
      f'{"JN MAE":>7} {"JN MAE":>7} {"JN±3%":>6} {"JN±3%":>6} | '
      f'{"EP MAE":>7} {"EP MAE":>7} {"EP±3%":>6} {"EP±3%":>6}')
print(f'{"":>10} | {"Bot":>8} {"Top":>8} | {"Bot":>8} {"Top":>8} | '
      f'{"Bot":>7} {"Top":>7} {"Bot":>6} {"Top":>6} | '
      f'{"Bot":>7} {"Top":>7} {"Bot":>6} {"Top":>6}')
print('-'*110)
for m in models:
    ar = all_r2[m].get('area_R', 0); ab = all_r2[m].get('area_B', 0)
    lr = all_r2[m].get('length_R', 0); lb = all_r2[m].get('length_B', 0)
    jr = all_ba[m]['junctions']['R']; jb = all_ba[m]['junctions']['B']
    er = all_ba[m]['endpoints']['R']; eb = all_ba[m]['endpoints']['B']
    print(f'{m:>10} | {ar:>7.3f} {ab:>8.3f} | {lr:>7.3f} {lb:>8.3f} | '
          f'{jr["mae"]:>6.1f} {jb["mae"]:>7.1f} {jr["pct_3"]:>5.0f}% {jb["pct_3"]:>5.0f}% | '
          f'{er["mae"]:>6.1f} {eb["mae"]:>7.1f} {er["pct_3"]:>5.0f}% {eb["pct_3"]:>5.0f}%')
print('='*110)
print('\nDone!')
