"""
Per-layer concordance v2:
  - Area/Length: R² scatter (as before)
  - JN/EP: Bland-Altman + MAE + % within ±N
  - Individual panel PNGs for PPT
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
model_colors = {
    'PBBDM': '#2980b9', 'LBBDM': '#8e44ad',
    'WGANGP': '#e67e22', 'Pix2pix': '#27ae60', 'LSGAN': '#c0392b'
}
layer_colors = {'R': '#CC4444', 'B': '#4477CC'}
layer_markers = {'R': 'o', 'B': 's'}
layer_labels = {'R': 'Bottom', 'B': 'Top'}

# ============================================================
# 1. Area/Length: R² scatter plots (same as before)
# ============================================================
is_first_area = True
for mname in models:
    for met, mtitle in [('area', 'Vessel Area'), ('length', 'Vessel Length')]:
        fig, ax = plt.subplots(figsize=(4, 4))
        # Plot Top (B) first so it appears first in legend
        for layer in ['B', 'R']:
            gt_v = np.array(data['GT'][layer][met])
            mod_v = np.array(data[mname][layer][met])
            ax.scatter(gt_v, mod_v, c=layer_colors[layer], marker=layer_markers[layer],
                      s=20, alpha=0.35, edgecolors='none', zorder=2, label=layer_labels[layer])

        all_vals = np.concatenate([data['GT']['R'][met], data['GT']['B'][met],
                                   data[mname]['R'][met], data[mname]['B'][met]])
        maxval = np.max(all_vals) * 1.08 if len(all_vals) > 0 else 1
        ax.plot([0, maxval], [0, maxval], color='#888888', linewidth=1, linestyle='--', zorder=1)

        ax.set_xlim(0, maxval); ax.set_ylim(0, maxval)
        ax.set_aspect('equal')
        ax.set_xlabel('', fontsize=1)
        ax.set_ylabel('', fontsize=1)
        import matplotlib.ticker as mticker
        def sci_fmt(x, pos):
            if x == 0: return '0'
            exp = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
            coeff = x / 10**exp
            if coeff == int(coeff):
                return f'{int(coeff)}E{exp}'
            else:
                return f'{coeff:.1f}E{exp}'
        if met in ('area', 'length'):
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(sci_fmt))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_fmt))
            if met == 'area':
                # Fewer ticks for area to avoid overlap
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
                # Sync x to y
                ax.set_xlim(0, maxval); ax.set_ylim(0, maxval)
                yticks = [t for t in ax.get_yticks() if 0 <= t <= maxval]
                ax.set_xticks(yticks)
                ax.set_xlim(0, maxval); ax.set_ylim(0, maxval)
            ax.tick_params(axis='x', labelsize=12, rotation=45)
            ax.tick_params(axis='y', labelsize=12)
        else:
            ax.tick_params(labelsize=16)
        # Legend only on the very first area plot (PBBDM area)
        if met == 'area' and is_first_area:
            ax.legend(fontsize=14, loc='lower right')
            is_first_area = False
        plt.tight_layout()
        plt.savefig(PANEL / f'scatter_{mname}_{met}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

print('Saved Area/Length scatter plots')

# ============================================================
# 2. JN/EP: Bland-Altman plots
# ============================================================
def bland_altman(ax, gt_v, mod_v, color, marker, label, layer_name):
    """Bland-Altman: x=mean, y=difference (model-GT)"""
    mean_vals = (gt_v + mod_v) / 2
    diff_vals = mod_v - gt_v

    ax.scatter(mean_vals, diff_vals, c=color, marker=marker,
              s=20, alpha=0.35, edgecolors='none', zorder=2, label=label)

    # Stats
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    mae = np.mean(np.abs(diff_vals))

    # % within thresholds
    pct_1 = np.mean(np.abs(diff_vals) <= 1) * 100
    pct_2 = np.mean(np.abs(diff_vals) <= 2) * 100
    pct_3 = np.mean(np.abs(diff_vals) <= 3) * 100

    return {
        'mean_diff': mean_diff, 'std_diff': std_diff,
        'loa_upper': loa_upper, 'loa_lower': loa_lower,
        'mae': mae, 'pct_1': pct_1, 'pct_2': pct_2, 'pct_3': pct_3
    }

# Individual Bland-Altman + stats for each model × JN/EP
all_stats = {}
is_first_jn = True
for mname in models:
    all_stats[mname] = {}
    for met, mtitle in [('junctions', 'Junctions'), ('endpoints', 'Endpoints')]:
        fig, ax = plt.subplots(figsize=(4.5, 4))

        stats_per_layer = {}
        # Plot Top (B) first so it appears first in legend
        for layer in ['B', 'R']:
            gt_v = np.array(data['GT'][layer][met], dtype=float)
            mod_v = np.array(data[mname][layer][met], dtype=float)
            s = bland_altman(ax, gt_v, mod_v, layer_colors[layer],
                           layer_markers[layer], layer_labels[layer], layer)
            stats_per_layer[layer] = s

        # Combined stats for Bland-Altman lines
        gt_all = np.concatenate([data['GT']['R'][met], data['GT']['B'][met]]).astype(float)
        mod_all = np.concatenate([data[mname]['R'][met], data[mname]['B'][met]]).astype(float)
        mean_all = (gt_all + mod_all) / 2
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

        # Fill LoA band
        ax.fill_between([xmin-5, xmax+5], loa_l, loa_u, alpha=0.08, color='#888888', zorder=0)

        ax.set_xlabel('', fontsize=1)
        ax.set_ylabel('', fontsize=1)
        ax.tick_params(labelsize=14)
        # Legend only on the very first junctions plot (PBBDM junctions)
        if met == 'junctions' and is_first_jn:
            ax.legend(fontsize=14, loc='lower left')
            is_first_jn = False
        plt.tight_layout()
        plt.savefig(PANEL / f'ba_{mname}_{met}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        all_stats[mname][met] = stats_per_layer

print('Saved JN/EP Bland-Altman plots')

# ============================================================
# 3. Summary bar chart: MAE comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax_idx, (met, mtitle) in enumerate([('junctions', 'Junctions'), ('endpoints', 'Endpoints')]):
    ax = axes[ax_idx]
    x = np.arange(len(models))
    width = 0.35

    r_maes = [all_stats[m][met]['R']['mae'] for m in models]
    b_maes = [all_stats[m][met]['B']['mae'] for m in models]

    bars_r = ax.bar(x - width/2, r_maes, width, color='#CC4444', alpha=0.75, label='Bottom')
    bars_b = ax.bar(x + width/2, b_maes, width, color='#4477CC', alpha=0.75, label='Top')

    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=18)
    ax.set_ylabel('MAE (count)', fontsize=20)
    ax.set_title(f'{mtitle} — Mean Absolute Error', fontsize=22, fontweight='bold')
    ax.legend(fontsize=16)
    ax.tick_params(labelsize=16)
    ax.set_ylim(0, max(max(r_maes), max(b_maes)) * 1.3)

plt.tight_layout()
plt.savefig(PANEL / 'summary_mae_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# 4. Summary bar chart: % within ±3
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax_idx, (met, mtitle) in enumerate([('junctions', 'Junctions'), ('endpoints', 'Endpoints')]):
    ax = axes[ax_idx]
    x = np.arange(len(models))
    width = 0.35

    r_pct = [all_stats[m][met]['R']['pct_3'] for m in models]
    b_pct = [all_stats[m][met]['B']['pct_3'] for m in models]

    bars_r = ax.bar(x - width/2, r_pct, width, color='#CC4444', alpha=0.75, label='Bottom')
    bars_b = ax.bar(x + width/2, b_pct, width, color='#4477CC', alpha=0.75, label='Top')

    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=18)
    ax.set_ylabel('% within ±3', fontsize=20)
    ax.set_title(f'{mtitle} — Agreement within ±3', fontsize=22, fontweight='bold')
    ax.legend(fontsize=16)
    ax.tick_params(labelsize=16)
    ax.set_ylim(0, 105)
    ax.axhline(80, color='#888888', linewidth=0.5, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(PANEL / 'summary_pct3_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# 5. Print summary table
# ============================================================
print('\n' + '='*90)
print(f'{"":>10} | {"Junctions":^40} | {"Endpoints":^40}')
print(f'{"":>10} | {"R MAE":>6} {"R ±3%":>6} {"B MAE":>6} {"B ±3%":>6} | {"R MAE":>6} {"R ±3%":>6} {"B MAE":>6} {"B ±3%":>6}')
print('-'*90)
for m in models:
    jr = all_stats[m]['junctions']['R']
    jb = all_stats[m]['junctions']['B']
    er = all_stats[m]['endpoints']['R']
    eb = all_stats[m]['endpoints']['B']
    print(f'{m:>10} | {jr["mae"]:>5.1f} {jr["pct_3"]:>5.0f}% {jb["mae"]:>5.1f} {jb["pct_3"]:>5.0f}% | '
          f'{er["mae"]:>5.1f} {er["pct_3"]:>5.0f}% {eb["mae"]:>5.1f} {eb["pct_3"]:>5.0f}%')
print('='*90)

# Also print Area/Length R² for reference
print('\nArea/Length R²:')
print(f'{"":>10} | {"Area R":>8} {"Area B":>8} {"Length R":>8} {"Length B":>8}')
for m in models:
    vals = []
    for met in ['area', 'length']:
        for layer in ['R', 'B']:
            gt_v = np.array(data['GT'][layer][met])
            mod_v = np.array(data[m][layer][met])
            valid = gt_v > 0
            if np.sum(valid) > 2:
                _, _, r_val, _, _ = stats.linregress(gt_v[valid], mod_v[valid])
                vals.append(r_val**2)
            else:
                vals.append(0)
    print(f'{m:>10} | {vals[0]:>7.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} {vals[3]:>8.3f}')

print('\nDone!')
