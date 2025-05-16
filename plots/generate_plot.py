import matplotlib.pyplot as plt
import numpy as np

# --- Data ------------------------------------------------------------
metrics = ['BERT-Cat', 'BERT-Fine', 'Count EM']
base_vals = [39.95, 34.54, 78.57]
sft_vals  = [81.78, 79.36, 87.98]

# --- Style -----------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "axes.linewidth": 1.0})

# --- Geometry --------------------------------------------------------
bar_width   = 0.28          # narrower bars
bar_offset  = 0.15          # half distance between blue ↔ red centres
group_gap   = 1.2           # spacing between successive metrics

x_centres = np.arange(len(metrics)) * group_gap

fig, ax = plt.subplots(figsize=(7.2, 3.5))

for spine in ax.spines.values():              # 'left', 'right', 'top', 'bottom'
    spine.set_linewidth(0.4)

plt.rcParams['axes.linewidth'] = 0.4 

# Horizontal dotted grid (behind opaque bars, so only visible in gaps)
dot_len  = 2           # length of each drawn segment  [points]
gap_len  = 5           # gap between dots             [points]
ax.set_axisbelow(True)
ax.yaxis.grid(True, lw=0.4, color='gray', alpha=0.6, zorder=0, linestyle=(0, (dot_len, gap_len)))

# Vertical separators between groups (tall but hidden under bars)
for i in range(1, len(metrics)):
    ax.axvline(x_centres[i] - group_gap/2, lw=0.4,
               color='gray', alpha=0.6, zorder=0, linestyle=(0, (dot_len, gap_len)))

# Bars ----------------------------------------------------------------
bars1 = ax.bar(x_centres - bar_offset, base_vals, width=bar_width,
               label='Base', color='#b2b2ff',
               edgecolor='#0011ff', linewidth=0.7, zorder=3)

bars2 = ax.bar(x_centres + bar_offset, sft_vals,  width=bar_width,
               label='SFT', color='#ffb2b2',
               edgecolor='#ff3939', linewidth=0.7, zorder=3)

# Value labels --------------------------------------------------------
for bar in bars1:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 1,
            f'{y:.1f}', ha='center', va='bottom',
            color='blue', fontsize=10)

for bar in bars2:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 1,
            f'{y:.1f}', ha='center', va='bottom',
            color='red', fontsize=10)

# Axis / legend -------------------------------------------------------
ax.set_ylabel('Score (%)')
ax.set_ylim(30, 100)
ax.set_xticks(x_centres)
ax.set_xticklabels(metrics, fontsize=11)

ax.legend(loc='upper center',
          bbox_to_anchor=(0.5, -0.22),
          ncol=2,
          frameon=True,
          fancybox=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("plots/base_vs_finetuned.png", dpi=300, bbox_inches='tight')
# plt.show()
