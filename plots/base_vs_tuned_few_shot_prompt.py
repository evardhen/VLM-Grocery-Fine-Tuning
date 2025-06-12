import matplotlib.pyplot as plt
import numpy as np

# --- Data ------------------------------------------------------------
metrics = ['BERT-Cat', 'BERT-Fine']
models  = ['Qwen2-7B', 'Pixtral-12B', 'Qwen2.5-7B']

# Matrix shape: metrics x models
base_vals = np.array([
    [36.15, 36.33, 32.33],   # BERT-Cat
    [32.28, 37.82, 29.69],   # BERT-Fine
])
sft_vals  = np.array([
    [36.39, 38.26, 34.47],
    [33.55, 37.62, 29.89],
])

# --- Style -----------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": .6,
    "axes.edgecolor": "#444"
})
LIGHT_BLUE, DARK_BLUE = "#8ecae6", "#023047"

bar_width = 0.3
group_gap = 1.2                        # <--- bigger gap between model groups
xpos      = np.arange(len(models)) * group_gap
offset    = bar_width / 2

paths = []

for i, metric in enumerate(metrics):
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=100)

    # bars
    bars_base = ax.bar(xpos - offset, base_vals[i], bar_width,
                       label="Simple", color=LIGHT_BLUE,
                       edgecolor="#015b8a", linewidth=.7, zorder=3)
    bars_sft  = ax.bar(xpos + offset, sft_vals[i], bar_width,
                       label="Structured + Few shot", color=DARK_BLUE,
                       edgecolor="#012941", linewidth=.7, zorder=3)

    # labels above bars
    for bars in (bars_base, bars_sft):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    # axes cosmetics
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(models)
    ax.yaxis.grid(True, linestyle=":", linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)

    # legend below the plot
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=True, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    outfile = f"plots/{metric.replace(' ','_').lower()}_base_vs_tuned_few_shot.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    paths.append(outfile)

