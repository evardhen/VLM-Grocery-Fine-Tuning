from wandb import Api
import pandas as pd
import sys
import math

api  = Api()
# projects = api.projects("evardhen-tu-berlin")  

# print("Projects in my-org:")
# for proj in projects:
#     print("  •", proj.name)
# sys.exit(0)
runs = api.runs("evardhen-tu-berlin/llamafactory")

rows  = []
for run in runs:
    # 1) model name ---------------------------------------------------------
    model = run.config.get("model_name", run.name)

    # 2) fine-tuning wall-clock time (h) ------------------------------------
    ft_hours = round(run.summary.get("_runtime", 0) / 3600, 2)

    # 3) peak GPU VRAM (MB) -------------------------------------------------
    #    Pull the whole "events" stream ONCE; it’s already sampled.
    events = run.history(stream="events", samples=10000, pandas=True)
    if events.empty:
        max_vram_mb = None          # system stats were never recorded
    else:
        # columns look like  gpu.0.memoryAllocatedBytes  (bytes)
        gpu_cols     = [c for c in events.columns
                        if "memoryAllocatedBytes" in c]
        max_bytes    = events[gpu_cols].max().max()
        max_vram_mb  = int(math.ceil(max_bytes / 1024**2))

    rows.append({
        "Model":               model,
        "FT Time (hr)":        ft_hours,
        "Max FT VRAM (MB)":    max_vram_mb,
    })

df = pd.DataFrame(rows).sort_values("Model")
print(df.to_latex(index=False))