"""
Plot train vs val loss (and val Pearson) for the current experiment.

Reads metrics_<tissue>_<mode>.csv produced by finetune.py (which logs both
stage 1 and stage 2), using tissue/mode from config.json. Writes loss_curves.png.

Usage: python plot_curves.py
"""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_dir = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(_dir, "config.json")))
tissue, mode = cfg["tissue"], cfg["mode"]
csv = os.path.join(_dir, f"metrics_{tissue}_{mode}.csv")

df = pd.read_csv(csv)
stages = [s for s in (1, 2) if (df.stage == s).any()]
names = {1: "stage 1 (probing)", 2: "stage 2 (full finetune)"}

fig, axes = plt.subplots(1, len(stages), figsize=(7 * len(stages), 5), squeeze=False)
for ax, stg in zip(axes[0], stages):
    d = df[df.stage == stg].reset_index(drop=True)
    x = range(len(d))
    ax.plot(x, d.train_loss, "-o", ms=3, label="train loss")
    ax.plot(x, d.val_loss, "-o", ms=3, label="val loss")
    ax.set_title(f"{tissue} {mode} — {names[stg]}")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
    ax.grid(alpha=0.3); ax.legend()

fig.suptitle(f"NTv3 Jores21 {tissue} {mode} — train vs val loss")
fig.tight_layout()
out = os.path.join(_dir, "loss_curves.png")
fig.savefig(out, dpi=130)
print("saved", out)
