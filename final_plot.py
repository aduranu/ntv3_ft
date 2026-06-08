"""Final comparison: tuned stage-2 recipe vs the old flat recipe, both tissues."""
import os, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1 = {"leaf": 0.842555, "proto": 0.841284}   # frozen-backbone test baseline
TUNED = {"leaf": "metrics_leaf_combined_lr4e-4_bb1.0.csv",
         "proto": "metrics_proto_combined_lr6e-4_bb1.0.csv"}
OLD   = {"leaf": "curves_leaf_noES.csv", "proto": "curves_proto_noES.csv"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
for ax, tissue in zip(axes, ["leaf", "proto"]):
    t = pd.read_csv(os.path.join(_DIR, TUNED[tissue]))
    t = t[t.stage == 2].reset_index(drop=True)
    ax.plot(range(len(t)), t.val_pearson, "-o", ms=3, color="tab:green",
            label="tuned stage-2 (schedule, lr~4e-4, RC, wd)")

    o = pd.read_csv(os.path.join(_DIR, OLD[tissue]))
    o2 = o[o.stage == 2].reset_index(drop=True)
    ax.plot(range(len(o2)), o2.val_pearson, "-o", ms=3, color="tab:orange",
            label="old flat stage-2 (const 1e-5)")

    ax.axhline(STAGE1[tissue], color="gray", ls="--",
               label=f"stage-1 frozen (val_r start)")
    ax.set_title(f"{tissue}  —  val Pearson vs stage-2 epoch")
    ax.set_xlabel("stage-2 epoch"); ax.set_ylabel("val_r")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

fig.suptitle("NTv3 Jores21 combined — tuned vs old stage-2 fine-tuning")
fig.tight_layout()
out = os.path.join(_DIR, "stage2_tuned_vs_old.png")
fig.savefig(out, dpi=130)
print("saved", out)
