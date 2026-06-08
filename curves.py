"""
Retrieve train/val learning curves from NTv3 fine-tuning SLURM logs and plot them.

Per-epoch log lines only ever contained `train_loss` (MSE) and `val_r` (Pearson),
so those are the two curves we can reconstruct. Per-epoch val-MSE was never logged
by the runs that have already executed.

Usage:  python curves.py
Writes: curves_<label>.csv  (one per run)  and  curves.png
"""
from __future__ import annotations
import glob, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))

# epoch lines look like:
#   epoch  17 * | train_loss: 0.409861 | val_r: 0.8908 (best: 0.8908) | elapsed: 2895s
_EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s+\*?\s*\|\s*train_loss:\s*([\d.eE+-]+)\s*\|\s*val_r:\s*([\d.eE+-]+)"
)

# curated set of meaningful runs (jobid -> human label). Others are partial/cancelled.
RUNS = {
    "ntv3_ft_leaf_2453311.out":  "leaf  no-ES (full)",
    "ntv3_ft_proto_2453312.out": "proto no-ES (full)",
    "ntv3_ft_leaf_2459730.out":  "leaf  early-stop (stage2 resume)",
    "ntv3_ft_proto_2459731.out": "proto early-stop (stage2 resume)",
}


def parse_log(path: str) -> pd.DataFrame:
    rows, stage = [], 1
    resumed = False
    with open(path) as f:
        for line in f:
            if "stage 1: SKIPPED" in line:
                resumed = True
            if "--- stage 2:" in line:
                stage = 2
            m = _EPOCH_RE.search(line)
            if m:
                rows.append({
                    "disp_epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_pearson": float(m.group(3)),
                    "stage": stage,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "step", range(len(df)))   # sequential x across stage1+stage2
        df.attrs["resumed"] = resumed
    return df


def main():
    parsed = {}
    for fname, label in RUNS.items():
        path = os.path.join(_DIR, fname)
        if not os.path.exists(path):
            print(f"skip (missing): {fname}")
            continue
        df = parse_log(path)
        if df.empty:
            print(f"skip (no epochs yet): {fname}")
            continue
        csv = os.path.join(_DIR, f"curves_{label.split()[0]}_{'ES' if 'early' in label else 'noES'}.csv")
        df.to_csv(csv, index=False)
        parsed[label] = df
        s2 = df[df.stage == 2]
        print(f"{label:34s} | {len(df):3d} epochs | best val_r={df.val_pearson.max():.4f} "
              f"| final train_loss={df.train_loss.iloc[-1]:.4f} -> {csv}")

    if not parsed:
        print("no runs to plot yet")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {}
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, df) in enumerate(parsed.items()):
        c = cyc[i % len(cyc)]
        colors[label] = c
        ax1.plot(df.step, df.train_loss, label=label, color=c)
        ax2.plot(df.step, df.val_pearson, label=label, color=c)
        # mark stage1->stage2 transition for full runs
        s2start = df[df.stage == 2]["step"]
        if not df.attrs.get("resumed") and len(s2start):
            x = s2start.iloc[0]
            ax1.axvline(x, color=c, ls=":", alpha=0.4)
            ax2.axvline(x, color=c, ls=":", alpha=0.4)

    ax1.set_title("Train loss (MSE)"); ax1.set_xlabel("epoch index"); ax1.set_ylabel("MSE")
    ax2.set_title("Validation Pearson r"); ax2.set_xlabel("epoch index"); ax2.set_ylabel("val_r")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)
    fig.suptitle("NTv3 Jores21 combined — learning curves (dotted = stage1->stage2)")
    fig.tight_layout()
    out = os.path.join(_DIR, "curves.png")
    fig.savefig(out, dpi=130)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
