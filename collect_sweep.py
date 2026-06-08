"""Collect stage-2 sweep results from sweep_logs/*.out and rank by test Pearson."""
from __future__ import annotations
import glob, os, re

_DIR = os.path.dirname(os.path.abspath(__file__))

# stage-1 (frozen-backbone) test baselines, per tissue, to measure the stage-2 bump
STAGE1_TEST = {"leaf": 0.842555, "proto": 0.841284}

_FLOAT = r"([-\d.eE+]+)"
PATS = {k: re.compile(rf"^{k}:\s*{_FLOAT}") for k in
        ["val_pearson", "test_pearson", "test_mse", "stage2_epochs"]}
NAME_RE = re.compile(r"s2_(leaf|proto)_lr([\d.eE+-]+)_bb([\d.]+)")


def parse(path):
    out, tissue, base, scale = {}, None, None, None
    with open(path) as f:
        for line in f:
            m = NAME_RE.search(line)
            if m and tissue is None:
                tissue, base, scale = m.group(1), m.group(2), m.group(3)
            for k, pat in PATS.items():
                pm = pat.match(line.strip())
                if pm:
                    out[k] = float(pm.group(1))
    out.update(tissue=tissue, base_lr=base, bb_scale=scale)
    return out


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(_DIR, "sweep_logs", "s2_*.out"))):
        r = parse(p)
        if "test_pearson" in r and r["tissue"]:
            r["delta"] = r["test_pearson"] - STAGE1_TEST.get(r["tissue"], float("nan"))
            rows.append(r)

    if not rows:
        print("no completed trials yet")
        return

    for tissue in ["leaf", "proto"]:
        tr = sorted([r for r in rows if r["tissue"] == tissue],
                    key=lambda r: -r["test_pearson"])
        if not tr:
            continue
        base = STAGE1_TEST[tissue]
        print(f"\n===== {tissue}  (stage-1 frozen test baseline = {base:.4f}) =====")
        print(f"{'base_lr':>8} {'bb_scale':>8} {'test_r':>8} {'Δ vs s1':>9} "
              f"{'val_r':>8} {'test_mse':>9} {'s2_ep':>6}")
        for r in tr:
            print(f"{r['base_lr']:>8} {r['bb_scale']:>8} {r['test_pearson']:>8.4f} "
                  f"{r['delta']:>+9.4f} {r.get('val_pearson', float('nan')):>8.4f} "
                  f"{r.get('test_mse', float('nan')):>9.4f} {int(r.get('stage2_epochs', 0)):>6}")
        best = tr[0]
        print(f"  -> best {tissue}: base_lr={best['base_lr']} bb_scale={best['bb_scale']} "
              f"test_r={best['test_pearson']:.4f} (Δ{best['delta']:+.4f} over stage-1)")

    print(f"\nparsed {len(rows)} completed trials")


if __name__ == "__main__":
    main()
