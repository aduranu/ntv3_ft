"""
Jores et al. 2021 plant promoter STARR-seq dataset.

Single-output regression on enrichment values for plant protoplast / leaf systems.
Provides:
  - Jores21Dataset            — pytorch Dataset, returns (dna_string, target_float)
  - make_ntv3_collate_fn       — tokenizer-based collate, returns numpy arrays for JAX
  - create_dataloaders         — train/val/test DataLoader factory
  - build_dataset              — reproduces the paper's enrichment pipeline from raw files

CLI:
  python data.py --build --output-dir ./data
"""

from __future__ import annotations

import argparse
import math as _math
import os
import urllib.request
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


SEQUENCE_LENGTH = 437  # bp — max construct length (Sb: 153+170+102+12)
PROMOTER_LENGTH = 170  # bp — raw promoter sequence (no construct building)

# full CaMV 35S enhancer (-199 to -47 relative to 35S TSS, 153bp)
ENHANCER_153 = (
    "AGATCTCTCTGCCGACAGTGGTCCCAAAGATGGACCCCCACCCACGAGGAGCATCGTGGAAAAAGAAGAC"
    "GTTCCAACCACGTCTTCAAAGCAAGTGGATTGATGTGACATCTCCACTGACGTAAGGGATGACGCACAAT"
    "CCCACTATCCTTC"
)

# species-specific 5' UTRs (from Jores et al. 2021 Supplementary Table 8)
UTR_MAP = {
    "At": "CCCGTCCGAACTCCGAACCCCAGAACAGAGCAAAGCCTCCTCGGCCTCCCTGTCCCCAGCCTTCCCCG",
    "Zm": "CCCGTCCGAACTCCGAACCCCAGAACAGAGCAAAGCCTCCTCGGCCTCCCTGTCCCCAGCCTTCCCCG",
    "Sb": "TACCACCCTCGTCTCGCTCCAATTCCCCACCGCAAATCCAGAGCCTTCCATTTCAAACACTTCGGAGCAACATCTCCCTTCTCCCCAGCCCAATCACCCGCC",
}


# ---------------------------------------------------------------------------
# String helpers (augmentations)
# ---------------------------------------------------------------------------

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement_str(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def circular_shift_str(seq: str, shift: int) -> str:
    if shift == 0:
        return seq
    return seq[-shift:] + seq[:-shift]


def pad_or_trim_str(seq: str, length: int) -> str:
    if len(seq) < length:
        return seq + "N" * (length - len(seq))
    return seq[:length]


def build_construct(promoter: str, sp: str, use_enhancer: bool, rng: np.random.Generator) -> str:
    """Concatenate [upstream 153bp] + [170bp promoter] + [UTR] + [barcode 12bp]."""
    upstream = ENHANCER_153 if use_enhancer else "".join(rng.choice(list("ACGT"), size=len(ENHANCER_153)))
    utr = UTR_MAP[sp]
    barcode = "".join(rng.choice(list("ACGT"), size=12))
    return upstream + promoter + utr + barcode


# ---------------------------------------------------------------------------
# NTv3 collate — returns numpy arrays for JAX
# ---------------------------------------------------------------------------

def make_ntv3_collate_fn(tokenizer) -> Callable:
    """Single-nucleotide tokenizer collate. Pads to multiple of 128 bp before tokenizing
    so the conv_tower's stride-128 downsampling is exact."""

    def collate_fn(batch):
        strings, targets = zip(*batch)
        max_len = max(len(s) for s in strings)
        pad_len = _math.ceil(max_len / 128) * 128
        padded = [s + "N" * (pad_len - len(s)) for s in strings]
        token_ids = tokenizer.batch_np_tokenize(padded)
        return token_ids, np.array(targets, dtype=np.float32)

    return collate_fn


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Jores21Dataset(Dataset):
    """Jores et al. 2021 plant promoter STARR-seq data with construct building.
    Returns (dna_string, target_float). Encoding is handled by collate functions."""

    _VALID_MODES = ("enhancer", "combined", "promoter_only")

    def __init__(
        self,
        data_dir: str,
        tissue: str,
        split: str = "train",
        mode: str = "enhancer",
        reverse_complement: bool = False,
        rc_prob: float = 0.5,
        random_shift: bool = False,
        shift_prob: float = 0.5,
        max_shift: int = 25,
        val_frac: float = 0.1,
        seed: int = 42,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got {mode!r}")

        self.mode = mode
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self._rng = np.random.default_rng(seed)

        if mode in ("enhancer", "promoter_only"):
            if split in ("train", "val"):
                df = pd.read_csv(os.path.join(data_dir, f"jores21_{tissue}_35SEnh_train.tsv"), sep="\t")
                n = len(df)
                indices = np.random.default_rng(seed).permutation(n)
                n_val = int(n * val_frac)
                df = df.iloc[indices[:n_val]] if split == "val" else df.iloc[indices[n_val:]]
            else:
                df = pd.read_csv(os.path.join(data_dir, f"jores21_{tissue}_35SEnh_test.tsv"), sep="\t")
            df = df.copy()
            df["_use_enh"] = True
        else:  # combined
            if split in ("train", "val"):
                df_enh = pd.read_csv(os.path.join(data_dir, f"jores21_{tissue}_35SEnh_train.tsv"), sep="\t")
                df_noenh = pd.read_csv(os.path.join(data_dir, f"jores21_{tissue}_noEnh_train.tsv"), sep="\t")
                df_enh["_use_enh"] = True
                df_noenh["_use_enh"] = False
                df = pd.concat([df_enh, df_noenh], ignore_index=True)
                n = len(df)
                indices = np.random.default_rng(seed).permutation(n)
                n_val = int(n * val_frac)
                df = df.iloc[indices[:n_val]] if split == "val" else df.iloc[indices[n_val:]]
            else:
                df_enh = pd.read_csv(os.path.join(data_dir, f"jores21_{tissue}_35SEnh_test.tsv"), sep="\t")
                df_noenh = pd.read_csv(os.path.join(data_dir, f"jores21_{tissue}_noEnh_test.tsv"), sep="\t")
                df_enh["_use_enh"] = True
                df_noenh["_use_enh"] = False
                df = pd.concat([df_enh, df_noenh], ignore_index=True)

        self.sequences = df["sequence"].tolist()
        self.targets = df["enrichment"].values.astype(np.float32)
        self.species = df["sp"].tolist()
        self._use_enh_flags = df["_use_enh"].tolist()

        print(f"Loaded {len(self.sequences):,} {split} samples (Jores21 {tissue} {mode})")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        seq = self.sequences[idx]
        target = float(self.targets[idx])

        if self.mode == "promoter_only":
            seq = pad_or_trim_str(seq, PROMOTER_LENGTH)
            if self.reverse_complement and self._rng.random() < self.rc_prob:
                seq = reverse_complement_str(seq)
            return seq, target

        construct = build_construct(seq, self.species[idx], self._use_enh_flags[idx], self._rng)

        if self.random_shift and self._rng.random() < self.shift_prob:
            shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
            construct = circular_shift_str(construct, shift)

        construct = pad_or_trim_str(construct, SEQUENCE_LENGTH)

        if self.reverse_complement and self._rng.random() < self.rc_prob:
            construct = reverse_complement_str(construct)

        return construct, target


def create_dataloaders(
    data_dir: str,
    tissue: str,
    batch_size: int,
    mode: str = "enhancer",
    reverse_complement: bool = False,
    rc_prob: float = 0.5,
    random_shift: bool = False,
    shift_prob: float = 0.5,
    max_shift: int = 25,
    collate_fn: Callable | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders. num_workers=0 for JAX (forked workers
    break JAX device handles)."""

    train_ds = Jores21Dataset(
        data_dir=data_dir, tissue=tissue, split="train", mode=mode,
        reverse_complement=reverse_complement, rc_prob=rc_prob,
        random_shift=random_shift, shift_prob=shift_prob, max_shift=max_shift,
    )
    val_ds = Jores21Dataset(data_dir=data_dir, tissue=tissue, split="val", mode=mode)
    test_ds = Jores21Dataset(data_dir=data_dir, tissue=tissue, split="test", mode=mode)

    loader_kwargs: dict = dict(num_workers=num_workers, pin_memory=True)
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    return train_loader, val_loader, test_loader


# ===========================================================================
# Dataset builder — reproduces the paper's R enrichment pipeline.
# Run: python data.py --build --output-dir ./data
# ===========================================================================

_REPO_BASE = (
    "https://raw.githubusercontent.com/tobjores/"
    "Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/main"
)

_SPECIES = ["At", "Zm", "Sb"]
_TISSUES = ["leaf", "proto"]
_ENHANCER_CONDS = [True, False]
_REPS = [1, 2]
_READ_COUNT_CUTOFF = 5
_SP_PREFIXES = {"AT": "At", "Zm": "Zm", "ENSRNA": "Sb", "SORBI_": "Sb"}


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return
    urllib.request.urlretrieve(url, dest)


def download_raw_data(cache_dir: str) -> None:
    """Download all raw files from the paper's GitHub repo into cache_dir."""
    sub_dir = os.path.join(cache_dir, "subassembly")
    _download(f"{_REPO_BASE}/data/subassembly/subassembly_pPSup_plPRO_variant.tsv.gz",
              os.path.join(sub_dir, "subassembly_pPSup_plPRO_variant.tsv.gz"))
    _download(f"{_REPO_BASE}/data/subassembly/controls_ZmUTR.tsv",
              os.path.join(sub_dir, "controls_ZmUTR.tsv"))
    _download(f"{_REPO_BASE}/data/subassembly/controls_SbUTR.tsv",
              os.path.join(sub_dir, "controls_SbUTR.tsv"))

    annot_dir = os.path.join(cache_dir, "annotation")
    for species_full in ["Arabidopsis", "Maize", "Sorghum"]:
        _download(
            f"{_REPO_BASE}/data/promoter_annotation/{species_full}_all_promoters_unique.tsv",
            os.path.join(annot_dir, f"{species_full}_all_promoters_unique.tsv"),
        )

    for tissue in _TISSUES:
        for sp in _SPECIES:
            for rep in _REPS:
                rep_dir = os.path.join(cache_dir, "barcode_counts", tissue, f"{sp}_Rep{rep}")
                for enh in ["35SEnh", "noEnh"]:
                    roman = "I" if rep == 1 else "II"
                    base = f"barcodes_pPSup_{sp}PRO_{enh}_{roman}"
                    for tag in ["_dark.count.gz", "_inp.count.gz"]:
                        if tag == "_inp.count.gz" and tissue == "proto" and sp == "At" and rep == 2:
                            continue
                        fname = f"{base}{tag}"
                        _download(
                            f"{_REPO_BASE}/data/barcode_counts/{tissue}/{sp}_Rep{rep}/{fname}",
                            os.path.join(rep_dir, fname),
                        )
                    if tissue == "leaf":
                        fname = f"{base}_light.count.gz"
                        _download(
                            f"{_REPO_BASE}/data/barcode_counts/{tissue}/{sp}_Rep{rep}/{fname}",
                            os.path.join(rep_dir, fname),
                        )

    cnn_dir = os.path.join(cache_dir, "cnn_splits")
    for tissue in _TISSUES:
        _download(f"{_REPO_BASE}/CNN/CNN_test_{tissue}.tsv",
                  os.path.join(cnn_dir, f"CNN_test_{tissue}.tsv"))


def _gene_to_sp(gene: str) -> str:
    for prefix, sp in _SP_PREFIXES.items():
        if gene.startswith(prefix):
            return sp
    return "unknown"


def _load_annotation(cache_dir: str) -> pd.DataFrame:
    annot_dir = os.path.join(cache_dir, "annotation")
    sp_map = {"Arabidopsis": "At", "Maize": "Zm", "Sorghum": "Sb"}
    frames = []
    for species_full, sp in sp_map.items():
        df = pd.read_csv(os.path.join(annot_dir, f"{species_full}_all_promoters_unique.tsv"), sep="\t")
        df["sp"] = sp
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_controls(cache_dir: str) -> pd.DataFrame:
    sub_dir = os.path.join(cache_dir, "subassembly")
    zm_ctrl = pd.read_csv(os.path.join(sub_dir, "controls_ZmUTR.tsv"), sep="\t")
    zm_ctrl["utr_group"] = "ZmUTR"
    sb_ctrl = pd.read_csv(os.path.join(sub_dir, "controls_SbUTR.tsv"), sep="\t")
    sb_ctrl["utr_group"] = "SbUTR"
    return pd.concat([zm_ctrl, sb_ctrl], ignore_index=True)


def _load_count_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=["count", "barcode"], compression="gzip")


def filter_subassembly(cache_dir: str) -> pd.DataFrame:
    sub_path = os.path.join(cache_dir, "subassembly", "subassembly_pPSup_plPRO_variant.tsv.gz")
    raw = pd.read_csv(sub_path, sep="\t", compression="gzip")
    controls = _load_controls(cache_dir)
    annotation = _load_annotation(cache_dir)

    raw["sp"] = raw["gene"].apply(_gene_to_sp)

    ctrl_barcodes = set(controls["barcode"])
    ctrl_map = controls.set_index("barcode")[["promoter", "enhancer"]].to_dict("index")

    def _recode_control(row):
        if row["barcode"] in ctrl_barcodes:
            info = ctrl_map[row["barcode"]]
            pro = "withPRO" if info["promoter"] == "35S" else "noPRO"
            enh = "withENH" if info["enhancer"] == "35S" else "noENH"
            return f"control-{pro}-{enh}"
        return row["gene"]

    raw["gene"] = raw.apply(_recode_control, axis=1)

    bc_gene_counts = raw.groupby("barcode")["gene"].nunique()
    ambiguous_bcs = set(bc_gene_counts[bc_gene_counts > 1].index)

    resolved = []
    for bc, group in raw[raw["barcode"].isin(ambiguous_bcs)].groupby("barcode"):
        counts = group.groupby("gene")["assembly.count"].sum().sort_values(ascending=False)
        if len(counts) >= 2 and counts.iloc[0] >= 10 * counts.iloc[1]:
            resolved.append(group[group["gene"] == counts.index[0]])

    unambiguous = raw[~raw["barcode"].isin(ambiguous_bcs)]
    raw = pd.concat([unambiguous] + resolved, ignore_index=True) if resolved else unambiguous

    raw["FL"] = (raw["start"] == 1) & (raw["stop"] == 170)
    annot_slim = annotation[["gene", "type"]].drop_duplicates()
    raw = raw.merge(annot_slim, on="gene", how="left")
    raw.loc[raw["gene"].str.startswith("control"), "type"] = "control"

    return raw[["barcode", "gene", "start", "stop", "variant", "FL", "sp", "type"]]


def compute_enrichment(cache_dir: str, subassembly: pd.DataFrame) -> pd.DataFrame:
    all_results = []

    for tissue in _TISSUES:
        for sp in _SPECIES:
            sub_sp = subassembly[subassembly["sp"] == sp].copy()
            controls_mask = subassembly["gene"].str.startswith("control")
            sub_with_ctrl = pd.concat([
                sub_sp[~sub_sp["gene"].str.startswith("control")],
                subassembly[controls_mask],
            ]).drop_duplicates(subset=["barcode"])

            for enh in _ENHANCER_CONDS:
                enh_str = "35SEnh" if enh else "noEnh"
                rep_enrichments = []

                for rep in _REPS:
                    roman = "I" if rep == 1 else "II"
                    rep_dir = os.path.join(cache_dir, "barcode_counts", tissue, f"{sp}_Rep{rep}")
                    base = f"barcodes_pPSup_{sp}PRO_{enh_str}_{roman}"

                    if tissue == "proto" and sp == "At" and rep == 2:
                        inp_dir = os.path.join(cache_dir, "barcode_counts", tissue, f"{sp}_Rep1")
                        inp_base = f"barcodes_pPSup_{sp}PRO_{enh_str}_I"
                        inp_path = os.path.join(inp_dir, f"{inp_base}_inp.count.gz")
                    else:
                        inp_path = os.path.join(rep_dir, f"{base}_inp.count.gz")

                    dark_path = os.path.join(rep_dir, f"{base}_dark.count.gz")
                    if not os.path.exists(dark_path) or not os.path.exists(inp_path):
                        continue

                    inp = _load_count_file(inp_path)
                    dark = _load_count_file(dark_path)
                    merged = dark.merge(inp, on="barcode", suffixes=("_out", "_inp"))
                    merged = merged.merge(sub_with_ctrl[["barcode", "gene", "variant", "FL", "type"]], on="barcode")
                    merged = merged[(merged["count_inp"] >= _READ_COUNT_CUTOFF)
                                    & (merged["count_out"] >= _READ_COUNT_CUTOFF)]
                    if len(merged) == 0:
                        continue

                    total_out = merged["count_out"].sum()
                    total_inp = merged["count_inp"].sum()
                    merged["enrichment"] = np.log2(
                        (merged["count_out"] / total_out) / (merged["count_inp"] / total_inp)
                    )
                    rep_enrichments.append(merged.assign(rep=rep))

                if not rep_enrichments:
                    continue

                combined = pd.concat(rep_enrichments, ignore_index=True)
                wt_mask = (
                    combined["gene"].str.startswith("control")
                    | ((combined["variant"] == "WT") & combined["FL"])
                ) & (combined["gene"] != "35Spr")
                combined = combined[wt_mask]

                ctrl_gene = "control-withPRO-noENH"
                for rep_val in combined["rep"].unique():
                    rep_mask = combined["rep"] == rep_val
                    ctrl_mask = rep_mask & (combined["gene"] == ctrl_gene)
                    if ctrl_mask.sum() > 0:
                        ctrl_median = combined.loc[ctrl_mask, "enrichment"].median()
                        combined.loc[rep_mask, "enrichment"] -= ctrl_median

                combined = combined[~combined["gene"].str.startswith("control")]

                agg_by_bc = (
                    combined.groupby(["gene", "type", "rep"])["enrichment"].median().reset_index()
                )
                agg_by_rep = (
                    agg_by_bc.groupby(["gene", "type"])["enrichment"].mean().reset_index()
                )
                agg_by_rep["sys"] = tissue
                agg_by_rep["sp"] = sp
                agg_by_rep["enhancer"] = enh
                all_results.append(agg_by_rep)

    return pd.concat(all_results, ignore_index=True)


def _load_test_genes(cache_dir: str, tissue: str) -> set[str]:
    path = os.path.join(cache_dir, "cnn_splits", f"CNN_test_{tissue}.tsv")
    df = pd.read_csv(path, sep="\t")
    return set(df["gene"])


def export_datasets(cache_dir: str, output_dir: str, enrichment: pd.DataFrame, annotation: pd.DataFrame) -> None:
    os.makedirs(output_dir, exist_ok=True)
    seq_map = annotation[["gene", "sequence"]].drop_duplicates().set_index("gene")["sequence"]

    for tissue in _TISSUES:
        test_genes = _load_test_genes(cache_dir, tissue)

        for enh in _ENHANCER_CONDS:
            enh_str = "35SEnh" if enh else "noEnh"
            subset = enrichment[(enrichment["sys"] == tissue) & (enrichment["enhancer"] == enh)].copy()
            subset["sequence"] = subset["gene"].map(seq_map)
            subset = subset.dropna(subset=["sequence"])

            train = subset[~subset["gene"].isin(test_genes)]
            test = subset[subset["gene"].isin(test_genes)]
            cols = ["gene", "sp", "type", "sequence", "enrichment"]

            train[cols].to_csv(os.path.join(output_dir, f"jores21_{tissue}_{enh_str}_train.tsv"),
                               sep="\t", index=False)
            test[cols].to_csv(os.path.join(output_dir, f"jores21_{tissue}_{enh_str}_test.tsv"),
                              sep="\t", index=False)

            print(f"{tissue} {enh_str}: train={len(train):,}, test={len(test):,}, "
                  f"enrichment=[{subset['enrichment'].min():.3f}, {subset['enrichment'].max():.3f}]")


def build_dataset(output_dir: str, cache_dir: str | None = None) -> None:
    """Build the full Jores21 dataset from the paper's raw GitHub data."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "jores21-raw")

    print("Downloading raw data...")
    download_raw_data(cache_dir)

    print("Filtering subassembly...")
    subassembly = filter_subassembly(cache_dir)
    print(f"Subassembly: {len(subassembly):,} barcode-gene pairs")

    print("Computing enrichment...")
    enrichment = compute_enrichment(cache_dir, subassembly)
    print(f"Enrichment: {len(enrichment):,} gene-condition pairs")

    print("Loading annotation...")
    annotation = _load_annotation(cache_dir)

    print("Exporting datasets...")
    export_datasets(cache_dir, output_dir, enrichment, annotation)
    print(f"Built Jores21 TSVs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jores21 plant STARR-seq builder")
    parser.add_argument("--build", action="store_true", help="Build dataset from raw paper data")
    parser.add_argument("--output-dir", default="./data")
    args = parser.parse_args()

    if args.build:
        build_dataset(args.output_dir)
    else:
        parser.print_help()
