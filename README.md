# ntv3_ft

Standalone fine-tuning of the **pretrained** Nucleotide Transformer v3 (NTv3) encoder
on the **Jores et al. 2021** plant promoter STARR-seq dataset. JAX/Flax NNX.

This is a self-contained extract — no dependence on the original autotune research repo.

## What it does

- Loads `NTv3_650M_pre` via `nucleotide_transformer_v3.pretrained.get_pretrained_ntv3_model`
- Taps the **conv-tower features only** (`embed_layer → stem → conv_tower`). The
  transformer stack is loaded into memory but never executed and never updated —
  this is an "early-tap" fine-tune that uses NTv3 as a 128×-downsampling convolutional
  feature extractor.
- Grafts a small `MPRAHead` (LayerNorm → flatten + attention-pool concat → MLP → scalar)
  on top of the conv features.
- Trains with **MSE loss**, AdamW, gradient clipping, in two stages:
  - **Stage 1**: backbone frozen, head trained until `val_pearson` plateaus
    (early stopping, default patience = 5 epochs).
  - **Stage 2**: backbone unfrozen, trained under a wall-time LR schedule
    (warmup → constant → warmdown), peak LR ~4e-4, with early stopping + a wall-clock
    budget. This tuned recipe lifts test Pearson ~+0.03–0.04 over the frozen-stage-1 head.

## Layout

```
ntv3_ft/
├── README.md
├── pyproject.toml     # env spec — JAX + flax/nnx + optax + nucleotide_transformer pkg
├── config.json        # experiment selection: tissue, mode, data_dir, model_name
├── data.py            # Jores21 Dataset + dataloaders + NTv3 collate + dataset builder CLI
├── finetune.py        # main script — head, two-stage loop, hardcoded per-stage HPs
├── plot_curves.py     # plot train/val loss curves from the run's metrics CSV
└── run_finetune.sbatch # SLURM launcher
```

## Install

```bash
conda create -n ntv3_ft python=3.12 -y
conda activate ntv3_ft
pip install -e .
```

If you run into `libstdc++` resolution issues with the conda-bundled libs (common on
older HPC nodes), prepend:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Build the dataset

```bash
python data.py --build --output-dir ./data
```

This downloads ~200 MB of raw barcode counts + annotation TSVs from the Jores et al.
paper's GitHub repo and reproduces their R enrichment pipeline. Output: 8 TSV files
in `./data/` (`jores21_{leaf,proto}_{35SEnh,noEnh}_{train,test}.tsv`).

## Train

```bash
python finetune.py
```

That's it — one run does **both** stages with the validated hyperparameters and saves
**both** checkpoints:

- **stage 1 ("probing")** — backbone frozen, train head only → `best_stage1_<tissue>_<mode>.pkl`
- **stage 2 ("full fine-tune")** — backbone unfrozen, tuned recipe + wall-time LR
  schedule → `best_<tissue>_<mode>.pkl`

It also writes `metrics_<tissue>_<mode>.csv` (per-epoch `train_loss,val_loss,val_pearson,
head_lr` for both stages) and prints `val_pearson:`, `test_pearson:`, `test_mse:`
(held-out, **gene-split** test set), `peak_vram_mb:`.

Pick the experiment in **`config.json`** (no code edits, no env vars):

```json
{ "data_dir": "./data", "tissue": "leaf", "mode": "combined", "model_name": "NTv3_650M_pre" }
```

- `tissue`: `"leaf"` or `"proto"`.
- `mode`: `"combined"` (both `35SEnh` + `noEnh`; recommended), `"enhancer"` (35S only),
  or `"promoter_only"` (raw 170 bp promoters).

To train only the probing head, set `RUN_STAGE2 = False` at the top of `finetune.py`.

Plot the learning curves after a run with `python plot_curves.py` (reads the metrics CSV
for the tissue/mode in `config.json`, writes `loss_curves.png`).

## Hyperparameters

All hyperparameters are **hardcoded** near the top of `finetune.py`, separated per stage
(`S1_*` for the frozen-backbone probe, `S2_*` for the full fine-tune) — no env vars. The
defaults are the validated recipe behind the reported test Pearson (leaf 0.885 / proto
0.875, vs frozen-stage-1 0.843 / 0.841). Edit there only if you want to experiment.

| Stage 1 (`S1_*`) | Default | | Stage 2 (`S2_*`) | Default |
|---|---|---|---|---|
| `S1_BATCH_SIZE` | 128 | | `S2_BATCH_SIZE` | 256 (~47 GB VRAM) |
| `S1_LR` | 5e-4 (constant) | | `S2_BASE_LR` | 4e-4 (peak) |
| `S1_WEIGHT_DECAY` | 0.0 | | `S2_BACKBONE_LR_SCALE` | 1.0 (uniform) |
| `S1_REVERSE_COMPLEMENT` | False | | `S2_WEIGHT_DECAY` | 0.01 |
| `S1_SHIFT_PROB`/`S1_MAX_SHIFT` | 0.5 / 25 | | `S2_REVERSE_COMPLEMENT` | True |
| `S1_MAX_EPOCHS` | 100 | | `S2_SHIFT_PROB`/`S2_MAX_SHIFT` | 1.0 / 50 |
| `S1_EARLY_STOPPING_PATIENCE` | 5 | | `S2_TIME_BUDGET` | 2400 s (schedule span) |
| `HIDDEN_SIZE` / `DROPOUT` | 1024 / 0.2 | | `S2_WARMUP`/`S2_WARMDOWN`/`S2_FINAL` | 0.05 / 0.30 / 0.01 |
| | | | `S2_MAX_EPOCHS` / `S2_EARLY_STOPPING_PATIENCE` | 50 / 8 |

Stage 2 uses a wall-time LR schedule (linear **warmup → constant → linear warmdown** to
1% of peak) spanning `S2_TIME_BUDGET` seconds, with the head at the scheduled LR and the
backbone at `S2_BACKBONE_LR_SCALE ×` that.

## How the encoder is isolated

There is no model surgery, no submodule extraction. `get_embeddings()` in `finetune.py`
just walks three child modules of the loaded `NTv3Pretrained` instance:

```python
def get_embeddings(model, tokens):
    x = model.embed_layer(tokens)
    x = model.stem(x)
    x, _ = model.conv_tower(x)
    return x
```

The transformer stack and species-conditioning machinery still exist as attributes
on the model but are never called. Freezing during stage 1 is done with an optimizer
mask (`jax.lax.stop_gradient` on the backbone output) — no gradients ever touch any
backbone parameter.

## Notes

- `BATCH_SIZE=128` requires ~30 GB GPU memory at `SEQUENCE_LENGTH=437`. Halve if you
  hit OOM.
- `num_workers=0` in `create_dataloaders` is intentional — forked PyTorch DataLoader
  workers break JAX device handles. Performance impact is small because the collate
  function is cheap (tokenization only).
- The conv_tower downsamples by 128× along the sequence axis: 437 bp input → 4 encoder
  positions; 170 bp input → 2 encoder positions. The head adapts via `N_ENC_POSITIONS`.

## Dataset / model attribution

- Jores et al. 2021, *Synthetic Promoter Designs Enabled by a Comprehensive Analysis
  of Plant Core Promoters*. Raw data is fetched from the paper's
  [GitHub repo](https://github.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters).
- NTv3 / Nucleotide Transformer v3 by InstaDeep
  ([repo](https://github.com/instadeepai/nucleotide-transformer)). Weights are pulled
  from the official HuggingFace mirror at first model load.
