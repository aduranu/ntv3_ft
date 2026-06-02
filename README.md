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
  - **Stage 2**: backbone unfrozen, both at `1e-5`, runs up to `STAGE2_EPOCHS=50`.

## Layout

```
ntv3_ft/
├── README.md
├── pyproject.toml     # env spec — JAX + flax/nnx + optax + nucleotide_transformer pkg
├── config.toml        # tissue, mode, data_dir, model_name
├── data.py            # Jores21 Dataset + dataloaders + NTv3 collate + dataset builder CLI
└── finetune.py        # main script — head, train_step / eval_step, two-stage loop
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

Reads `config.toml`. Default config is `tissue = "leaf"`, `mode = "combined"`. Change
`tissue` to `"proto"` to swap to the protoplast assay. `mode` choices:

- `"combined"` — train on both `35SEnh` and `noEnh` data, constructs are built with
  per-row enhancer flag. Recommended (broadest data coverage).
- `"enhancer"` — train on `35SEnh` only, with the 35S enhancer prepended to constructs.
- `"promoter_only"` — train on the raw 170 bp promoter sequences without construct
  context.

Outputs `best.pkl` (pickle of head + optional backbone state + summary). Final metrics
print as `val_pearson:`, `test_pearson:`, `test_mse:`, `peak_vram_mb:`.

## Hyperparameter tuning

Edit constants at the top of `finetune.py`:

| Name | Default | What it does |
|---|---|---|
| `HIDDEN_SIZE` | 1024 | head MLP width |
| `DROPOUT` | 0.2 | head dropout |
| `BATCH_SIZE` | 128 | batch size |
| `LEARNING_RATE` | 5e-4 | stage 1 lr (head only) |
| `STAGE1_EPOCHS` | 100 | epoch cap; early stopping usually triggers first |
| `STAGE2_LR` | 1e-5 | stage 2 lr (head + backbone) |
| `STAGE2_EPOCHS` | 50 | stage 2 epoch cap |
| `STAGE2_ENABLED` | True | set to `False` to skip stage 2 |
| `EARLY_STOPPING_PATIENCE` | 5 | consecutive epochs without val_r improvement → break stage 1 |
| `REVERSE_COMPLEMENT` | False | random RC augmentation |
| `RANDOM_SHIFT` | True | random circular shift augmentation (±25 bp) |

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
