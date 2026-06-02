"""
Standalone NTv3 fine-tuning on Jores21 plant STARR-seq.

Uses the *pretrained* NTv3 checkpoint and taps only the conv-tower features
(embed_layer -> stem -> conv_tower) — the transformer stack is loaded but never
executed. A small attention-pool MLP head is trained on those features.

Two-stage training:
  stage 1 — backbone frozen, train head only with MSE loss and early stopping
  stage 2 — backbone unfrozen, lower LR, continues from best stage-1 head

Usage: python finetune.py
"""

from __future__ import annotations

import gc
import math
import os
import pickle
import time
import tomllib
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from scipy.stats import pearsonr

from nucleotide_transformer_v3.pretrained import get_pretrained_ntv3_model

from data import (
    PROMOTER_LENGTH, SEQUENCE_LENGTH,
    create_dataloaders, make_ntv3_collate_fn,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)

DATA_DIR = os.path.expanduser(_config["data_dir"])
MODEL_NAME = _config["model_name"]
TISSUE = _config["tissue"]
MODE = _config["mode"]


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 1024
DROPOUT = 0.2

BATCH_SIZE = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0

# augmentations
REVERSE_COMPLEMENT = False
RC_PROB = 0.5
RANDOM_SHIFT = True
SHIFT_PROB = 0.5
MAX_SHIFT = 25

# two-stage
STAGE1_EPOCHS = 100              # hard cap; early stopping will usually finish sooner
STAGE2_ENABLED = True
STAGE2_LR = 1e-5
STAGE2_EPOCHS = 50

EARLY_STOPPING_PATIENCE = 5      # consecutive epochs of no val_r -> improvement stop stage 1

# encoder output positions: conv_tower downsamples by 128×
SEQ_LEN = PROMOTER_LENGTH if MODE == "promoter_only" else SEQUENCE_LENGTH
N_ENC_POSITIONS = math.ceil(SEQ_LEN / 128)

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pkl")


# ---------------------------------------------------------------------------
# Embed function — pretrained NTv3, conv-tower features only
# ---------------------------------------------------------------------------

# walk embed_layer -> stem -> conv_tower; skip transformer + species conditioning
def get_embeddings(model, tokens: jnp.ndarray) -> jnp.ndarray:
    x = model.embed_layer(tokens)
    x = model.stem(x)
    x, _ = model.conv_tower(x)
    return x


# ---------------------------------------------------------------------------
# Head — MLP with flatten + attention-pool concat input
# ---------------------------------------------------------------------------

class MPRAHead(nnx.Module):

    def __init__(self, n_positions: int, encoder_dim: int, hidden_size: int = 1024,
                 dropout: float = 0.2, *, rngs: nnx.Rngs) -> None:
        flatten_dim = n_positions * encoder_dim

        self.norm = nnx.LayerNorm(encoder_dim, rngs=rngs)
        self.attn = nnx.Linear(encoder_dim, 1, rngs=rngs)
        self.fc1 = nnx.Linear(flatten_dim + encoder_dim, hidden_size, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.fc_out = nnx.Linear(hidden_size, 1, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, encoder_output: jnp.ndarray) -> jnp.ndarray:
        x = self.norm(encoder_output)

        flat = x.reshape(x.shape[0], -1)
        attn_weights = jax.nn.softmax(self.attn(x), axis=1)
        attn_pool = (x * attn_weights).sum(axis=1)
        x = jnp.concatenate([flat, attn_pool], axis=1)

        x = jax.nn.gelu(self.fc1(x))
        x = self.dropout(x)

        x = x + jax.nn.gelu(self.fc2(x))
        x = self.dropout(x)

        return self.fc_out(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / eval step factories
# ---------------------------------------------------------------------------

# frozen-backbone training: backbone runs eagerly (stop_gradient), only head is JIT-compiled.
# unfrozen-backbone training: the whole forward+backward is JIT-compiled together.
def make_train_step(embed_fn: Callable, freeze_backbone: bool):

    if freeze_backbone:
        @nnx.jit
        def _head_step(head, head_opt, enc_out, targets):
            def loss_fn(head):
                preds = head(enc_out)
                loss = jnp.mean((preds - targets) ** 2)
                return loss, preds

            (loss, preds), grads = nnx.value_and_grad(loss_fn, has_aux=True)(head)
            head_opt.update(grads)
            return loss, preds

        def train_step(model, head, head_opt, backbone_opt, tokens, targets):
            enc_out = jax.lax.stop_gradient(embed_fn(model, tokens))
            return _head_step(head, head_opt, enc_out, targets)

    else:
        @nnx.jit
        def train_step(model, head, head_opt, backbone_opt, tokens, targets):
            def loss_fn(model, head):
                enc_out = embed_fn(model, tokens)
                preds = head(enc_out)
                loss = jnp.mean((preds - targets) ** 2)
                return loss, preds

            (loss, preds), (model_grads, head_grads) = (
                nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(model, head)
            )
            head_opt.update(head_grads)
            backbone_opt.update(model_grads)
            return loss, preds

    return train_step


def make_eval_step(embed_fn: Callable):

    @nnx.jit
    def _head_eval(head, enc_out, targets):
        preds = head(enc_out)
        loss = jnp.mean((preds - targets) ** 2)
        return loss, preds

    def eval_step(model, head, tokens, targets):
        enc_out = embed_fn(model, tokens)
        return _head_eval(head, enc_out, targets)

    return eval_step


def pearson_r(preds: np.ndarray, targets: np.ndarray) -> float:
    preds = np.asarray(preds).flatten()
    targets = np.asarray(targets).flatten()
    if len(preds) < 2:
        return 0.0
    r, _ = pearsonr(preds, targets)
    return float(r)


def evaluate(model, head, loader, eval_step_fn: Callable) -> tuple[float, float]:
    total_loss = 0.0
    all_preds, all_targets = [], []

    for tokens_np, targets_np in loader:
        tokens = jnp.array(tokens_np)
        targets = jnp.array(targets_np)
        loss, preds = eval_step_fn(model, head, tokens, targets)
        total_loss += float(loss)
        all_preds.append(np.asarray(preds))
        all_targets.append(np.asarray(targets_np))

    preds_cat = np.concatenate(all_preds)
    targets_cat = np.concatenate(all_targets)
    return total_loss / max(len(loader), 1), pearson_r(preds_cat, targets_cat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    print("Loading pretrained NTv3...")
    model, tokenizer, config = get_pretrained_ntv3_model(MODEL_NAME, use_bfloat16=True)
    encoder_dim = config.embed_dim

    n_backbone = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))

    rngs = nnx.Rngs(42)
    head = MPRAHead(
        n_positions=N_ENC_POSITIONS, encoder_dim=encoder_dim,
        hidden_size=HIDDEN_SIZE, dropout=DROPOUT, rngs=rngs,
    )
    n_head = sum(p.size for p in jax.tree.leaves(nnx.state(head, nnx.Param)))

    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: Jores21 {TISSUE} {MODE}")
    print(f"Sequence length: {SEQ_LEN} bp -> {N_ENC_POSITIONS} encoder positions")
    print(f"Encoder dim: {encoder_dim}")
    print(f"Backbone params: {n_backbone:,} (stage 1 frozen -> {'unfrozen stage 2' if STAGE2_ENABLED else 'stays frozen'})")
    print(f"Head params: {n_head:,}")

    collate_fn = make_ntv3_collate_fn(tokenizer)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=DATA_DIR, tissue=TISSUE,
        batch_size=BATCH_SIZE, mode=MODE,
        reverse_complement=REVERSE_COMPLEMENT, rc_prob=RC_PROB,
        random_shift=RANDOM_SHIFT, shift_prob=SHIFT_PROB, max_shift=MAX_SHIFT,
        collate_fn=collate_fn, num_workers=0,
    )

    # stage 1 optimizer: head only, constant lr, grad clipping
    head_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    )
    head_opt = nnx.Optimizer(head, head_tx)

    train_step_s1 = make_train_step(get_embeddings, freeze_backbone=True)
    eval_step_fn = make_eval_step(get_embeddings)

    # gc tricks: collect + freeze + disable to avoid 500ms stalls during training
    gc.collect()
    gc.freeze()
    gc.disable()

    # -------------------------------------------------------------------
    # Stage 1: frozen backbone, train head only
    # -------------------------------------------------------------------

    print(f"--- stage 1: frozen backbone, lr={LEARNING_RATE}, patience={EARLY_STOPPING_PATIENCE} epochs ---")

    t_train_start = time.time()
    best_val_pearson = -float("inf")
    best_head_state = None
    best_backbone_state = None
    epochs_since_best = 0
    stage1_epochs_done = 0
    total_steps = 0

    for epoch in range(STAGE1_EPOCHS):
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, (tokens_np, targets_np) in enumerate(train_loader):
            tokens = jnp.array(tokens_np)
            targets = jnp.array(targets_np)

            loss, _ = train_step_s1(model, head, head_opt, None, tokens, targets)
            loss_val = float(loss)

            if math.isnan(loss_val):
                raise RuntimeError(f"NaN loss at stage 1 epoch {epoch} batch {batch_idx}")

            epoch_loss += loss_val
            epoch_batches += 1
            total_steps += 1

        val_loss, val_pr = evaluate(model, head, val_loader, eval_step_fn)
        is_best = val_pr > best_val_pearson

        if is_best:
            best_val_pearson = val_pr
            best_head_state = jax.tree.map(jnp.copy, nnx.state(head))
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        stage1_epochs_done = epoch + 1
        avg_loss = epoch_loss / max(epoch_batches, 1)
        star = "*" if is_best else " "
        print(
            f"epoch {epoch:3d} {star} | "
            f"train_loss: {avg_loss:.6f} | "
            f"val_r: {val_pr:.4f} (best: {best_val_pearson:.4f}) | "
            f"elapsed: {time.time() - t_train_start:.0f}s",
            flush=True,
        )

        if (epoch + 1) % 5 == 0:
            gc.collect()

        if epochs_since_best >= EARLY_STOPPING_PATIENCE:
            print(f"early stopped after {stage1_epochs_done} epochs")
            break

    print(f"stage 1 done: {stage1_epochs_done} epochs, best val_pearson: {best_val_pearson:.6f}")

    # -------------------------------------------------------------------
    # Stage 2: unfrozen backbone
    # -------------------------------------------------------------------

    stage2_epochs_done = 0

    if STAGE2_ENABLED:
        if best_head_state is not None:
            nnx.update(head, best_head_state)

        head_tx_s2 = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=STAGE2_LR, weight_decay=WEIGHT_DECAY),
        )
        head_opt = nnx.Optimizer(head, head_tx_s2)

        backbone_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=STAGE2_LR, weight_decay=WEIGHT_DECAY),
        )
        backbone_opt = nnx.Optimizer(model, backbone_tx)

        train_step_s2 = make_train_step(get_embeddings, freeze_backbone=False)

        print(f"--- stage 2: unfrozen backbone, lr={STAGE2_LR}, {STAGE2_EPOCHS} epochs ---")

        for epoch in range(STAGE2_EPOCHS):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, (tokens_np, targets_np) in enumerate(train_loader):
                tokens = jnp.array(tokens_np)
                targets = jnp.array(targets_np)

                loss, _ = train_step_s2(model, head, head_opt, backbone_opt, tokens, targets)
                loss_val = float(loss)

                if math.isnan(loss_val):
                    raise RuntimeError(f"NaN loss at stage 2 epoch {epoch} batch {batch_idx}")

                epoch_loss += loss_val
                epoch_batches += 1
                total_steps += 1

            val_loss, val_pr = evaluate(model, head, val_loader, eval_step_fn)
            is_best = val_pr > best_val_pearson

            if is_best:
                best_val_pearson = val_pr
                best_head_state = jax.tree.map(jnp.copy, nnx.state(head))
                best_backbone_state = jax.tree.map(jnp.copy, nnx.state(model))

            stage2_epochs_done = epoch + 1
            avg_loss = epoch_loss / max(epoch_batches, 1)
            star = "*" if is_best else " "
            print(
                f"epoch {stage1_epochs_done + epoch:3d} {star} | "
                f"train_loss: {avg_loss:.6f} | "
                f"val_r: {val_pr:.4f} (best: {best_val_pearson:.4f}) | "
                f"elapsed: {time.time() - t_train_start:.0f}s",
                flush=True,
            )

            if (epoch + 1) % 5 == 0:
                gc.collect()

        print(f"stage 2 done: {stage2_epochs_done} epochs, best val_pearson: {best_val_pearson:.6f}")

    gc.enable()

    # restore best weights for final eval
    if best_head_state is not None:
        nnx.update(head, best_head_state)
    if best_backbone_state is not None:
        nnx.update(model, best_backbone_state)

    val_loss, val_pr = evaluate(model, head, val_loader, eval_step_fn)

    # test eval — scipy pearsonr for reporting
    all_preds, all_tgts = [], []
    for tokens_np, targets_np in test_loader:
        tokens = jnp.array(tokens_np)
        enc_out = get_embeddings(model, tokens)
        preds = head(enc_out)
        all_preds.append(np.asarray(preds))
        all_tgts.append(np.asarray(targets_np))

    preds_np = np.concatenate(all_preds)
    tgts_np = np.concatenate(all_tgts)
    test_pearson_scipy, _ = pearsonr(preds_np, tgts_np)
    test_mse = float(np.mean((preds_np - tgts_np) ** 2))

    t_end = time.time()
    backend = jax.default_backend()
    if backend == "gpu":
        peak_vram_mb = jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    print("---")
    print(f"val_pearson:      {best_val_pearson:.6f}")
    print(f"val_mse:          {val_loss:.6f}")
    print(f"test_pearson:     {test_pearson_scipy:.6f}")
    print(f"test_mse:         {test_mse:.6f}")
    print(f"training_seconds: {time.time() - t_train_start:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_epochs:       {stage1_epochs_done + stage2_epochs_done}")
    print(f"num_steps:        {total_steps}")
    print(f"tissue:           {TISSUE}")
    print(f"mode:             {MODE}")
    print(f"stage1_epochs:    {stage1_epochs_done}")
    print(f"stage2_epochs:    {stage2_epochs_done}")

    # only save trainable Param leaves; nnx.state(.) without a filter includes the
    # dropout RngStream (PRNGKey dtype) which can't be np.asarray'd.
    head_params = nnx.state(head, nnx.Param)
    backbone_params = nnx.state(model, nnx.Param) if STAGE2_ENABLED else None

    ckpt = {
        "head_state": jax.tree.map(np.asarray, head_params),
        "backbone_state": jax.tree.map(np.asarray, backbone_params) if backbone_params is not None else None,
        "val_pearson": best_val_pearson,
        "test_pearson": test_pearson_scipy,
        "config": {"tissue": TISSUE, "mode": MODE, "model_name": MODEL_NAME},
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"checkpoint:       {CHECKPOINT_PATH}")
