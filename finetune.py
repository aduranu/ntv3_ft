"""
Standalone NTv3 fine-tuning on Jores21 plant STARR-seq.

Uses the *pretrained* NTv3 checkpoint and taps only the conv-tower features
(embed_layer -> stem -> conv_tower) — the transformer stack is loaded but never
executed. A small attention-pool MLP head is trained on those features.

One run does both stages and saves both checkpoints:
  stage 1 ("probing") — backbone frozen, train head only; light augmentation.
                        -> best_stage1_<tissue>_<mode>.pkl
  stage 2 ("full")    — backbone unfrozen, tuned recipe + wall-time LR schedule.
                        -> best_<tissue>_<mode>.pkl

Hyperparameters below are the validated values (separated per stage); just run it.
Experiment selection (tissue / mode / paths) lives in config.json.

Usage: python finetune.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import time
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
# Config — experiment selection only (HPs are hardcoded below)
# ---------------------------------------------------------------------------

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
with open(_config_path) as f:
    _config = json.load(f)

DATA_DIR = os.path.expanduser(_config["data_dir"])
MODEL_NAME = _config["model_name"]
TISSUE = _config["tissue"]
MODE = _config["mode"]


# ---------------------------------------------------------------------------
# Hyperparameters — validated values, separated per stage. Edit here only.
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 1024
DROPOUT = 0.2

# --- Stage 1: frozen backbone, train head ("probing"). Light augmentation. ---
S1_BATCH_SIZE = 128
S1_LR = 5e-4                    # constant head LR
S1_WEIGHT_DECAY = 0.0
S1_REVERSE_COMPLEMENT = False
S1_RC_PROB = 0.5
S1_RANDOM_SHIFT = True
S1_SHIFT_PROB = 0.5
S1_MAX_SHIFT = 25
S1_MAX_EPOCHS = 100            # hard cap; early stopping usually finishes sooner
S1_EARLY_STOPPING_PATIENCE = 5

# --- Stage 2: unfrozen backbone, full fine-tune. Tuned recipe. ---
RUN_STAGE2 = True             # set False for a stage-1-only ("probe-only") run
S2_BATCH_SIZE = 256
S2_BASE_LR = 4e-4             # peak head LR (sweep winner; robust across leaf/proto)
S2_BACKBONE_LR_SCALE = 1.0   # backbone LR = base * this (uniform LR won the sweep)
S2_WEIGHT_DECAY = 0.01
S2_REVERSE_COMPLEMENT = True
S2_RC_PROB = 0.5
S2_RANDOM_SHIFT = True
S2_SHIFT_PROB = 1.0
S2_MAX_SHIFT = 50
S2_MAX_EPOCHS = 50           # hard cap; wall-time budget usually binds first
S2_EARLY_STOPPING_PATIENCE = 8
# wall-time LR schedule: linear warmup -> constant -> linear warmdown to FINAL*base
S2_TIME_BUDGET = 2400.0      # stage-2 wall-clock seconds the schedule spans
S2_WARMUP_RATIO = 0.05
S2_WARMDOWN_RATIO = 0.30
S2_FINAL_LR_FRAC = 0.01

# encoder output positions: conv_tower downsamples by 128×
SEQ_LEN = PROMOTER_LENGTH if MODE == "promoter_only" else SEQUENCE_LENGTH
N_ENC_POSITIONS = math.ceil(SEQ_LEN / 128)

_dir = os.path.dirname(os.path.abspath(__file__))
STAGE1_CHECKPOINT_PATH = os.path.join(_dir, f"best_stage1_{TISSUE}_{MODE}.pkl")  # probing head
CHECKPOINT_PATH = os.path.join(_dir, f"best_{TISSUE}_{MODE}.pkl")                # full fine-tune
METRICS_CSV_PATH = os.path.join(_dir, f"metrics_{TISSUE}_{MODE}.csv")            # learning curves


# wall-time LR multiplier: warmup -> constant -> linear warmdown (stage 2)
def lr_multiplier(progress: float) -> float:
    if progress < S2_WARMUP_RATIO:
        return progress / S2_WARMUP_RATIO if S2_WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - S2_WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / S2_WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * S2_FINAL_LR_FRAC


def _log_metrics(stage: int, epoch: int, train_loss: float, val_loss: float,
                 val_pr: float, head_lr: float = 0.0) -> None:
    new = not os.path.exists(METRICS_CSV_PATH)
    with open(METRICS_CSV_PATH, "a") as f:
        if new:
            f.write("stage,epoch,train_loss,val_loss,val_pearson,head_lr\n")
        f.write(f"{stage},{epoch},{train_loss:.6f},{val_loss:.6f},{val_pr:.6f},{head_lr:.3e}\n")


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


def test_eval(model, head) -> tuple[float, float]:
    """scipy Pearson + MSE on the held-out (gene-split) test set."""
    all_preds, all_tgts = [], []
    for tokens_np, targets_np in test_loader:
        enc_out = get_embeddings(model, jnp.array(tokens_np))
        all_preds.append(np.asarray(head(enc_out)))
        all_tgts.append(np.asarray(targets_np))
    preds_np = np.concatenate(all_preds)
    tgts_np = np.concatenate(all_tgts)
    r, _ = pearsonr(preds_np, tgts_np)
    return float(r), float(np.mean((preds_np - tgts_np) ** 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    if os.path.exists(METRICS_CSV_PATH):
        os.remove(METRICS_CSV_PATH)

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
    print(f"Backbone params: {n_backbone:,} | Head params: {n_head:,}")

    collate_fn = make_ntv3_collate_fn(tokenizer)

    # stage-1 train loader (light aug, small batch); val/test are unaugmented and shared
    train_loader_s1, val_loader, test_loader = create_dataloaders(
        data_dir=DATA_DIR, tissue=TISSUE, mode=MODE, batch_size=S1_BATCH_SIZE,
        reverse_complement=S1_REVERSE_COMPLEMENT, rc_prob=S1_RC_PROB,
        random_shift=S1_RANDOM_SHIFT, shift_prob=S1_SHIFT_PROB, max_shift=S1_MAX_SHIFT,
        collate_fn=collate_fn, num_workers=0,
    )

    eval_step_fn = make_eval_step(get_embeddings)

    # gc tricks: collect + freeze + disable to avoid 500ms stalls during training
    gc.collect()
    gc.freeze()
    gc.disable()

    # -------------------------------------------------------------------
    # Stage 1: frozen backbone, train head ("probing")
    # -------------------------------------------------------------------

    print(f"--- stage 1 (probing): frozen backbone, lr={S1_LR}, batch={S1_BATCH_SIZE}, "
          f"rc={S1_REVERSE_COMPLEMENT}, patience={S1_EARLY_STOPPING_PATIENCE} ---")

    head_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=S1_LR, weight_decay=S1_WEIGHT_DECAY),
    )
    head_opt = nnx.Optimizer(head, head_tx)
    train_step_s1 = make_train_step(get_embeddings, freeze_backbone=True)

    t_train_start = time.time()
    best_val_pearson = -float("inf")
    best_head_state = None
    best_backbone_state = None
    epochs_since_best = 0
    stage1_epochs_done = 0
    total_steps = 0

    for epoch in range(S1_MAX_EPOCHS):
        epoch_loss = 0.0
        epoch_batches = 0

        for tokens_np, targets_np in train_loader_s1:
            tokens = jnp.array(tokens_np)
            targets = jnp.array(targets_np)
            loss, _ = train_step_s1(model, head, head_opt, None, tokens, targets)
            loss_val = float(loss)
            if math.isnan(loss_val):
                raise RuntimeError(f"NaN loss at stage 1 epoch {epoch}")
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
        print(f"epoch {epoch:3d} {'*' if is_best else ' '} | train_loss: {avg_loss:.6f} | "
              f"val_loss: {val_loss:.4f} | val_r: {val_pr:.4f} (best: {best_val_pearson:.4f}) | "
              f"elapsed: {time.time() - t_train_start:.0f}s", flush=True)
        _log_metrics(1, epoch, avg_loss, val_loss, val_pr)

        if (epoch + 1) % 5 == 0:
            gc.collect()
        if epochs_since_best >= S1_EARLY_STOPPING_PATIENCE:
            print(f"early stopped after {stage1_epochs_done} epochs")
            break

    # restore best stage-1 head, eval test, save probing checkpoint
    if best_head_state is not None:
        nnx.update(head, best_head_state)
    stage1_test_pearson, stage1_test_mse = test_eval(model, head)
    with open(STAGE1_CHECKPOINT_PATH, "wb") as f:
        pickle.dump({
            "head_state": jax.tree.map(np.asarray, nnx.state(head, nnx.Param)),
            "backbone_state": None,  # frozen == original pretrained weights
            "val_pearson": best_val_pearson,
            "test_pearson": stage1_test_pearson,
            "test_mse": stage1_test_mse,
            "stage": 1,
            "config": {"tissue": TISSUE, "mode": MODE, "model_name": MODEL_NAME},
        }, f)
    print(f"stage 1 done: {stage1_epochs_done} epochs | "
          f"val_pearson={best_val_pearson:.6f} test_pearson={stage1_test_pearson:.6f} "
          f"test_mse={stage1_test_mse:.6f}")
    print(f"stage1_checkpoint: {STAGE1_CHECKPOINT_PATH}")

    # -------------------------------------------------------------------
    # Stage 2: unfrozen backbone, full fine-tune (wall-time LR schedule)
    # -------------------------------------------------------------------

    stage2_epochs_done = 0
    if RUN_STAGE2:
        # stage-2 train loader: heavier aug, larger batch
        train_loader_s2, _, _ = create_dataloaders(
            data_dir=DATA_DIR, tissue=TISSUE, mode=MODE, batch_size=S2_BATCH_SIZE,
            reverse_complement=S2_REVERSE_COMPLEMENT, rc_prob=S2_RC_PROB,
            random_shift=S2_RANDOM_SHIFT, shift_prob=S2_SHIFT_PROB, max_shift=S2_MAX_SHIFT,
            collate_fn=collate_fn, num_workers=0,
        )

        # inject_hyperparams lets us mutate the LR per-step for the wall-time schedule
        head_opt = nnx.Optimizer(head, optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.inject_hyperparams(optax.adamw)(learning_rate=S2_BASE_LR, weight_decay=S2_WEIGHT_DECAY),
        ))
        backbone_opt = nnx.Optimizer(model, optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=S2_BASE_LR * S2_BACKBONE_LR_SCALE, weight_decay=S2_WEIGHT_DECAY),
        ))
        train_step_s2 = make_train_step(get_embeddings, freeze_backbone=False)

        print(f"--- stage 2 (full): unfrozen backbone | base_lr={S2_BASE_LR:.1e} "
              f"backbone_scale={S2_BACKBONE_LR_SCALE} batch={S2_BATCH_SIZE} rc={S2_REVERSE_COMPLEMENT} "
              f"wd={S2_WEIGHT_DECAY} | wall-time schedule budget={S2_TIME_BUDGET:.0f}s "
              f"(warmup={S2_WARMUP_RATIO}, warmdown={S2_WARMDOWN_RATIO}, final={S2_FINAL_LR_FRAC}) "
              f"| max_epochs={S2_MAX_EPOCHS} patience={S2_EARLY_STOPPING_PATIENCE} ---")

        s2_epochs_since_best = 0
        t_stage2_start = time.time()
        cur_head_lr = S2_BASE_LR
        timed_out = False

        for epoch in range(S2_MAX_EPOCHS):
            epoch_loss = 0.0
            epoch_batches = 0

            for tokens_np, targets_np in train_loader_s2:
                tokens = jnp.array(tokens_np)
                targets = jnp.array(targets_np)

                # per-step wall-time LR schedule (head) + discriminative backbone LR
                progress = (time.time() - t_stage2_start) / S2_TIME_BUDGET
                cur_head_lr = S2_BASE_LR * max(lr_multiplier(progress), 0.0)
                head_opt.opt_state[1].hyperparams["learning_rate"].value = jnp.array(cur_head_lr)
                backbone_opt.opt_state[1].hyperparams["learning_rate"].value = jnp.array(
                    cur_head_lr * S2_BACKBONE_LR_SCALE)

                loss, _ = train_step_s2(model, head, head_opt, backbone_opt, tokens, targets)
                loss_val = float(loss)
                if math.isnan(loss_val):
                    raise RuntimeError(f"NaN loss at stage 2 epoch {epoch}")
                epoch_loss += loss_val
                epoch_batches += 1
                total_steps += 1

                if (time.time() - t_stage2_start) >= S2_TIME_BUDGET:
                    timed_out = True
                    break

            val_loss, val_pr = evaluate(model, head, val_loader, eval_step_fn)
            is_best = val_pr > best_val_pearson
            if is_best:
                best_val_pearson = val_pr
                best_head_state = jax.tree.map(jnp.copy, nnx.state(head))
                best_backbone_state = jax.tree.map(jnp.copy, nnx.state(model))
                s2_epochs_since_best = 0
            else:
                s2_epochs_since_best += 1

            stage2_epochs_done = epoch + 1
            avg_loss = epoch_loss / max(epoch_batches, 1)
            print(f"epoch {stage1_epochs_done + epoch:3d} {'*' if is_best else ' '} | "
                  f"train_loss: {avg_loss:.6f} | val_loss: {val_loss:.4f} | "
                  f"val_r: {val_pr:.4f} (best: {best_val_pearson:.4f}) | lr: {cur_head_lr:.2e} | "
                  f"t: {time.time() - t_stage2_start:.0f}/{S2_TIME_BUDGET:.0f}s", flush=True)
            _log_metrics(2, stage1_epochs_done + epoch, avg_loss, val_loss, val_pr, cur_head_lr)

            if (epoch + 1) % 5 == 0:
                gc.collect()
            if timed_out:
                print(f"stage 2 wall-time budget reached after {stage2_epochs_done} epochs")
                break
            if s2_epochs_since_best >= S2_EARLY_STOPPING_PATIENCE:
                print(f"stage 2 early stopped after {stage2_epochs_done} epochs")
                break

        print(f"stage 2 done: {stage2_epochs_done} epochs, best val_pearson: {best_val_pearson:.6f}")

    gc.enable()

    # restore overall-best weights for final eval + checkpoint
    if best_head_state is not None:
        nnx.update(head, best_head_state)
    if best_backbone_state is not None:
        nnx.update(model, best_backbone_state)

    val_loss, val_pr = evaluate(model, head, val_loader, eval_step_fn)
    test_pearson, test_mse = test_eval(model, head)

    backend = jax.default_backend()
    peak_vram_mb = (jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1024 / 1024
                    if backend == "gpu" else 0.0)

    print("---")
    print(f"val_pearson:      {best_val_pearson:.6f}")
    print(f"val_mse:          {val_loss:.6f}")
    print(f"test_pearson:     {test_pearson:.6f}")
    print(f"test_mse:         {test_mse:.6f}")
    print(f"stage1_test_pearson: {stage1_test_pearson:.6f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"tissue:           {TISSUE}")
    print(f"mode:             {MODE}")
    print(f"stage1_epochs:    {stage1_epochs_done}")
    print(f"stage2_epochs:    {stage2_epochs_done}")

    # final ("full fine-tune") checkpoint: head + fine-tuned backbone (if stage 2 ran)
    ckpt = {
        "head_state": jax.tree.map(np.asarray, nnx.state(head, nnx.Param)),
        "backbone_state": jax.tree.map(np.asarray, nnx.state(model, nnx.Param)) if RUN_STAGE2 else None,
        "val_pearson": best_val_pearson,
        "test_pearson": test_pearson,
        "test_mse": test_mse,
        "stage": 2 if RUN_STAGE2 else 1,
        "config": {"tissue": TISSUE, "mode": MODE, "model_name": MODEL_NAME},
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"checkpoint:       {CHECKPOINT_PATH}")
