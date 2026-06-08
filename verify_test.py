"""Independent verification: reload a champion checkpoint and recompute TEST Pearson.

Confirms the reported test_pearson is genuine held-out test-set performance.
Loads head + fine-tuned backbone from the pickle, evals on the test split.
"""
import os, pickle, sys
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from scipy.stats import pearsonr

from nucleotide_transformer_v3.pretrained import get_pretrained_ntv3_model
from data import PROMOTER_LENGTH, SEQUENCE_LENGTH, create_dataloaders, make_ntv3_collate_fn
from finetune import MPRAHead, get_embeddings, N_ENC_POSITIONS

CKPTS = {
    "leaf":  ("best_leaf_combined_tuned.pkl",  0.885400),  # (file, reported test_pearson)
    "proto": ("best_proto_combined_tuned.pkl", 0.874500),
}

print("Loading pretrained NTv3...")
model, tokenizer, config = get_pretrained_ntv3_model("NTv3_650M_pre", use_bfloat16=True)
encoder_dim = config.embed_dim
collate = make_ntv3_collate_fn(tokenizer)

for tissue, (ckpt_file, reported) in CKPTS.items():
    d = pickle.load(open(ckpt_file, "rb"))
    assert d["config"]["tissue"] == tissue and d["config"]["mode"] == "combined"

    # rebuild head + load weights
    head = MPRAHead(n_positions=N_ENC_POSITIONS, encoder_dim=encoder_dim,
                    hidden_size=1024, dropout=0.2, rngs=nnx.Rngs(0))
    nnx.update(head, jax.tree.map(jnp.asarray, d["head_state"]))
    assert d["backbone_state"] is not None, "tuned ckpt must carry fine-tuned backbone"
    nnx.update(model, jax.tree.map(jnp.asarray, d["backbone_state"]))
    head.eval()  # deterministic inference (dropout off)

    _, _, test_loader = create_dataloaders(
        data_dir="./data", tissue=tissue, batch_size=256, mode="combined",
        reverse_complement=False, random_shift=False, collate_fn=collate, num_workers=0,
    )

    preds, tgts = [], []
    for tokens_np, targets_np in test_loader:
        enc = get_embeddings(model, jnp.array(tokens_np))
        preds.append(np.asarray(head(enc))); tgts.append(np.asarray(targets_np))
    preds = np.concatenate(preds); tgts = np.concatenate(tgts)
    r, _ = pearsonr(preds, tgts)
    mse = float(np.mean((preds - tgts) ** 2))
    print(f"\n=== {tissue} | {ckpt_file} | n_test={len(preds)} ===")
    print(f"  reported test_pearson (in-run, dropout ON): {reported:.4f}")
    print(f"  RECOMPUTED test_pearson (reload, dropout OFF): {r:.6f}")
    print(f"  recomputed test_mse: {mse:.6f}")
    print(f"  match within ~0.005? {'YES' if abs(r - reported) < 0.01 else 'NO -- INVESTIGATE'}")
