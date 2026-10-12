#!/usr/bin/env python3
"""
stack_ensemble.py -- combine GNN and ridge predictions with learned weights.

Uses build_dataset() from train_world_model.py so the graphs are built exactly
the way they were during training. Adds diagnostics so any failure is visible.

    python stack_ensemble.py --config configs/world_model.yaml \
                             --checkpoint checkpoints/world_model_best_v4.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge, LinearRegression
from torch_geometric.data import Batch

sys.path.insert(0, ".")
from models.world_model.bipartite_gnn import BipartiteFormationGNN
from baseline_world_model import temporal_split, featurise
from train_world_model import build_dataset, TARGETS


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world_model.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="results/stack_ensemble.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    norm = ckpt["norm_stats"]
    cfg_ck = ckpt.get("cfg", cfg)

    w = pd.read_parquet(cfg["stable_windows_path"])
    rms = pd.read_parquet(cfg["role_matchup_stats_path"])
    sty = pd.read_parquet(cfg["style_clusters_path"])

    with open(cfg["formation_vocab_path"]) as f:
        vocab_list = [int(c) for c in json.load(f)["vocab"]]
    formation_vocab = {c: i for i, c in enumerate(vocab_list)}

    train_df, test_df = temporal_split(w, cfg)
    ppda_mean = train_df["ppda"].mean()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["ppda"] = train_df["ppda"].fillna(ppda_mean)
    test_df["ppda"] = test_df["ppda"].fillna(ppda_mean)

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Use the exact graph builder the training used
    print("Building val graphs (via train_world_model.build_dataset) ...")
    val_graphs, val_targets = build_dataset(test_df, rms, sty, formation_vocab, cfg)
    print(f"  {len(val_graphs):,} val graphs built")
    if len(val_graphs) == 0:
        print("FAIL: no val graphs. Check formation vocab coverage.")
        return 1

    # Raw targets in original units
    val_y = np.array([t.numpy() for t in val_targets])

    # ---- GNN predictions ----
    model = BipartiteFormationGNN(
        node_feat_dim=cfg_ck.get("node_feat_dim", 80),
        node_dim=cfg_ck.get("node_dim", 128),
        gat_heads=cfg_ck.get("gnn_heads", 4),
        gnn_layers=cfg_ck.get("gnn_layers", 3),
        output_dim=4,
        edge_feat_dim=2,
        n_formations=len(vocab_list),
    )
    try:
        model.load_state_dict(ckpt["model_state"])
    except Exception as e:
        print(f"WARN: strict load failed ({e}); trying without n_formations")
        model = BipartiteFormationGNN(
            node_feat_dim=cfg_ck.get("node_feat_dim", 80),
            node_dim=cfg_ck.get("node_dim", 128),
            gat_heads=cfg_ck.get("gnn_heads", 4),
            gnn_layers=cfg_ck.get("gnn_layers", 3),
            output_dim=4,
            edge_feat_dim=2,
        )
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Running GNN forward ...")
    gnn_preds = np.zeros((len(val_graphs), 4))
    bs = 128
    with torch.no_grad():
        for s in range(0, len(val_graphs), bs):
            e = min(s + bs, len(val_graphs))
            batch = Batch.from_data_list(val_graphs[s:e])
            out = model(batch).view(e - s, -1).numpy()
            gnn_preds[s:e] = out

    # Denormalise GNN predictions to original units
    for j, t in enumerate(TARGETS):
        gnn_preds[:, j] = gnn_preds[:, j] * norm[t]["std"] + norm[t]["mean"]

    # ---- Ridge predictions ----
    print("Training ridge on the training split ...")
    sty_map = dict(zip(sty["team_id"].astype(float), sty["style_cluster"].astype(int)))
    X_tr = featurise(train_df, vocab_list, sty_map)
    X_va = featurise(test_df, vocab_list, sty_map)

    # Only keep the rows corresponding to graphs that actually built
    # (vocab filter may have dropped some)
    valid_mask = test_df["home_formation"].astype(int).isin(set(vocab_list)) & \
                 test_df["away_formation"].astype(int).isin(set(vocab_list))
    X_va = X_va[valid_mask.values]
    if X_va.shape[0] != gnn_preds.shape[0]:
        print(f"  aligning: X_va {X_va.shape[0]} vs gnn {gnn_preds.shape[0]}")

    ridge_preds = np.zeros_like(gnn_preds)
    for j, t in enumerate(TARGETS):
        y_tr = train_df[t].astype(float).values
        y_tr = np.where(np.isfinite(y_tr), y_tr, ppda_mean) if t == "ppda" else y_tr
        ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
        ridge_preds[:, j] = ridge.predict(X_va[:gnn_preds.shape[0]])

    # ---- Stack ----
    # Split val set in half: fit the combiner on the first half, evaluate on
    # the second. Otherwise the combiner weights are fitted and evaluated on
    # the same rows, which is a mild form of leakage.
    n = val_y.shape[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    fit_idx, eval_idx = perm[: n // 2], perm[n // 2:]

    header = f"\n{'target':<18}{'ridge r':>10}{'gnn r':>10}{'stack r':>10}{'stack-ridge':>14}"
    print(header); print("-" * len(header))
    print(f"  (combiner fit on {len(fit_idx):,} rows, evaluated on {len(eval_idx):,} rows)")
    results = {}
    for j, t in enumerate(TARGETS):
        yt_full = val_y[:, j]
        ok_full = np.isfinite(yt_full) & np.isfinite(gnn_preds[:, j]) & np.isfinite(ridge_preds[:, j])
        # fit combiner on fit_idx rows only
        fit_ok = np.zeros(n, dtype=bool); fit_ok[fit_idx] = True
        fit_rows = ok_full & fit_ok
        X_fit = np.column_stack([ridge_preds[fit_rows, j], gnn_preds[fit_rows, j]])
        y_fit = yt_full[fit_rows]
        comb = LinearRegression().fit(X_fit, y_fit)

        # evaluate on eval_idx rows only
        eval_ok = np.zeros(n, dtype=bool); eval_ok[eval_idx] = True
        eval_rows = ok_full & eval_ok
        yt = yt_full[eval_rows]
        r = ridge_preds[eval_rows, j]
        g = gnn_preds[eval_rows, j]
        X_eval = np.column_stack([r, g])
        stacked = comb.predict(X_eval)
        r_ridge = pearson(yt, r)
        r_gnn = pearson(yt, g)
        r_stack = pearson(yt, stacked)
        d = r_stack - r_ridge
        mark = " OK" if d > 0 else (" NO" if d < -0.005 else " =")
        print(f"{t:<18}{r_ridge:>10.4f}{r_gnn:>10.4f}{r_stack:>10.4f}{d:>+11.4f}{mark}")
        results[t] = {
            "ridge_r": r_ridge, "gnn_r": r_gnn, "stack_r": r_stack,
            "improvement": d,
            "weights": {"ridge": float(comb.coef_[0]), "gnn": float(comb.coef_[1])},
        }
    wins = sum(1 for v in results.values() if v["improvement"] > 0)
    print(f"\n{wins}/{len(TARGETS)} targets where stack beats ridge")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())