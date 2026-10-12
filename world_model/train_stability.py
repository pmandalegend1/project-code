#!/usr/bin/env python3
"""
train_stability.py -- train the stability critic (paper Eq. 25) and decide,
against a bar fixed before the numbers arrive, whether it is worth using.

The bar, written down first
---------------------------
The critic currently in the inference loop is the placeholder

    score = -structural_distance(f_src, f_tgt) / 100

i.e. structural distance alone, with no learning. So the honest reference is
not chance and not the base rate: it is a logistic regression on
delta_struct alone. The MLP earns its place ONLY IF its cross-validated
AUROC beats that logistic by more than the fold-to-fold standard deviation.
If it does not, the placeholder is already as good as the learned model, and
the paper should say the learned stability critic was tested and did not
improve on structural distance.

Accuracy is not reported. With a base rate near 0.16, predicting "never
concedes" scores 0.84 and means nothing. Read average precision against the
base rate.

    python train_stability.py --labels data/processed/stability_labels.parquet
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from critics import N_FORMATIONS, StabilityMLP, structural_distance


def build_vocab(df):
    codes = sorted(set(df["f_src"]) | set(df["f_tgt"]))
    return {c: i for i, c in enumerate(codes)}


def features(df, vocab):
    src = np.array([vocab[c] for c in df["f_src"]], dtype=np.int64)
    tgt = np.array([vocab[c] for c in df["f_tgt"]], dtype=np.int64)
    delta = np.array([structural_distance(a, b) for a, b in zip(df["f_src"], df["f_tgt"])],
                     dtype=np.float32)
    return src, tgt, delta


def train_fold(src, tgt, delta, y, tr, te, epochs, seed, device):
    torch.manual_seed(seed)
    n_vocab = int(max(src.max(), tgt.max())) + 1
    model = StabilityMLP(n_formations=max(n_vocab, N_FORMATIONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    d_mu, d_sd = delta[tr].mean(), delta[tr].std() or 1.0
    dz = (delta - d_mu) / d_sd

    xs = torch.from_numpy(src[tr]).to(device)
    xt = torch.from_numpy(tgt[tr]).to(device)
    xd = torch.from_numpy(dz[tr]).to(device)
    yt = torch.from_numpy(y[tr].astype(np.float32)).to(device)

    n_pos, n_neg = float(yt.sum()), float(len(yt) - yt.sum())
    pos_weight = torch.tensor(n_neg / n_pos, device=device) if n_pos > 0 and n_neg > 0 else None
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xs, xt, xd), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(
            torch.from_numpy(src[te]).to(device),
            torch.from_numpy(tgt[te]).to(device),
            torch.from_numpy(dz[te]).to(device),
        )).cpu().numpy()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/stability_labels.parquet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/stability_critic.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.labels)
    y = df["conceded"].values.astype(int)
    base = y.mean()
    print(f"{len(df):,} transitions | base rate {base:.4f} | "
          f"{df['match_id'].nunique():,} matches | grouped CV by match\n")

    vocab = build_vocab(df)
    src, tgt, delta = features(df, vocab)
    groups = df["match_id"].values

    gkf = GroupKFold(n_splits=args.folds)
    rows = {"mlp": [], "logistic_delta_struct": [], "delta_struct_raw": []}

    for seed in range(args.seeds):
        for tr, te in gkf.split(src, y, groups):
            if len(np.unique(y[te])) < 2:
                continue
            p_mlp = train_fold(src, tgt, delta, y, tr, te, args.epochs, seed, "cpu")
            rows["mlp"].append((roc_auc_score(y[te], p_mlp),
                                average_precision_score(y[te], p_mlp)))

            if seed == 0:
                lr = LogisticRegression(max_iter=1000, class_weight="balanced")
                lr.fit(delta[tr].reshape(-1, 1), y[tr])
                p_lr = lr.predict_proba(delta[te].reshape(-1, 1))[:, 1]
                rows["logistic_delta_struct"].append(
                    (roc_auc_score(y[te], p_lr), average_precision_score(y[te], p_lr)))
                # the placeholder critic itself: larger distance -> worse, so
                # concession risk is taken as increasing in distance
                rows["delta_struct_raw"].append(
                    (roc_auc_score(y[te], delta[te]), average_precision_score(y[te], delta[te])))

    print(f"{'model':<26}{'AUROC':>16}{'avg precision':>18}")
    summary = {}
    for name, vals in rows.items():
        a = np.array(vals)
        summary[name] = {
            "auroc_mean": float(a[:, 0].mean()), "auroc_std": float(a[:, 0].std()),
            "ap_mean": float(a[:, 1].mean()), "ap_std": float(a[:, 1].std()),
            "n_folds": len(a),
        }
        print(f"{name:<26}{a[:,0].mean():>9.4f} +- {a[:,0].std():.4f}"
              f"{a[:,1].mean():>11.4f} +- {a[:,1].std():.4f}")
    print(f"{'base rate':<26}{'--':>16}{base:>13.4f}")

    mlp, ref = summary["mlp"], summary["logistic_delta_struct"]
    gap = mlp["auroc_mean"] - ref["auroc_mean"]
    noise = max(mlp["auroc_std"], ref["auroc_std"])
    print(f"\nbar: MLP AUROC must beat logistic-on-delta_struct by more than "
          f"fold noise ({noise:.4f})")
    print(f"gap = {gap:+.4f}")
    verdict = "USE the learned critic" if gap > noise else \
              "KEEP the structural-distance placeholder; the learned critic does not improve on it"
    print(f"verdict: {verdict}")

    summary["base_rate"] = float(base)
    summary["gap"] = float(gap)
    summary["fold_noise"] = float(noise)
    summary["verdict"] = verdict
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
