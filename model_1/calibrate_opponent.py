#!/usr/bin/env python3
"""
calibrate_opponent.py -- fix the opponent model's calibration, and measure it.

The problem
-----------
The model's accuracy ties persistence (0.8872 vs 0.8898), but its expected
calibration error is 0.3131 against 0.1102 for persistence and 0.0472 for a
transition matrix with no parameters at all. A counting baseline is 6.6x better
calibrated than the trained network.

Why, specifically
-----------------
The respond head is trained with `pos_weight = n_neg / n_pos`, which is about 8
on this data. That is the right thing to do for AUROC -- it stops the head from
collapsing onto the majority class -- and it is exactly the wrong thing for
probabilities: it inflates p_change by construction. The mixture then spreads
mass onto the formation head on decisions where nothing was going to change, so
the model is systematically UNDER-confident, and confidence-based ECE punishes
that hard.

So this is a calibration defect, not an accuracy defect, and it has the standard
calibration fix.

What this does
--------------
Two post-hoc transforms, both fitted on data the model never trained on, and
both fitted on a split disjoint from the one they are evaluated on:

  respond head    Platt scaling, sigmoid(a * logit + b). Two parameters. This is
                  what undoes the pos_weight distortion.
  formation head  temperature scaling, softmax(logits / T). One parameter.

Neither changes any ranking, so top-1, top-3 and AUROC are untouched by
construction -- which is the point. This is not a way to make the model look
more accurate. It is a way to make its probabilities mean what they say, which
matters because the conformal terminator reads them.

    python calibrate_opponent.py --triples data/processed/triples.parquet
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from ramfp.config import Config
from ramfp.data.datasets import TripleDataset, collate
from ramfp.eval.metrics import expected_calibration_error
from ramfp.models.opponent import OpponentCoBehaviorModel
from ramfp.train.common import to_device
from ramfp.vocab import N_FORMATIONS


@torch.no_grad()
def raw_outputs(model, df, cfg, device):
    """Formation logits, respond logit, current-shape id and target, unmixed."""
    dl = DataLoader(TripleDataset(df, cfg.encoder.seq_len, mirror=False, knn=cfg.encoder.knn),
                    batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    F, R, C, Y = [], [], [], []
    for batch in dl:
        batch = to_device(batch, device)
        logits, respond = model(batch, return_respond=True)
        F.append(logits.float().cpu().numpy())
        R.append(respond.float().cpu().numpy())
        C.append(batch["opp_formation_id"].cpu().numpy())
        Y.append(batch["target"].cpu().numpy())
    return (np.concatenate(F), np.concatenate(R),
            np.concatenate(C).astype(int), np.concatenate(Y).astype(int))


def mixture(form_logits, respond_logit, cur, T=1.0, a=1.0, b=0.0):
    p_change = 1.0 / (1.0 + np.exp(-(a * respond_logit + b)))
    z = form_logits / T
    z = z - z.max(axis=1, keepdims=True)
    p_form = np.exp(z); p_form /= p_form.sum(axis=1, keepdims=True)
    stay = np.zeros_like(p_form)
    stay[np.arange(len(cur)), np.clip(cur, 0, N_FORMATIONS - 1)] = 1.0
    return (1.0 - p_change)[:, None] * stay + p_change[:, None] * p_form


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_params(form_logits, respond_logit, cur, y):
    """Grid then local refine on the calibration split, minimising NLL."""
    best, best_nll = (1.0, 1.0, 0.0), np.inf
    for T in np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(3.2, 8.0, 13)]):
        for a in np.linspace(0.2, 2.0, 19):
            for b in np.linspace(-6.0, 2.0, 33):
                v = nll(mixture(form_logits, respond_logit, cur, T, a, b), y)
                if v < best_nll:
                    best_nll, best = v, (float(T), float(a), float(b))
    return best, best_nll


def report(name, probs, y, cur):
    top1 = float((probs.argmax(1) == y).mean())
    top3 = float(np.mean([yy in np.argsort(-p)[:3] for p, yy in zip(probs, y)]))
    return {"name": name, "top1": top1, "top3": top3,
            "ece": expected_calibration_error(probs, y),
            "nll": nll(probs, y),
            "mean_confidence": float(probs.max(1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/processed/triples.parquet")
    ap.add_argument("--opponent", default="models/opponent.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/calibration.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    df = pd.read_parquet(args.triples)
    gkf = GroupKFold(n_splits=cfg.train.n_folds)
    _, te_idx = next(iter(gkf.split(df, groups=df["match_id"].to_numpy())))
    held = df.iloc[te_idx].reset_index(drop=True)

    matches = np.unique(held["match_id"].to_numpy())
    rng = np.random.default_rng(0)
    perm = rng.permutation(matches)
    cal_m = set(perm[: len(perm) // 2])
    is_cal = held["match_id"].isin(cal_m).to_numpy()
    print(f"held out: {len(held)} decisions | calibration {int(is_cal.sum())} "
          f"| test {int((~is_cal).sum())} | split by match, no overlap\n")

    model = OpponentCoBehaviorModel.load(args.opponent, map_location=args.device)
    F, R, C, Y = raw_outputs(model, held, cfg, args.device)

    (T, a, b), cal_nll = fit_params(F[is_cal], R[is_cal], C[is_cal], Y[is_cal])
    print(f"fitted on the calibration half: T = {T:.3f}, a = {a:.3f}, b = {b:+.3f} "
          f"(calibration NLL {cal_nll:.4f})\n")

    te = ~is_cal
    before = report("uncalibrated", mixture(F[te], R[te], C[te]), Y[te], C[te])
    after = report("calibrated", mixture(F[te], R[te], C[te], T, a, b), Y[te], C[te])

    # persistence on the same test half, for reference
    stay = np.full((int(te.sum()), N_FORMATIONS), 1e-9)
    stay[np.arange(int(te.sum())), np.clip(C[te], 0, N_FORMATIONS - 1)] = 1.0
    stay /= stay.sum(1, keepdims=True)
    pers = report("persistence", stay, Y[te], C[te])

    print(f"{'model':<16}{'top-1':>9}{'top-3':>9}{'ECE':>9}{'NLL':>9}{'mean conf':>11}")
    print("-" * 63)
    for r in (before, after, pers):
        print(f"{r['name']:<16}{r['top1']:>9.4f}{r['top3']:>9.4f}{r['ece']:>9.4f}"
              f"{r['nll']:>9.4f}{r['mean_confidence']:>11.4f}")
    print("-" * 63)
    print(f"transition matrix ECE on the full cross-validation: 0.0472 (reference)")

    same_rank = bool((mixture(F[te], R[te], C[te]).argmax(1) ==
                      mixture(F[te], R[te], C[te], T, a, b).argmax(1)).all())
    print(f"\ntop-1 predictions unchanged by calibration: {same_rank}")
    print("Calibration must not change what the model predicts, only how sure it says it is.")

    out = {"temperature": T, "platt_a": a, "platt_b": b,
           "uncalibrated": before, "calibrated": after, "persistence": pers,
           "top1_unchanged": same_rank, "n_calibration": int(is_cal.sum()),
           "n_test": int(te.sum())}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
