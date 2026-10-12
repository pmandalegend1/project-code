#!/usr/bin/env python3
"""
eval_conformal.py -- test the conformal termination rule (N5) on real data.

Why this script exists
----------------------
`ConformalTerminator` is the one novel component that is genuinely ours: it
reads the opponent model's own predictive distribution to decide when another
revision round has stopped meaning anything. It has never been fitted on real
data. Every planner run so far left `qhat = None`, so `set_size()` returned 0
and `should_stop()` was always False -- the termination rule was in the code
and switched off in every result.

What is and is not evidence here
--------------------------------
Split conformal's coverage guarantee is distribution-free: under exchangeability
it holds by construction. So "empirical coverage came out near 1 - alpha" is a
CORRECTNESS CHECK on the implementation, not a finding, and it must not be
reported as though the method earned it.

The thing that is actually in question is whether the sets are USEFUL, and
whether varying their size per decision beats not bothering. Hence two
references, both calibrated to the same coverage on the same calibration split:

  persistence      the singleton set {the shape they are already in}. Size 1,
                   always. This is the bar the rest of the project uses, and
                   it is the honest one here: the marginal baseline below does
                   not know the opponent's current formation, so beating it is
                   easy and means little.
  marginal top-k   ignore the model entirely. Rank formations by their frequency
                   in the calibration set and take the smallest fixed k that
                   reaches 1 - alpha. Included only to show what "no model at
                   all" costs.
  model top-k      use the model's ranking but a FIXED set size: smallest k such
                   that the true label is in the model's top k at least
                   (1 - alpha) of the time. If conformal does not beat this,
                   the adaptivity -- the whole point of a conformal set -- is
                   adding nothing, and the terminator could be a constant.

THE BAR, fixed before the numbers:
  1. coverage within 2 standard errors of 1 - alpha            (correctness)
  2. conformal mean set size strictly smaller than model top-k,
     by more than the standard deviation across calibration/test partitions
                                                               (adaptivity earns its place)
If (2) fails, the honest conclusion is that a fixed top-k terminator is as good,
and the paper should say so.

Calibration must be data the model never saw. The shipped checkpoint is seed 0,
fold 0 of a GroupKFold(5) over match_id, which is deterministic, so fold 0's
held-out matches are recoverable exactly. Those matches are split again by
match into calibration and test, so no match appears in both.

    python eval_conformal.py --triples data/processed/triples.parquet \
                             --opponent models/opponent.pt
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
from ramfp.eval.metrics import conformal_qhat
from ramfp.models.opponent import OpponentCoBehaviorModel
from ramfp.train.train_opponent import predict
from ramfp.vocab import N_FORMATIONS, formation_id


def heldout_fold0(df, n_folds=5):
    """The exact rows the shipped checkpoint was not trained on."""
    gkf = GroupKFold(n_splits=n_folds)
    _, te_idx = next(iter(gkf.split(df, groups=df["match_id"].to_numpy())))
    return df.iloc[te_idx].reset_index(drop=True)


def calibrated_model_probs(model, df, cfg, device, params):
    """Mixture probabilities with the post-hoc calibration applied.

    The conformal set is {classes with p >= 1 - qhat}, so it reads the model's
    probabilities directly. An under-confident model spreads mass and produces
    wider sets at the same coverage; calibration is therefore not cosmetic here,
    it is upstream of the only thing this test measures.
    """
    from calibrate_opponent import mixture, raw_outputs
    F, R, C, Y = raw_outputs(model, df, cfg, device)
    return mixture(F, R, C, params["temperature"], params["platt_a"], params["platt_b"]), Y


def model_probs(model, df, cfg, device):
    dl = DataLoader(TripleDataset(df, cfg.encoder.seq_len, mirror=False, knn=cfg.encoder.knn),
                    batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate)
    _, mixture, _, y, _ = predict(model, dl, device)
    return mixture, y


def conformal_eval(cal_p, cal_y, te_p, te_y, alpha):
    qhat = conformal_qhat(cal_p, cal_y, alpha)
    in_set = (1.0 - te_p) <= qhat                      # (n, K) boolean
    sizes = in_set.sum(1)
    covered = in_set[np.arange(len(te_y)), te_y]
    return {
        "qhat": float(qhat),
        "coverage": float(covered.mean()),
        "mean_size": float(sizes.mean()),
        "median_size": float(np.median(sizes)),
        "p_singleton": float((sizes == 1).mean()),
        "p_full_vocab": float((sizes >= te_p.shape[1]).mean()),
        "p_empty": float((sizes == 0).mean()),
        "sizes": sizes,
    }


def persistence_eval(te_y, te_stay):
    """Singleton set containing the opponent's current shape."""
    return {"k": 1, "coverage": float((te_y == te_stay).mean()), "mean_size": 1.0}


def topk_eval(cal_p, cal_y, te_p, te_y, alpha, ranking):
    """Smallest fixed k reaching 1-alpha on calibration, measured on test.

    ranking='model'    : order classes by the model's own probability per row
    ranking='marginal' : one global order, by class frequency in calibration
    """
    K = cal_p.shape[1]
    if ranking == "model":
        cal_rank = np.argsort(-cal_p, axis=1)
        te_rank = np.argsort(-te_p, axis=1)
    else:
        counts = np.bincount(cal_y, minlength=K)
        order = np.argsort(-counts)
        cal_rank = np.tile(order, (len(cal_y), 1))
        te_rank = np.tile(order, (len(te_y), 1))

    cal_pos = np.argmax(cal_rank == cal_y[:, None], axis=1)      # 0-based rank of truth
    k = int(np.quantile(cal_pos, 1 - alpha, method="higher")) + 1
    k = min(k, K)
    te_pos = np.argmax(te_rank == te_y[:, None], axis=1)
    return {"k": k, "coverage": float((te_pos < k).mean()), "mean_size": float(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/processed/triples.parquet")
    ap.add_argument("--opponent", default="models/opponent.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--partitions", type=int, default=20,
                    help="calibration/test splits of the held-out matches")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--calibration", default="results/calibration.json",
                    help="post-hoc calibration params; pass '' to use raw probabilities")
    ap.add_argument("--out", default="results/conformal.json")
    args = ap.parse_args()

    cfg = Config.load(args.config) if hasattr(Config, "load") else Config()
    df = pd.read_parquet(args.triples)
    held = heldout_fold0(df, cfg.train.n_folds)
    print(f"{len(df):,} triples | fold-0 held-out: {len(held):,} rows "
          f"across {held['match_id'].nunique():,} matches (model never saw these)")

    model = OpponentCoBehaviorModel.load(args.opponent, map_location=args.device).to(args.device)
    if args.calibration and os.path.exists(args.calibration):
        params = json.load(open(args.calibration))
        probs, y = calibrated_model_probs(model, held, cfg, args.device, params)
        print(f"using calibrated probabilities (T={params['temperature']:.3f}, "
              f"a={params['platt_a']:.3f}, b={params['platt_b']:+.3f})")
    else:
        probs, y = model_probs(model, held, cfg, args.device)
        print("using raw probabilities (run calibrate_opponent.py for the calibrated ones)")
    matches = held["match_id"].to_numpy()
    stay = np.array([formation_id(int(c)) for c in held["opp_formation"]])
    print(f"predictive distributions: {probs.shape}, {N_FORMATIONS} formations\n")

    results = {}
    rng = np.random.default_rng(0)
    uniq = np.unique(matches)

    for alpha in args.alphas:
        rows = {"conformal": [], "model_topk": [], "persistence": [], "marginal_topk": []}
        ks = {"model_topk": [], "marginal_topk": []}
        for _ in range(args.partitions):
            perm = rng.permutation(uniq)
            cal_m = set(perm[: len(perm) // 2])
            cal = np.array([m in cal_m for m in matches])
            if cal.sum() < 30 or (~cal).sum() < 30:
                continue
            c = conformal_eval(probs[cal], y[cal], probs[~cal], y[~cal], alpha)
            rows["conformal"].append((c["coverage"], c["mean_size"]))
            for name, ranking in (("model_topk", "model"), ("marginal_topk", "marginal")):
                t = topk_eval(probs[cal], y[cal], probs[~cal], y[~cal], alpha, ranking)
                rows[name].append((t["coverage"], t["mean_size"]))
                ks[name].append(t["k"])
            pe = persistence_eval(y[~cal], stay[~cal])
            rows["persistence"].append((pe["coverage"], pe["mean_size"]))

        print(f"--- alpha = {alpha:.2f}  (target coverage {1 - alpha:.2f}) ---")
        print(f"{'method':<18}{'coverage':>20}{'mean set size':>22}")
        summ = {}
        for name in ("conformal", "model_topk", "persistence", "marginal_topk"):
            a = np.array(rows[name])
            summ[name] = {
                "coverage_mean": float(a[:, 0].mean()), "coverage_std": float(a[:, 0].std()),
                "size_mean": float(a[:, 1].mean()), "size_std": float(a[:, 1].std()),
            }
            if name in ks:
                summ[name]["k_mode"] = int(np.bincount(ks[name]).argmax())
            print(f"{name:<18}{a[:,0].mean():>12.4f} +- {a[:,0].std():.4f}"
                  f"{a[:,1].mean():>15.2f} +- {a[:,1].std():.2f}")

        n_test = int((~cal).sum())
        se = float(np.sqrt((1 - alpha) * alpha / max(n_test, 1)))
        cov_gap = abs(summ["conformal"]["coverage_mean"] - (1 - alpha))
        check1 = cov_gap <= 2 * se
        size_gap = summ["model_topk"]["size_mean"] - summ["conformal"]["size_mean"]
        noise = max(summ["conformal"]["size_std"], summ["model_topk"]["size_std"])
        check2 = size_gap > noise

        print(f"  (1) coverage within 2 SE of {1 - alpha:.2f}? "
              f"gap {cov_gap:.4f} vs 2 SE {2 * se:.4f} -> {'PASS' if check1 else 'FAIL'}")
        print(f"  (2) smaller than fixed model top-k by more than noise? "
              f"gap {size_gap:+.2f} vs noise {noise:.2f} -> {'PASS' if check2 else 'FAIL'}")

        # a set of average size below 1 contains no formation at all on some
        # decisions. An empty set is not a small set, it is a refusal, and it
        # must not be allowed to win a size comparison.
        if summ["conformal"]["size_mean"] < 1.0:
            print(f"      NOTE: mean size {summ['conformal']['size_mean']:.2f} < 1, so some "
                  f"prediction sets are EMPTY. Any size advantage here is bought by "
                  f"refusing to predict, not by being sharper.")

        pers = summ["persistence"]
        print(f"  (3) against persistence (size 1.00, coverage "
              f"{pers['coverage_mean']:.4f}): conformal buys "
              f"{summ['conformal']['coverage_mean'] - pers['coverage_mean']:+.4f} coverage "
              f"for {summ['conformal']['size_mean'] - 1.0:+.2f} formations")
        print()
        summ["coverage_check_passed"] = bool(check1)
        summ["adaptivity_check_passed"] = bool(check2)
        summ["n_test_rows"] = n_test
        results[f"alpha_{alpha}"] = summ

    # set-size distribution at alpha=0.10, to ground the revision-loop threshold
    a10 = 0.10
    perm = rng.permutation(uniq)
    cal_m = set(perm[: len(perm) // 2])
    cal = np.array([m in cal_m for m in matches])
    c = conformal_eval(probs[cal], y[cal], probs[~cal], y[~cal], a10)
    sizes = c["sizes"]
    print("set-size distribution at alpha = 0.10 (one partition)")
    for k in range(1, min(N_FORMATIONS, 10) + 1):
        print(f"  size <= {k:<3} {100 * (sizes <= k).mean():6.1f}% of decisions")
    print(f"\n  the revision loop stops when the set exceeds `threshold` "
          f"(configured 6): that fires on {100 * (sizes > 6).mean():.1f}% of decisions")
    results["set_size_distribution_alpha_0.10"] = {
        f"le_{k}": float((sizes <= k).mean()) for k in range(1, N_FORMATIONS + 1)
    }
    results["threshold_6_fires_pct"] = float((sizes > 6).mean())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
