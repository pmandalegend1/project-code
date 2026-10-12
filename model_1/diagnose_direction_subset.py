#!/usr/bin/env python3
"""
diagnose_direction_subset.py -- the direction test, done where it can be fair.

Why a subset
------------
diagnose_direction.py showed the full-sample direction result (0.61) is an
artefact: the label is sign(forwards after - forwards before), and for most
opponents the current shape forces the answer (1 forward: can never go "more
defensive" on this axis in our vocabulary; 3 forwards: 58 of 60 went down).
Only opponents currently playing TWO forwards can move in all three directions:

    2 forwards, n=76:  more defensive 30, lateral 31, more attacking 15

On this subset the label is not decided by arithmetic, so it is the one place
the question "does our action + their style tell us which way they go?" can be
asked honestly.

Pre-registered bar (written before this script was first run; do not edit)
--------------------------------------------------------------------------
Label definition: unchanged from diagnose_direction.py (forwards-based). We do
not search over alternative definitions.

Primary model ("N1 model"): logistic regression on
    opponent's current shape + what we just did + opponent style.
This is the novelty's own claim in its smallest form. The full feature set
(+ game state) is reported as secondary only.

Evaluation: GroupKFold(5) by match, out-of-fold predictions, 3 seeds of fold
assignment (match order shuffled per seed). Metric: out-of-fold accuracy.

PASS requires BOTH
  (a) N1 model accuracy > majority-class accuracy, and
  (b) paired cluster bootstrap (resampling matches, 2000 draws) of
      acc(N1 model) - acc(current-shape-only model) has a 95% CI above 0.

Anything else is FAIL, and we report it as FAIL. n=76 gives little power, so a
FAIL here means "not shown", not "shown absent".

    python diagnose_direction_subset.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ramfp.vocab import decode_formation, structural_distance

LABELS = {0: "more defensive", 1: "lateral", 2: "more attacking"}
SEEDS = 3
N_BOOT = 2000


def oof_predictions(X, y, groups, seed):
    """Out-of-fold predictions; the match->fold assignment is shuffled per seed."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = dict(zip(uniq, rng.permutation(len(uniq))))
    g = np.array([perm[x] for x in groups])
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/processed/triples.parquet")
    ap.add_argument("--out", default="results/direction_subset.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.triples)
    chg = df[df.opp_response.astype(int) != df.opp_formation.astype(int)].reset_index(drop=True)
    cur = [decode_formation(int(c)) for c in chg.opp_formation]
    nxt = [decode_formation(int(c)) for c in chg.opp_response]
    keep = np.array([c[-1] == 2 for c in cur])
    sub = chg[keep].reset_index(drop=True)
    cur = [c for c, k in zip(cur, keep) if k]
    nxt = [n for n, k in zip(nxt, keep) if k]
    d = np.array([n[-1] - c[-1] for c, n in zip(cur, nxt)])
    y = np.where(d > 0, 2, np.where(d < 0, 0, 1))
    groups = sub.match_id.to_numpy()

    counts = Counter(y)
    maj_class = max(counts, key=counts.get)
    majority = counts[maj_class] / len(y)
    print(f"opponents currently on 2 forwards who changed shape: n={len(y)}, "
          f"matches={len(np.unique(groups))}")
    for k in sorted(counts):
        print(f"   {LABELS[k]:<16}{counts[k]:>4}  ({counts[k] / len(y):.1%})")

    shape = pd.DataFrame({"opp_cur_def": [c[0] for c in cur],
                          "opp_cur_lines": [len(c) for c in cur]})
    shape = pd.concat([shape, pd.get_dummies(sub.opp_formation.astype(int).astype(str),
                                             prefix="opp_cur")], axis=1)
    ours = pd.DataFrame({
        "our_d_fwd": [decode_formation(int(b))[-1] - decode_formation(int(a))[-1]
                      for a, b in zip(sub.f_src, sub.f_tgt)],
        "our_d_def": [decode_formation(int(b))[0] - decode_formation(int(a))[0]
                      for a, b in zip(sub.f_src, sub.f_tgt)],
        "struct_dist": [structural_distance(int(a), int(b)) for a, b in zip(sub.f_src, sub.f_tgt)]})
    style = pd.get_dummies(sub.opp_style.astype(str), prefix="opp_style")
    state = pd.DataFrame({"minute": sub.minute.astype(float), "goal_diff": sub.goal_diff.astype(float),
                          "is_home": sub.is_home.astype(float), "period": sub.period.astype(float)})

    X = lambda *b: pd.concat(b, axis=1).astype(float).to_numpy()
    models = {
        "current shape only": X(shape),
        "N1 model: shape + our action + style  [PRIMARY]": X(shape, ours, style),
        "  ablation: shape + our action (no style)": X(shape, ours),
        "full: + game state  [secondary]": X(shape, ours, style, state),
    }

    correct = {}
    for name, Xm in models.items():
        correct[name] = np.stack([(oof_predictions(Xm, y, groups, s) == y) for s in range(SEEDS)]).mean(0)

    base, prim = "current shape only", "N1 model: shape + our action + style  [PRIMARY]"
    uniq = np.unique(groups)
    idx_by_match = {m: np.where(groups == m)[0] for m in uniq}
    rng = np.random.default_rng(0)
    boots = {name: [] for name in models}
    for _ in range(N_BOOT):
        take = np.concatenate([idx_by_match[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        for name in models:
            boots[name].append(correct[name][take].mean() - correct[base][take].mean())

    print(f"\n{'model':<50}{'OOF acc':>9}{'vs shape-only [95% CI]':>30}")
    print("-" * 89)
    print(f"{'majority class':<50}{majority:>9.4f}")
    out = {"n": int(len(y)), "matches": int(len(uniq)), "majority": majority,
           "class_counts": {LABELS[k]: int(v) for k, v in counts.items()}, "models": {}}
    for name in models:
        acc = float(correct[name].mean())
        lo, hi = np.percentile(boots[name], [2.5, 97.5])
        diff = acc - float(correct[base].mean())
        ci = "" if name == base else f"{diff:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(f"{name:<50}{acc:>9.4f}{ci:>30}")
        out["models"][name.strip()] = {"acc": acc, "diff_vs_shape": diff, "ci": [float(lo), float(hi)]}

    p = out["models"][prim]
    passed = p["acc"] > majority and p["ci"][0] > 0
    out["verdict"] = "PASS" if passed else "FAIL"
    print(f"\nPre-registered bar: N1 model > majority AND CI(N1 - shape-only) above 0")
    print(f"   N1 > majority: {p['acc']:.4f} vs {majority:.4f} -> {p['acc'] > majority}")
    print(f"   CI lower bound {p['ci'][0]:+.4f} > 0 -> {p['ci'][0] > 0}")
    print(f"VERDICT: {out['verdict']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
