#!/usr/bin/env python3
"""
diagnose_direction.py -- is the DIRECTION of the opponent's adjustment
predictable, even when the exact formation is not?

The idea, and why it was worth testing
--------------------------------------
Asking "which of 17 shapes" on 235 observed changes is asking a lot of a small
sample. A natural weaker question is: given that they changed, did they go more
defensive, more attacking, or sideways? Three classes instead of seventeen. If
that were predictable, N1's claim would survive in a weaker but honest form.

The first numbers looked like a result
--------------------------------------
    majority class          0.3745
    logistic regression     0.6128 +- 0.0511
    gradient boosting       0.5915 +- 0.0474
    shuffled-label control  0.2426 +- 0.0371

A 0.24 gain over the majority class with a clean shuffled control. We nearly
reported it.

Why it is not a result
----------------------
The label is the sign of (forwards after - forwards before), so the opponent's
CURRENT shape constrains which labels are even reachable: a side already playing
one forward cannot become more defensive on this axis. That is a structural fact
about the label, not tactical behaviour.

Feeding the model only the opponent's current shape isolates it:

    opponent's current shape only    0.6809 +- 0.0932
    + what we just did               0.6511 +- 0.0732
    + game state and style (full)    0.6128 +- 0.0511

Three numbers that do not decrease from left to right, and the first one is the
best. Everything the richer models know is already in the current shape, and
adding features makes it worse -- 235 samples is not enough to support them.

So the direction is not predictable either. The apparent signal is the label's
own arithmetic. We report this rather than the 0.61.

    python diagnose_direction.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ramfp.vocab import decode_formation, structural_distance

LABELS = {0: "more defensive", 1: "lateral", 2: "more attacking"}


def cv_accuracy(X, y, groups, make_model, seeds=3, n_splits=5):
    accs = []
    for s in range(seeds):
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = make_model(s)
            m.fit(X[tr], y[tr])
            accs.append(float((m.predict(X[te]) == y[te]).mean()))
    a = np.array(accs)
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/processed/triples.parquet")
    ap.add_argument("--out", default="results/direction.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.triples)
    chg = df[df.opp_response.astype(int) != df.opp_formation.astype(int)].reset_index(drop=True)
    cur = [decode_formation(int(c)) for c in chg.opp_formation]
    nxt = [decode_formation(int(c)) for c in chg.opp_response]
    d_fwd = np.array([n[-1] - c[-1] for c, n in zip(cur, nxt)])
    y = np.where(d_fwd > 0, 2, np.where(d_fwd < 0, 0, 1))
    groups = chg.match_id.to_numpy()

    counts = Counter(y)
    majority = max(counts.values()) / len(y)
    print(f"decisions where the opponent actually changed shape: {len(chg)}")
    for k in sorted(counts):
        print(f"   {LABELS[k]:<16}{counts[k]:>5}  ({counts[k] / len(y):.1%})")
    print(f"majority-class baseline: {majority:.4f}\n")

    shape = {"opp_cur_fwd": [c[-1] for c in cur],
             "opp_cur_def": [c[0] for c in cur],
             "opp_cur_lines": [len(c) for c in cur]}
    ours = {"our_d_fwd": [decode_formation(int(b))[-1] - decode_formation(int(a))[-1]
                          for a, b in zip(chg.f_src, chg.f_tgt)],
            "our_d_def": [decode_formation(int(b))[0] - decode_formation(int(a))[0]
                          for a, b in zip(chg.f_src, chg.f_tgt)],
            "struct_dist": [structural_distance(int(a), int(b))
                            for a, b in zip(chg.f_src, chg.f_tgt)]}
    state = {"minute": chg.minute.astype(float), "goal_diff": chg.goal_diff.astype(float),
             "abs_goal_diff": chg.goal_diff.abs().astype(float),
             "is_home": chg.is_home.astype(float), "period": chg.period.astype(float),
             "opp_ppda_pre": pd.to_numeric(chg.opp_ppda_pre, errors="coerce")}

    def build(*blocks, dummies=False):
        X = pd.DataFrame({k: v for blk in blocks for k, v in blk.items()})
        if "opp_ppda_pre" in X:
            X["opp_ppda_pre"] = X["opp_ppda_pre"].fillna(X["opp_ppda_pre"].median())
        if dummies:
            for col in ("intent", "phase", "opp_style"):
                X = pd.concat([X, pd.get_dummies(chg[col].astype(str), prefix=col)], axis=1)
        return X.astype(float).to_numpy()

    logistic = lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    boosting = lambda s: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3, class_weight="balanced", random_state=s)

    full = build(shape, ours, state, dummies=True)
    out = {"n": len(chg), "majority": majority, "class_counts": {LABELS[k]: int(v) for k, v in counts.items()}}

    print(f"{'model (full feature set)':<36}{'accuracy':>20}")
    print("-" * 56)
    print(f"{'majority class':<36}{majority:>12.4f}")
    for name, mk in (("logistic regression", logistic), ("gradient boosting", boosting)):
        m, sd = cv_accuracy(full, y, groups, mk)
        out[name] = {"mean": m, "std": sd}
        print(f"{name:<36}{m:>12.4f} +- {sd:.4f}")
    rng = np.random.default_rng(0)
    y_shuf = y.copy(); rng.shuffle(y_shuf)
    m, sd = cv_accuracy(full, y_shuf, groups, boosting)
    out["shuffled_control"] = {"mean": m, "std": sd}
    print(f"{'shuffled-label control':<36}{m:>12.4f} +- {sd:.4f}")

    print(f"\n{'where the accuracy comes from':<36}{'accuracy':>20}")
    print("-" * 56)
    print(f"{'majority class':<36}{majority:>12.4f}")
    ladder = [("opponent's current shape only", build(shape)),
              ("+ what we just did", build(shape, ours)),
              ("+ game state and style (full)", full)]
    out["ablation"] = {}
    for name, X in ladder:
        m, sd = cv_accuracy(X, y, groups, logistic)
        out["ablation"][name] = {"mean": m, "std": sd}
        print(f"{name:<36}{m:>12.4f} +- {sd:.4f}")

    best_block = max(out["ablation"].items(), key=lambda kv: kv[1]["mean"])
    print(f"\nBest feature set: '{best_block[0]}' at {best_block[1]['mean']:.4f}.")
    if best_block[0].startswith("opponent's current shape"):
        print("The opponent's current shape alone is the best predictor, and adding")
        print("everything else makes it worse. The label's own arithmetic explains the")
        print("accuracy: the current shape constrains which directions are reachable.")
        print("This is NOT evidence that the direction of the response is predictable.")
    out["verdict"] = ("structural artefact" if best_block[0].startswith("opponent's current shape")
                      else "possible signal, investigate further")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
