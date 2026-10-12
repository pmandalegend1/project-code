#!/usr/bin/env python3
"""
eval_splits.py -- the data splits for E-A / E-B / E-C and the paired bootstrap.

Everything here works on the stable_windows DataFrame only (no torch, no graphs),
so it is used identically for the GNN, Ridge and boosting.

E-A  learning_curve_train(train_df, frac, seed)
        keep a random `frac` of training MATCHES (never single windows, so no
        match is half in).
E-B  heldout_pair_split(train_df, test_df, frac=0.2, seed)
        remove 20% of the formation matchups from training; evaluate only on
        test windows that use those matchups. A matchup is UNORDERED
        ({442, 433} covers both 442-v-433 and 433-v-442), otherwise the flipped
        version would stay in training and leak through the swap symmetry.
E-C  heldout_formation_split(train_df, test_df, formations)
        remove every window where either side plays one of `formations`;
        evaluate on test windows where either side does.

paired_bootstrap(...)  resamples whole MATCHES, so windows from one match (which
        are correlated) stay together; compares two models on the same windows.

Run `python eval_splits.py` to execute the self-test on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RARE = (3412, 41212, 3511)


def _pair_key(df: pd.DataFrame) -> pd.Series:
    h = df["home_formation"].astype(int).values
    a = df["away_formation"].astype(int).values
    lo, hi = np.minimum(h, a), np.maximum(h, a)
    return pd.Series(list(zip(lo, hi)), index=df.index)


# ------------------------------- E-A ---------------------------------------- #
def learning_curve_train(train_df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    matches = np.sort(train_df["match_id"].unique())
    rng = np.random.default_rng(seed)
    k = max(1, int(round(frac * len(matches))))
    keep = set(rng.choice(matches, size=k, replace=False).tolist())
    return train_df[train_df["match_id"].isin(keep)].reset_index(drop=True)


# ------------------------------- E-B ---------------------------------------- #
def heldout_pair_split(train_df, test_df, frac: float = 0.2, seed: int = 0):
    """Returns (train_reduced, eval_df, held_pairs)."""
    all_pairs = sorted(set(_pair_key(train_df)) | set(_pair_key(test_df)))
    rng = np.random.default_rng(seed)
    k = max(1, int(round(frac * len(all_pairs))))
    idx = rng.choice(len(all_pairs), size=k, replace=False)
    held = {all_pairs[i] for i in idx}
    tr = train_df[~_pair_key(train_df).isin(held)].reset_index(drop=True)
    ev = test_df[_pair_key(test_df).isin(held)].reset_index(drop=True)
    return tr, ev, held


# ------------------------------- E-C ---------------------------------------- #
def heldout_formation_split(train_df, test_df, formations=DEFAULT_RARE):
    """Returns (train_reduced, eval_df)."""
    fs = {int(f) for f in formations}

    def uses(df):
        return df["home_formation"].astype(int).isin(fs) | df["away_formation"].astype(int).isin(fs)

    return train_df[~uses(train_df)].reset_index(drop=True), test_df[uses(test_df)].reset_index(drop=True)


# ----------------------------- bootstrap ------------------------------------ #
def paired_bootstrap(y, pred_a, pred_b, match_ids, n_boot: int = 2000, seed: int = 0):
    """Compare model A against model B on the same windows, clustered by match.

    y, pred_a, pred_b: [N] (one target) or lists of [N] arrays (one per training
    seed; squared errors are averaged across seeds per window, not ensembled).
    Returns dict: mse_a, mse_b, diff = mse_a - mse_b (negative => A better),
    ci95 of diff, p_a_better = share of resamples where A beats B.
    """
    y = np.asarray(y, float)

    def sqerr(p):
        ps = p if isinstance(p, (list, tuple)) else [p]
        return np.mean([(np.asarray(q, float) - y) ** 2 for q in ps], axis=0)

    ea, eb = sqerr(pred_a), sqerr(pred_b)
    ok = np.isfinite(ea) & np.isfinite(eb)
    ea, eb = ea[ok], eb[ok]
    codes, uniq = pd.factorize(np.asarray(match_ids)[ok])
    n_m = len(uniq)
    sa = np.bincount(codes, weights=ea, minlength=n_m)
    sb = np.bincount(codes, weights=eb, minlength=n_m)
    cnt = np.bincount(codes, minlength=n_m).astype(float)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_m, n_m)
        n = cnt[pick].sum()
        diffs[b] = sa[pick].sum() / n - sb[pick].sum() / n
    return {
        "n_windows": int(ok.sum()), "n_matches": int(n_m),
        "mse_a": float(ea.mean()), "mse_b": float(eb.mean()),
        "diff": float(ea.mean() - eb.mean()),
        "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "p_a_better": float((diffs < 0).mean()),
    }


# ------------------------------ self-test ----------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    forms = [442, 433, 4231, 352, 3412, 41212, 3511]
    rows = []
    for m in range(300):
        for _ in range(8):
            rows.append((m, int(rng.choice(forms)), int(rng.choice(forms))))
    df = pd.DataFrame(rows, columns=["match_id", "home_formation", "away_formation"])
    train, test = df[df.match_id < 240].reset_index(drop=True), df[df.match_id >= 240].reset_index(drop=True)

    # E-A: whole matches, right size, nested behaviour is not required
    sub = learning_curve_train(train, 0.05, seed=1)
    assert sub["match_id"].nunique() == 12 and set(sub.match_id) <= set(train.match_id)
    print(f"[ok] E-A: 5% -> {sub.match_id.nunique()} of {train.match_id.nunique()} matches")

    # E-B: held pairs never in train, always in eval, unordered
    tr, ev, held = heldout_pair_split(train, test, 0.2, seed=2)
    assert not set(_pair_key(tr)) & held and set(_pair_key(ev)) <= held and len(ev) > 0
    print(f"[ok] E-B: {len(held)} held-out matchups, train {len(tr)}, eval {len(ev)}")

    # E-C: rare formations absent from train, present in eval
    tr, ev = heldout_formation_split(train, test)
    rare = set(DEFAULT_RARE)
    assert not (set(tr.home_formation) | set(tr.away_formation)) & rare
    assert ((ev.home_formation.isin(rare)) | (ev.away_formation.isin(rare))).all()
    print(f"[ok] E-C: train {len(tr)}, eval {len(ev)}")

    # bootstrap: B is clearly worse than A -> diff<0, CI excludes 0; identical -> CI covers 0
    y = rng.normal(size=2000); mid = np.repeat(np.arange(200), 10)
    good, bad = y + rng.normal(scale=0.3, size=2000), y + rng.normal(scale=0.8, size=2000)
    r = paired_bootstrap(y, good, bad, mid)
    assert r["diff"] < 0 and r["ci95"][1] < 0 and r["p_a_better"] > 0.99
    r0 = paired_bootstrap(y, good, good, mid)
    assert r0["diff"] == 0
    r5 = paired_bootstrap(y, [good, good + 0.01], [bad, bad], mid)   # multi-seed input
    assert r5["diff"] < 0
    print(f"[ok] bootstrap: diff={r['diff']:+.3f} CI=({r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f})")
    print("all checks passed")


if __name__ == "__main__":
    _selftest()
