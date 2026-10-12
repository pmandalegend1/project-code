#!/usr/bin/env python3
"""
improve_ranker.py -- two follow-ups to the cross-fitted ranker, both inside the
novelty (Algorithm 1's critics and weights), both on the out-of-fold features
learn_ranker.py --crossfit already wrote. Nothing is retrained.

Bars fixed before the first run on the GPU features (do not edit)
------------------------------------------------------------------
A  CONTEXT-AWARE WEIGHTS. Algorithm 1 has one weight per critic. Tactically,
   what matters should depend on the game: chasing a goal, xG; protecting a
   lead, stability and counter-resistance. We let each weight vary with
   trailing / leading / late (minute >= 70), learned out-of-fold.
     A1 (primary)  context-aware critics-only beats constant-weight critics-only
                   on pooled top-1, match-clustered paired bootstrap CI above 0.
     A2 (secondary) priors + context-aware critics beats priors only, same test.

B  DO COACHES WHO DO WHAT THE PLANNER WOULD DO, DO BETTER AFTERWARDS?
   Top-1 against the coach's choice measures imitation. This asks about
   quality. For each decision: the team's xG difference in the first stable
   10-minute window after the change, minus the same in the last stable window
   before it (team's perspective). Regress that change on
       agree_planner (coach's choice == planner's top pick, planner = the
       out-of-fold learned critics-only ranker that passed B2)
       + goal_diff + minute + is_home + pre_xg.
   pre_xg (the team's xG difference in the window before the change) was added
   after the container smoke test and BEFORE the GPU run. It makes the test
   stricter: a coach usually changes shape after a bad spell, and the next
   window tends to be better anyway (regression to the mean). Without it, any
   rule that picks "changes made after a bad spell" would look like good advice.
     B  (primary)   coefficient on agree_planner > 0, match-clustered bootstrap
                    95% CI above 0.
     Reported, no bar: the same with agree_transition (the transition matrix's
     top pick) as a comparison, and both together.
   Observational: a PASS is an association, not proof the advice causes it.

    python improve_ranker.py --features-dir results --windows ../user/data/processed/stable_windows.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ramfp.config import Config
from learn_ranker import add_priors, fit_clogit, paired_boot, ranks, summarise

CRIT = ["xG", "stability", "counter_resistance", "prior"]
PRI = ["log_trans", "log_freq", "prior"]


def add_context(cand, df):
    st = df.set_index("_dec")[["goal_diff", "minute"]]
    c = cand.join(st, on="decision")
    ctx = {"trail": (c.goal_diff < 0).astype(float), "lead": (c.goal_diff > 0).astype(float),
           "late": (c.minute >= 70).astype(float)}
    cols = []
    for k in ["xG", "stability", "counter_resistance"]:
        for n, v in ctx.items():
            c[f"{k}*{n}"] = c[k] * v
            cols.append(f"{k}*{n}")
    return c, cols


def groups(c, cols):
    G, order = [], []
    for d, g in c.groupby("decision", sort=False):
        X = g[cols].to_numpy(float)
        hit = np.where(g["f_tgt"].to_numpy() == g["truth"].iloc[0])[0]
        G.append((X, int(hit[0]) if len(hit) else -1)); order.append((d, g["match_id"].iloc[0], g["f_tgt"].to_numpy()))
    return G, order


def oof(cand, df, folds, rankers):
    pooled, meta = defaultdict(list), []
    tops = defaultdict(dict)
    for k, (tr_i, te_i) in enumerate(folds):
        c = add_priors(cand, df.iloc[tr_i])
        trd, ted = set(df.iloc[tr_i]["_dec"]), set(df.iloc[te_i]["_dec"])
        tr, te = c[c.decision.isin(trd)].copy(), c[c.decision.isin(ted)].copy()
        allc = sorted({x for cols in rankers.values() for x in cols})
        mu, sd = tr[allc].mean(), tr[allc].std().replace(0, 1.0)
        tr[allc] = (tr[allc] - mu) / sd; te[allc] = (te[allc] - mu) / sd
        for name, cols in rankers.items():
            if name == "transition matrix":
                G, order = groups(te, ["log_trans"]); fn = lambda X: X[:, 0]
            else:
                w = fit_clogit(groups(tr, cols)[0]); G, order = groups(te, cols); fn = lambda X, w=w: X @ w
            pooled[name].extend(ranks(G, fn))
            for (X, _), (d, m, tg) in zip(G, order):
                tops[name][d] = int(tg[int(np.argmax(fn(X)))])
        meta.extend(m for _, m, _ in groups(te, ["prior"])[1])
    return pooled, meta, tops


def outcome_table(df, windows):
    w = windows.copy()
    out = []
    by = {m: g.sort_values("window_start_sec") for m, g in w.groupby("match_id")}
    for _, r in df.iterrows():
        g = by.get(int(r.match_id))
        if g is None:
            continue
        t = int(r.minute) * 60
        sign = 1.0 if int(r.is_home) == 1 else -1.0
        pre = g[g.window_start_sec + 600 <= t]
        post = g[g.window_start_sec >= t]
        if len(pre) == 0 or len(post) == 0:
            continue
        out.append({"_dec": int(r._dec), "match_id": int(r.match_id),
                    "d_xg": sign * (post.xg_delta.iloc[0] - pre.xg_delta.iloc[-1]),
                    "pre_xg": sign * pre.xg_delta.iloc[-1],
                    "goal_diff": float(r.goal_diff), "minute": float(r.minute),
                    "is_home": float(r.is_home)})
    return pd.DataFrame(out)


def ols_boot(y, X, groups_, B=2000, seed=0):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups_); idx = {m: np.where(groups_ == m)[0] for m in uniq}
    bs = []
    for _ in range(B):
        take = np.concatenate([idx[m] for m in rng.choice(uniq, len(uniq))])
        bs.append(np.linalg.lstsq(X1[take], y[take], rcond=None)[0])
    bs = np.array(bs)
    return beta, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="results")
    ap.add_argument("--triples", default="data/processed/triples.parquet")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/improve_ranker.json")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    df = pd.read_parquet(a.triples).reset_index(drop=True); df["_dec"] = np.arange(len(df))
    files = sorted(glob.glob(os.path.join(a.features_dir, "ranker_features_crossfit_fold*.parquet")))
    assert len(files) == cfg.train.n_folds, f"expected {cfg.train.n_folds} fold caches, found {files}"
    cand = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cand, ctx_cols = add_context(cand, df)
    folds = list(GroupKFold(n_splits=cfg.train.n_folds).split(df, groups=df["match_id"].to_numpy()))

    rankers = {"transition matrix": ["log_trans"],
               "critics, constant weights": CRIT,
               "critics, context-aware weights": CRIT + ctx_cols,
               "priors only": PRI,
               "priors + context-aware critics": PRI + CRIT[:3] + ctx_cols}
    pooled, meta, tops = oof(cand, df, folds, rankers)
    mat = np.array(meta)
    print(f"=== A: context-aware weights ({len(mat)} decisions, out-of-fold) ===")
    print(f"{'ranker':<34}{'top-1':>8}{'top-3':>8}{'MRR':>8}")
    for n, rk in pooled.items():
        s = summarise(rk); print(f"{n:<34}{s['top1']:>8.3f}{s['top3']:>8.3f}{s['mrr']:>8.3f}")
    a1 = paired_boot(pooled["critics, context-aware weights"], pooled["critics, constant weights"], groups=mat)
    a2 = paired_boot(pooled["priors + context-aware critics"], pooled["priors only"], groups=mat)
    v = lambda t: "PASS" if t[1] > 0 else "FAIL"
    print(f"A1 context vs constant (critics only)  {a1[0]:+.4f} [{a1[1]:+.4f}, {a1[2]:+.4f}] -> {v(a1)}")
    print(f"A2 priors + context critics vs priors  {a2[0]:+.4f} [{a2[1]:+.4f}, {a2[2]:+.4f}] -> {v(a2)}")

    print("\n=== B: do coaches who follow the planner do better afterwards? ===")
    ot = outcome_table(df, pd.read_parquet(a.windows, columns=["match_id", "window_start_sec", "xg_delta"]))
    truth = df.set_index("_dec")["f_tgt"].astype(int)
    ot["agree_planner"] = [float(tops["critics, constant weights"].get(d, -1) == truth[d]) for d in ot._dec]
    ot["agree_transition"] = [float(tops["transition matrix"].get(d, -1) == truth[d]) for d in ot._dec]
    y, g = ot.d_xg.to_numpy(), ot.match_id.to_numpy()
    ctrl = ot[["goal_diff", "minute", "is_home", "pre_xg"]].to_numpy()
    print(f"decisions with a stable window before and after: {len(ot)} of {len(df)}")
    print(f"agree with planner {ot.agree_planner.mean():.1%}, with transition matrix {ot.agree_transition.mean():.1%}")
    res = {}
    for name, cols in [("planner", ["agree_planner"]), ("transition", ["agree_transition"]),
                       ("both", ["agree_planner", "agree_transition"])]:
        b, lo, hi = ols_boot(y, np.column_stack([ot[cols].to_numpy(), ctrl]), g)
        for i, c in enumerate(cols, start=1):
            print(f"  [{name:<10}] {c:<18} coef {b[i]:+.4f} xG  95% CI [{lo[i]:+.4f}, {hi[i]:+.4f}]")
            res[f"{name}:{c}"] = {"coef": float(b[i]), "ci": [float(lo[i]), float(hi[i])]}
    # placebo: agreement with a RANDOM candidate. Tells us how large a
    # coefficient "agreement with anything" produces by chance.
    cands_by_dec = cand.groupby("decision")["f_tgt"].apply(lambda x: x.to_numpy()).to_dict()
    rng = np.random.default_rng(0)
    X1c = lambda ag: np.column_stack([np.ones(len(y)), ag, ctrl])
    plac = []
    for _ in range(200):
        ag = np.array([float(rng.choice(cands_by_dec[d]) == truth[d]) for d in ot._dec])
        plac.append(np.linalg.lstsq(X1c(ag), y, rcond=None)[0][1])
    plac = np.array(plac)
    pct = float((plac >= res["planner:agree_planner"]["coef"]).mean())
    print(f"  placebo (agree with a random candidate, 200 draws): coef mean {plac.mean():+.4f}, "
          f"95th pct {np.percentile(plac, 95):+.4f}; share >= planner's: {pct:.3f}")
    res["placebo"] = {"mean": float(plac.mean()), "p95": float(np.percentile(plac, 95)), "share_ge": pct}
    bp = res["planner:agree_planner"]
    print(f"B (primary) agree_planner > 0 with CI above 0 -> {'PASS' if bp['ci'][0] > 0 else 'FAIL'}")

    out = {"A": {n: summarise(rk) for n, rk in pooled.items()},
           "A1": {"diff": a1[0], "ci": a1[1:], "verdict": v(a1)},
           "A2": {"diff": a2[0], "ci": a2[1:], "verdict": v(a2)},
           "B": res, "B_verdict": "PASS" if bp["ci"][0] > 0 else "FAIL", "B_n": int(len(ot))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
