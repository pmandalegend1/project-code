#!/usr/bin/env python3
"""
learn_ranker.py -- learn how to combine the planner's signals instead of
averaging them, and test whether the critics add anything to a data prior.

Why
---
Algorithm 1 scores a candidate as w1*C_xG + w2*C_stab + w3*C_res. The weights
were set by hand to equal thirds. Measured end to end, that planner ranked the
coach's actual change below chance, while a transition matrix P(f_tgt | f_src)
-- counted on training decisions, reading nothing but the current formation --
ranked it first 30% of the time. Two things follow:

  1. The weights should be LEARNED, which is what the w's in Algorithm 1 invite.
  2. The transition matrix is not a rival to the planner; it is exactly the
     prior the generator's hand-written `_prior` was standing in for.

So this fits a conditional logit over each decision's candidates -- softmax over
the candidate set, one weight per signal, L2-regularised -- on training
decisions only, and asks the question that matters:

  THE BAR (fixed before running): does adding the critics to the transition
  prior improve held-out top-1, with a paired bootstrap 95% interval that
  excludes zero? If not, the critics add nothing that the data prior does not
  already know, and the paper should say so.

Leakage, checked and handled
----------------------------
The world model was trained on windows from 92% of the matches in the usual
held-out decision fold, so C_xG and C_res have in effect seen those matches'
outcomes. We therefore report two evaluations:
  fold-0     the same 427 decisions as eval_planner.py (comparable, contaminated)
  clean      the 146 decisions from matches the world model never saw, with the
             ranker and transition matrix fitted on all other decisions

    python learn_ranker.py --wm-dir ../preprocessed/world_model --features-dir results --clean

Cross-fitted evaluation (--crossfit), the proper fix for the leak
-----------------------------------------------------------------
The clean subset above is only 146 decisions, too few to detect a gain of a
few points. Cross-fitting uses all 2,133: the decisions are split into the same
five match-grouped folds the opponent model uses, and fold k is scored by a
world model, C_res table and opponent model that were all trained WITHOUT fold
k's matches (run_crossfit.sh, train_opponent.py --save-folds). The ranker is
then fitted on four folds and tested on the fifth, five times, and the ranks
pooled. Every score and every weight is out-of-fold.

  Bars for --crossfit, fixed before the first run (do not edit):
  B1  learned priors + critics beats learned priors only on pooled top-1,
      match-clustered paired bootstrap 95% CI above 0.     (critics add to the prior)
  B2  learned critics only -- Algorithm 1 with learned weights and no data
      prior -- beats random on pooled top-1, same test.    (the novel system ranks at all)
  Reported, no bar: learned vs equal weights; full ranker vs transition matrix.

    python learn_ranker.py --crossfit ~/user/data/processed/stable_windows.parquet --features-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from ramfp.data.ppda_impute import impute_out_of_fold
from ramfp.config import Config
from ramfp.inference import (RAMFPPlanner, StructuralStabilityCritic, TransitionGenerator,
                             XGCritic)
from ramfp.models.opponent import OpponentCoBehaviorModel
from eval_planner import state_from_row

# State-conditional-only variant: remove the marginal transition matrix so
# the state features have to carry the signal alone. If they can, top-1 should
# rise above 0.28 without the marginal; if they can't, the state features were
# just repackaging the marginal and we've confirmed the ceiling.
SIGNALS = ["log_trans", "log_trans_state", "log_trans_opp",
           "log_trans_team", "log_team_freq",
           "log_freq", "prior",
           "xG", "stability", "counter_resistance"]

# The state encoding (z_t) is added to the feature set dynamically, because
# its dimensionality comes from the opponent encoder's d_model and is only
# known once the extract step has run. `prepare_feature_cols` returns the
# full list to fit and rank on.
N_STATE_DIMS = 256


def prepare_feature_cols(cand):
    """All ranker features, in order: the six scalars, then z_t if present."""
    z_cols = [c for c in cand.columns if c.startswith("z_")]
    z_cols.sort(key=lambda c: int(c.split("_")[1]))
    return SIGNALS + z_cols


# ----------------------------------------------------------------- features
def extract(planner, gen, df, cfg, cache):
    """One row per (decision, candidate target), critics z-scored within decision."""
    if cache and os.path.exists(cache):
        print(f"loading cached candidate features from {cache}")
        return pd.read_parquet(cache)
    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(df.iterrows(), 1):
        st = state_from_row(r)
        prior = {}
        for c in gen.generate(st, k=cfg.inference.candidate_k):
            prior.setdefault(c.f_tgt, []).append(c.prior)
        recs = planner.plan(st, top_n=0)            # every candidate, no revision
        per = defaultdict(lambda: defaultdict(list))
        for x in recs:
            for k, v in x.critic_scores.items():
                per[x.f_tgt][k].append(v)
        # Capture the state encoding once per decision, then broadcast to every
        # candidate. z_t is a property of the decision, not the candidate.
        z_row = None
        if recs and getattr(recs[0], "z_t", None) is not None:
            z_row = np.asarray(recs[0].z_t, dtype=np.float32)

        for f_tgt, d in per.items():
            row = {"decision": int(r["triple_id"]) if "triple_id" in r else i,
                   "match_id": int(r["match_id"]), "f_src": int(r["f_src"]),
                   "f_tgt": int(f_tgt), "truth": int(r["f_tgt"]),
                   "prior": float(np.mean(prior.get(f_tgt, [0.0]))),
                   "opp_backline_bucket": _backline_bucket(int(r["opp_formation"])),
                   "opp_style": int(r["opp_style"]),
                   "team_id": int(r["team_id"]),
                   "goal_diff": int(st.goal_diff), "minute": int(st.minute),
                   **{k: float(np.mean(v)) for k, v in d.items()}}
            if z_row is not None:
                for zk, zv in enumerate(z_row):
                    row[f"z_{zk}"] = float(zv)
            rows.append(row)
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(df)} decisions  ({el/i:.2f}s each, ~{el/i*(len(df)-i)/60:.1f} min left)",
                  flush=True)
    out = pd.DataFrame(rows)
    # critics on a common scale within each decision
    # z-scoring is only for the critic columns: it puts them on a common scale
    # WITHIN each decision. The z_* columns are the state encoding and must
    # keep their absolute values -- they already live on a scale the opponent
    # encoder learned, and per-decision z-scoring would erase the signal that
    # distinguishes one decision from another.
    for c in ("xG", "stability", "counter_resistance"):
        g = out.groupby("decision")[c]
        out[c] = (out[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        out[c] = out[c].fillna(0.0)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        out.to_parquet(cache, index=False)
    return out


def _style_bucket(style: int) -> str:
    return f"style_{int(style)}"


def _backline_bucket(f_opp: int) -> str:
    """Coarse bucket for the opponent's outfield shape.

    17 distinct formations x 17 source formations = 289 cells, most with fewer
    than 5 training decisions. Collapsing the opponent's shape to three buckets
    gives 17 x 3 = 51 cells at ~35 decisions each, enough to estimate with
    smoothing. The split is by defender count -- the axis coaches most
    obviously respond to.
    """
    s = str(int(f_opp))
    back = int(s[0])
    if back == 3: return "back3"
    if back == 5: return "back5"
    return "back4"

def team_used_formations(train_decisions):
    """Per-team set of formations observed in training decisions.

    Used to filter test candidates: a team's own history narrows the
    candidate set. A team that used only 8 formations in training gets
    at most 8 candidates for its test decisions.
    """
    used = defaultdict(set)
    if "team_id" not in train_decisions.columns:
        return {}
    for tid, ftgt in zip(train_decisions["team_id"], train_decisions["f_tgt"]):
        used[int(tid)].add(int(ftgt))
    return dict(used)


def filter_candidates_by_team(cand, team_used):
    """Keep only candidates whose f_tgt the team has used in training.

    Decisions where the team has no training history keep the full set
    (there is no filter to apply). If the truth itself is filtered out,
    groups_of assigns y=-1 and summarise counts the decision as a miss,
    which is the honest treatment: the filter removed the answer.
    """
    if "team_id" not in cand.columns or not team_used:
        return cand
    parts = []
    for _, g in cand.groupby("decision", sort=False):
        tid = int(g["team_id"].iloc[0])
        allowed = team_used.get(tid)
        if allowed:
            parts.append(g[g["f_tgt"].isin(allowed)])
        else:
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_priors(cand, train_decisions):
    """Transition and frequency priors, counted on training decisions only.

    Four priors attached:
      log_trans        P(f_tgt | f_src), the marginal transition matrix
      log_trans_state  P(f_tgt | f_src, opp_style), smoothed toward the marginal
      log_trans_opp    P(f_tgt | f_src, opp_backline_bucket), smoothed
      log_freq         P(f_tgt), the global frequency
    """
    tr = train_decisions

    trans = defaultdict(Counter)
    trans_state = defaultdict(Counter)
    trans_opp = defaultdict(Counter)
    trans_team = defaultdict(Counter)      # P(f_tgt | f_src, team_id)
    team_freq = defaultdict(Counter)       # P(f_tgt | team_id)
    has_team = "team_id" in tr.columns
    if has_team:
        for a, b, style, opp_bucket, tid in zip(
                tr["f_src"], tr["f_tgt"], tr["opp_style"],
                tr["opp_backline_bucket"], tr["team_id"]):
            a, b, tid = int(a), int(b), int(tid)
            trans[a][b] += 1
            trans_state[(a, _style_bucket(int(style)))][b] += 1
            trans_opp[(a, str(opp_bucket))][b] += 1
            trans_team[(a, tid)][b] += 1
            team_freq[tid][b] += 1
    else:
        for a, b, style, opp_bucket in zip(tr["f_src"], tr["f_tgt"],
                                            tr["opp_style"], tr["opp_backline_bucket"]):
            a, b = int(a), int(b)
            trans[a][b] += 1
            trans_state[(a, _style_bucket(int(style)))][b] += 1
            trans_opp[(a, str(opp_bucket))][b] += 1

    freq = Counter(tr["f_tgt"].astype(int))
    n_tgt = len(set(cand["f_tgt"]) | set(freq))

    alpha = 5.0

    def _log_p_state(s, t, style):
        cell = trans_state.get((int(s), _style_bucket(int(style))), Counter())
        n_cell = sum(cell.values())
        p_marg = (trans[int(s)][t] + 0.5) / (sum(trans[int(s)].values()) + 0.5 * n_tgt)
        p = (cell[int(t)] + alpha * p_marg) / (n_cell + alpha)
        return float(np.log(max(p, 1e-9)))

    def _log_p_opp(s, t, opp_bucket):
        cell = trans_opp.get((int(s), str(opp_bucket)), Counter())
        n_cell = sum(cell.values())
        p_marg = (trans[int(s)][t] + 0.5) / (sum(trans[int(s)].values()) + 0.5 * n_tgt)
        p = (cell[int(t)] + alpha * p_marg) / (n_cell + alpha)
        return float(np.log(max(p, 1e-9)))

    def _log_p_team(s, t, tid):
        if not has_team:
            return 0.0
        cell = trans_team.get((int(s), int(tid)), Counter())
        n_cell = sum(cell.values())
        p_marg = (trans[int(s)][t] + 0.5) / (sum(trans[int(s)].values()) + 0.5 * n_tgt)
        p = (cell[int(t)] + 10.0 * p_marg) / (n_cell + 10.0)
        return float(np.log(max(p, 1e-9)))

    def _log_team_freq(t, tid):
        if not has_team:
            return 0.0
        cell = team_freq.get(int(tid), Counter())
        n_cell = sum(cell.values())
        p_glob = (freq[int(t)] + 0.5) / (sum(freq.values()) + 0.5 * n_tgt)
        p = (cell[int(t)] + 5.0 * p_glob) / (n_cell + 5.0)
        return float(np.log(max(p, 1e-9)))

    out = cand.copy()
    out["log_trans"] = [np.log((trans[int(s)][int(t)] + 0.5) /
                               (sum(trans[int(s)].values()) + 0.5 * n_tgt))
                        for s, t in zip(out["f_src"], out["f_tgt"])]
    out["log_freq"] = [np.log((freq[int(t)] + 0.5) / (sum(freq.values()) + 0.5 * n_tgt))
                       for t in out["f_tgt"]]
    out["log_trans_state"] = [_log_p_state(s, t, sty)
                              for s, t, sty in zip(out["f_src"], out["f_tgt"],
                                                   out["opp_style"])]
    out["log_trans_opp"] = [_log_p_opp(s, t, ob)
                            for s, t, ob in zip(out["f_src"], out["f_tgt"],
                                                out["opp_backline_bucket"])]
    if has_team and "team_id" in out.columns:
        out["log_trans_team"] = [_log_p_team(s, t, tid)
                                 for s, t, tid in zip(out["f_src"], out["f_tgt"],
                                                      out["team_id"])]
        out["log_team_freq"] = [_log_team_freq(t, tid)
                                for t, tid in zip(out["f_tgt"], out["team_id"])]
    else:
        out["log_trans_team"] = 0.0
        out["log_team_freq"] = 0.0
    return out

# ----------------------------------------------------------------- model
def groups_of(cand, cols):
    G = []
    for _, g in cand.groupby("decision", sort=False):
        X = g[cols].to_numpy(float)
        hit = np.where(g["f_tgt"].to_numpy() == g["truth"].iloc[0])[0]
        G.append((X, int(hit[0]) if len(hit) else -1))
    return G


def fit_clogit(G, l2=1.0):
    """Conditional logit: softmax over each decision's candidates."""
    G = [(X, y) for X, y in G if y >= 0]
    d = G[0][0].shape[1]

    def f(w):
        nll, grad = l2 * w @ w, 2 * l2 * w
        for X, y in G:
            s = X @ w
            m = s.max()
            p = np.exp(s - m); Z = p.sum(); p /= Z
            nll += -(s[y] - m - np.log(Z))
            grad += -(X[y] - p @ X)
        return nll, grad

    return minimize(f, np.zeros(d), jac=True, method="L-BFGS-B").x


def ranks(G, score_fn):
    out = []
    for X, y in G:
        if y < 0:
            out.append(None); continue
        s = score_fn(X)
        out.append(int((s > s[y]).sum()) + 1)      # ties resolved in the truth's favour? no: strict
    return out


def summarise(rk):
    n = len(rk)
    return {"top1": sum(r == 1 for r in rk if r) / n,
            "top3": sum(r <= 3 for r in rk if r) / n,
            "mrr": sum(1 / r for r in rk if r) / n}


def paired_boot(rk_a, rk_b, B=5000, seed=0, groups=None):
    """Paired bootstrap of the top-1 difference. With `groups` (match ids) the
    resampling unit is the match, since decisions in one match are not
    independent; without it, the decision."""
    a = np.array([1.0 if r == 1 else 0.0 for r in rk_a])
    b = np.array([1.0 if r == 1 else 0.0 for r in rk_b])
    rng = np.random.default_rng(seed)
    d = a - b
    if groups is None:
        idx = rng.integers(0, len(a), size=(B, len(a)))
        diff = d[idx].mean(1)
    else:
        g = np.asarray(groups)
        uniq, inv = np.unique(g, return_inverse=True)
        s_g = np.bincount(inv, weights=d, minlength=len(uniq))
        n_g = np.bincount(inv, minlength=len(uniq)).astype(float)
        pick = rng.integers(0, len(uniq), size=(B, len(uniq)))
        diff = s_g[pick].sum(1) / n_g[pick].sum(1)
    return float(d.mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def fit_rank(cand, train_dec, test_dec, seed=1):
    """Fit every ranker on train decisions; return per-decision ranks on test.

    Ranks come back in the order of the test decisions' candidate groups, with
    the decision id and match id alongside so folds can be pooled."""
    c = add_priors(cand, train_dec)
    tr = c[c["decision"].isin(set(train_dec["_dec"]))]
    te = c[c["decision"].isin(set(test_dec["_dec"]))]

    # Feature columns: 6 scalars + the state encoding z_t when it is present.
    feats = prepare_feature_cols(c)
    mu = tr[feats].mean(); sd = tr[feats].std().replace(0, 1.0)
    trs, tes = tr.copy(), te.copy()
    trs[feats] = (trs[feats] - mu) / sd
    tes[feats] = (tes[feats] - mu) / sd

    idx = {s_: i for i, s_ in enumerate(feats)}
    crit = ["xG", "stability", "counter_resistance"]

    # Team-conditional candidate filtering: the per-team "used formations"
    # set is built from the TRAINING fold only (no leakage). Both the
    # learned ranker and the transition matrix are ranked on the same
    # filtered test set, so the comparison against the baseline stays fair.
    #team_used = team_used_formations(train_dec)
    #tes = filter_candidates_by_team(tes, team_used)

    G_te = groups_of(tes, feats)
    order = tes.groupby("decision", sort=False)["match_id"].first()

    w_full = fit_clogit(groups_of(trs, feats))
    w_crit = fit_clogit(groups_of(trs, crit + ["prior"]))
    w_prior = fit_clogit(groups_of(trs, ["log_trans", "log_freq", "prior"]))
    w_state = fit_clogit(groups_of(trs, ["log_trans_state", "log_trans_opp",
                                          "log_trans_team", "log_team_freq",
                                          "log_freq", "prior"]))

    rng = np.random.default_rng(seed)
    rankers = {
        "random": lambda X: rng.random(len(X)),
        "transition matrix": lambda X: X[:, idx["log_trans"]],
        "planner (equal weights)": lambda X: X[:, [idx[k] for k in crit]].mean(1),
        "learned: critics only": lambda X: X[:, [idx[k] for k in crit + ["prior"]]] @ w_crit,
        "learned: priors only": lambda X: X[:, [idx[k] for k in ("log_trans", "log_freq", "prior")]] @ w_prior,
        "learned: priors + critics": lambda X: X[:, [idx[k] for k in SIGNALS]] @ w_full[:len(SIGNALS)],
        "learned: state-only (no marginal)": lambda X: X[:, [idx[k] for k in ("log_trans_state", "log_trans_opp", "log_trans_team", "log_team_freq", "log_freq", "prior")]] @ w_state,
    }
    # Only try the state-encoding ranker when z_t is actually present.
    if len(feats) > len(SIGNALS):
        rankers["learned: priors + critics + z_t"] = lambda X: X @ w_full

    ranks_ = {name: ranks(G_te, fn) for name, fn in rankers.items()}
    return {"ranks": ranks_, "decision": order.index.to_numpy(), "match": order.to_numpy(),
            "w_full": dict(zip(feats, map(float, w_full))),
            "w_crit": dict(zip(crit + ["prior"], map(float, w_crit)))}

def evaluate(label, cand, train_dec, test_dec):
    c = add_priors(cand, train_dec)
    tr = c[c["decision"].isin(set(train_dec["_dec"]))]
    te = c[c["decision"].isin(set(test_dec["_dec"]))]

    mu = tr[SIGNALS].mean(); sd = tr[SIGNALS].std().replace(0, 1.0)
    trs, tes = tr.copy(), te.copy()
    trs[SIGNALS] = (tr[SIGNALS] - mu) / sd
    tes[SIGNALS] = (te[SIGNALS] - mu) / sd

    feats = prepare_feature_cols(c)
    mu = tr[feats].mean(); sd = tr[feats].std().replace(0, 1.0)
    trs, tes = tr.copy(), te.copy()
    trs[feats] = (trs[feats] - mu) / sd
    tes[feats] = (tes[feats] - mu) / sd

    idx = {s: i for i, s in enumerate(feats)}
    crit = ["xG", "stability", "counter_resistance"]

    # Team-conditional candidate filtering is a variant; not applied here.
    # team_used = team_used_formations(train_dec)
    # tes = filter_candidates_by_team(tes, team_used)

    G_te = groups_of(tes, feats)

    w_full = fit_clogit(groups_of(trs, feats))
    w_crit = fit_clogit(groups_of(trs, crit + ["prior"]))
    w_prior = fit_clogit(groups_of(trs, ["log_trans", "log_freq", "prior"]))
    w_state = fit_clogit(groups_of(trs, ["log_trans_state", "log_trans_opp", "log_freq", "prior"]))

    rankers = {
        "transition matrix": lambda X: X[:, idx["log_trans"]],
        "planner (equal weights)": lambda X: X[:, [idx[k] for k in crit]].mean(1),
        "learned: critics only": lambda X: X[:, [idx[k] for k in crit + ["prior"]]] @ w_crit,
        "learned: priors only": lambda X: X[:, [idx[k] for k in ("log_trans", "log_freq", "prior")]] @ w_prior,
        "learned: priors + critics": lambda X: X[:, [idx[k] for k in SIGNALS]] @ w_full[:len(SIGNALS)],
    }
    if len(feats) > len(SIGNALS):
        rankers["learned: priors + critics + z_t"] = lambda X: X @ w_full
    res = {name: ranks(G_te, fn) for name, fn in rankers.items()}

    print(f"\n=== {label}: {len(G_te)} held-out decisions ===")
    print(f"{'ranker':<28}{'top-1':>8}{'top-3':>8}{'MRR':>8}")
    print("-" * 52)
    table = {}
    for name, rk in res.items():
        m = summarise(rk); table[name] = m
        print(f"{name:<28}{m['top1']:>8.3f}{m['top3']:>8.3f}{m['mrr']:>8.3f}")

    print("\nlearned weights (standardised features), priors + critics:")
    for s, wv in zip(SIGNALS, w_full):
        print(f"   {s:<20}{wv:+.3f}")

    d, lo, hi = paired_boot(res["learned: priors + critics"], res["learned: priors only"])
    verdict = "PASS" if lo > 0 else "FAIL"
    print(f"\nTHE BAR: critics on top of the priors, top-1 gain {d:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
    d2, lo2, hi2 = paired_boot(res["learned: priors + critics"], res["transition matrix"])
    print(f"learned full ranker vs transition matrix alone: {d2:+.3f}  [{lo2:+.3f}, {hi2:+.3f}]")

    # The novel system on its own terms: critics + generator prior, no data
    # prior. This is Algorithm 1 with its weights learned instead of set by hand.
    rng = np.random.default_rng(1)
    res["random"] = ranks(G_te, lambda X: rng.random(len(X)))
    d3, lo3, hi3 = paired_boot(res["learned: critics only"], res["planner (equal weights)"])
    d4, lo4, hi4 = paired_boot(res["learned: critics only"], res["random"])
    print(f"\nnovel system only (critics, no data prior):")
    print(f"   random top-1 {summarise(res['random'])['top1']:.3f}")
    print(f"   learned weights vs the paper's equal weights: {d3:+.3f}  [{lo3:+.3f}, {hi3:+.3f}]")
    print(f"   learned weights vs random:                    {d4:+.3f}  [{lo4:+.3f}, {hi4:+.3f}]")
    return {"table": table, "weights": dict(zip(SIGNALS, map(float, w_full))),
            "critics_over_priors": {"diff": d, "ci": [lo, hi], "verdict": verdict},
            "full_vs_transition": {"diff": d2, "ci": [lo2, hi2]},
            "novel_learned_vs_equal": {"diff": d3, "ci": [lo3, hi3]},
            "novel_learned_vs_random": {"diff": d4, "ci": [lo4, hi4]}}


# ----------------------------------------------------------------- cross-fit
def crossfit(args, cfg, df):
    wmd = os.path.abspath(os.path.expanduser(args.crossfit))
    here = os.getcwd()
    sys.path.insert(0, wmd)
    from run_planner import load_counter_resistance, load_stability_critic
    gkf = GroupKFold(n_splits=cfg.train.n_folds)
    folds = list(gkf.split(df, groups=df["match_id"].to_numpy()))
    gen = TransitionGenerator()
    parts, provenance = [], {}
    for k, (_, te_i) in enumerate(folds):
        te = df.iloc[te_i]
        # the fold files written by make_crossfit_folds.py must be this split
        listed = {int(x) for x in (ln.split("#")[0].strip()
                  for ln in open(os.path.join(wmd, f"crossfit/fold{k}_matches.txt"))) if x}
        if listed != set(te["match_id"].astype(int)):
            raise SystemExit(f"fold {k}: crossfit/fold{k}_matches.txt does not match this "
                             "GroupKFold split; rerun make_crossfit_folds.py")
        cache = args.cache.replace(".parquet", f"_crossfit_fold{k}.parquet")
        ck = os.path.join(wmd, f"checkpoints/world_model_best_excl{k}.pt")
        opp_path = args.opponent_pattern.format(k=k)
        if not os.path.exists(opp_path):
            clean = "clean for this fold" if k == 0 else "it HAS seen this fold's decisions"
            print(f"WARNING fold {k}: {opp_path} missing, using {args.opponent} (the fold-0 "
                  f"model, trained on folds 1-4: {clean}). Run train_opponent.py "
                  "--save-folds for a fully clean result.")
            opp_path = args.opponent
        provenance[k] = {"world_model": ck, "opponent": opp_path, "decisions": int(len(te))}
        if os.path.exists(cache):
            print(f"fold {k}: cached features {cache}")
            parts.append(pd.read_parquet(cache)); continue
        opponent = OpponentCoBehaviorModel.load(opp_path, map_location=args.device)
        os.chdir(wmd)
        try:
            from world_model_adapter import TrainedWorldModel
            wm = TrainedWorldModel(checkpoint_path=ck)
        finally:
            os.chdir(here)
        exf = getattr(wm, "excluded_matches_file", None)
        print(f"fold {k}: world model {ck} (trained without {exf}), opponent {opp_path}")
        if not exf:
            raise SystemExit(f"{ck} records no excluded matches; it is not a cross-fit model")
        cres = load_counter_resistance(wmd, f"results/counter_resistance_lookup_excl{k}.json")
        stab_file = f"results/stability_logistic_excl{k}.json"
        if not os.path.exists(os.path.join(wmd, stab_file)):
            raise SystemExit(f"{stab_file} missing in {wmd}: run_crossfit.sh fits the "
                             "stability critic per fold; rerun it")
        stab = load_stability_critic(wmd, stab_file)
        planner = RAMFPPlanner(opponent, world_model=wm,
                               critics=[XGCritic(wm), stab, cres],
                               cfg=cfg.inference, device=args.device)
        tmp = te.copy(); tmp["triple_id"] = tmp["_dec"]
        parts.append(extract(planner, gen, tmp, cfg, cache))
    cand = pd.concat(parts, ignore_index=True)

    pooled = defaultdict(list); dec, mat = [], []
    per_fold, weights = {}, {}
    for k, (tr_i, te_i) in enumerate(folds):
        r = fit_rank(cand, df.iloc[tr_i], df.iloc[te_i], seed=k)
        for name, rk in r["ranks"].items():
            pooled[name].extend(rk)
        dec.extend(r["decision"]); mat.extend(r["match"])
        per_fold[k] = {n: summarise(rk)["top1"] for n, rk in r["ranks"].items()}
        weights[k] = {"priors+critics": r["w_full"], "critics only": r["w_crit"]}

    n = len(mat)
    print(f"\n=== CROSS-FITTED: {n} decisions, {len(set(mat))} matches, every score out-of-fold ===")
    print(f"{'ranker':<28}{'top-1':>8}{'top-3':>8}{'MRR':>8}   per-fold top-1")
    print("-" * 90)
    table = {}
    for name, rk in pooled.items():
        m = summarise(rk); table[name] = m
        pf = " ".join(f"{per_fold[k][name]:.3f}" for k in per_fold)
        print(f"{name:<28}{m['top1']:>8.3f}{m['top3']:>8.3f}{m['mrr']:>8.3f}   {pf}")

    print("\nlearned weights per fold (standardised features):")
    for k, w in weights.items():
        print(f"  fold {k} critics only : " + "  ".join(f"{a} {b:+.3f}" for a, b in w["critics only"].items()))
    for k, w in weights.items():
        print(f"  fold {k} full         : " + "  ".join(f"{a} {b:+.3f}" for a, b in w["priors+critics"].items()))

    def test(a, b):
        return paired_boot(pooled[a], pooled[b], groups=mat)
    b1 = test("learned: priors + critics", "learned: priors only")
    b2 = test("learned: critics only", "random")
    r1 = test("learned: critics only", "planner (equal weights)")
    r2 = test("learned: priors + critics", "transition matrix")
    r3 = test("planner (equal weights)", "random")
    v = lambda t: "PASS" if t[1] > 0 else "FAIL"
    print("\nmatch-clustered paired bootstrap, top-1 difference [95% CI]")
    print(f"  B1 critics on top of priors      {b1[0]:+.4f} [{b1[1]:+.4f}, {b1[2]:+.4f}]  -> {v(b1)}")
    print(f"  B2 learned critics vs random     {b2[0]:+.4f} [{b2[1]:+.4f}, {b2[2]:+.4f}]  -> {v(b2)}")
    print(f"     learned vs equal weights      {r1[0]:+.4f} [{r1[1]:+.4f}, {r1[2]:+.4f}]")
    print(f"     equal weights vs random       {r3[0]:+.4f} [{r3[1]:+.4f}, {r3[2]:+.4f}]")
    print(f"     full ranker vs transition     {r2[0]:+.4f} [{r2[1]:+.4f}, {r2[2]:+.4f}]")
    return {"n": n, "matches": len(set(mat)), "table": table, "per_fold_top1": per_fold,
            "weights": weights, "provenance": provenance,
            "B1_critics_over_priors": {"diff": b1[0], "ci": b1[1:], "verdict": v(b1)},
            "B2_learned_critics_vs_random": {"diff": b2[0], "ci": b2[1:], "verdict": v(b2)},
            "learned_vs_equal": {"diff": r1[0], "ci": r1[1:]},
            "equal_vs_random": {"diff": r3[0], "ci": r3[1:]},
            "full_vs_transition": {"diff": r2[0], "ci": r2[1:]}}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/processed/triples.parquet")
    ap.add_argument("--opponent", default="models/opponent.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--wm-dir", default=None, help="single world model (contaminated / clean-subset evaluation)")
    ap.add_argument("--crossfit", default=None,
                    help="User dir holding the run_crossfit.sh outputs: fully out-of-fold evaluation")
    ap.add_argument("--opponent-pattern", default="models/opponent_fold{k}.pt")
    ap.add_argument("--cache", default="results/ranker_features.parquet")
    ap.add_argument("--limit", type=int, default=None, help="decisions to extract (timing tests)")
    ap.add_argument("--out", default="results/learned_ranker.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    df = impute_out_of_fold(pd.read_parquet(args.triples).reset_index(drop=True), cfg.train.n_folds)
    df["_dec"] = np.arange(len(df))
    # Pre-compute the opponent-backline bucket on the triples so every consumer
    # of this frame -- add_priors in particular -- can read it without needing
    # to derive it per row. extract() adds the same column to the candidate
    # frame; this makes the two sources consistent.
    if "opp_backline_bucket" not in df.columns:
        df["opp_backline_bucket"] = df["opp_formation"].map(_backline_bucket)
    # team_id is in the triples and has never been used. Add it here so
    # add_priors can condition on which team made the change, not just on
    # the state at the moment.
    if "team_id" in df.columns:
        df["team_id"] = df["team_id"].astype(int)
    if args.limit:
        df = df.iloc[: args.limit].copy()

    if args.crossfit:
        res = crossfit(args, cfg, df)
        out = args.out.replace(".json", "_crossfit.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=2, default=float)
        print(f"\nwrote {out}")
        return
    if not args.wm_dir:
        raise SystemExit("give --wm-dir (single model) or --crossfit (cross-fitted models)")

    opponent = OpponentCoBehaviorModel.load(args.opponent, map_location=args.device)
    here, wmd = os.getcwd(), os.path.abspath(os.path.expanduser(args.wm_dir))
    sys.path.insert(0, wmd); os.chdir(wmd)
    try:
        from world_model_adapter import TrainedWorldModel
        wm = TrainedWorldModel()
    finally:
        os.chdir(here)
    from run_planner import load_counter_resistance
    critics = [XGCritic(wm), StructuralStabilityCritic(), load_counter_resistance(wmd)]
    planner = RAMFPPlanner(opponent, world_model=wm, critics=critics,
                           cfg=cfg.inference, device=args.device)
    gen = TransitionGenerator()

    tmp = df.rename(columns={"_dec": "triple_id_tmp"}).copy()
    tmp["triple_id"] = tmp["triple_id_tmp"]
    cand = extract(planner, gen, tmp, cfg, None if args.limit else args.cache)
    cand = cand.rename(columns={}).copy()
    if args.limit:
        print(f"\nextracted {cand['decision'].nunique()} decisions (timing run only)")
        return

    # split 1: the usual fold, comparable with eval_planner.py
    gkf = GroupKFold(n_splits=cfg.train.n_folds)
    tr_i, te_i = next(iter(gkf.split(df, groups=df["match_id"].to_numpy())))
    results = {"fold0": evaluate("fold-0 (comparable to eval_planner; world model has seen 92% of these matches)",
                                 cand, df.iloc[tr_i], df.iloc[te_i])}

    # split 2: only decisions from matches the world model never trained on
    w = pd.read_parquet(os.path.join(wmd, "data/processed/stable_windows.parquet"),
                        columns=["match_id", "competition_id", "season_id"])
    pairs = set(map(tuple, w[["competition_id", "season_id"]].dropna().drop_duplicates().values))
    test_keys = set()
    for cid in {c for c, _ in pairs}:
        ss = sorted(s for c, s in pairs if c == cid)
        if len(ss) > 1:
            test_keys.add((cid, ss[-1]))
    unseen = set(w[[(c, s) in test_keys for c, s in zip(w.competition_id, w.season_id)]].match_id)
    clean_te = df[df["match_id"].isin(unseen)]
    clean_tr = df[~df["match_id"].isin(unseen)]
    results["clean"] = evaluate("clean (matches the world model never saw)",
                                cand, clean_tr, clean_te)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
