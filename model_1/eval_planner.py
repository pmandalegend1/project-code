#!/usr/bin/env python3
"""
eval_planner.py -- the first end-to-end evaluation of the planner.

Why this exists
---------------
Every result in this project so far measures one component against its own
baseline: the opponent model against persistence, the world model against ridge,
the stability critic against a logistic on structural distance, the conformal
terminator against a fixed top-k. Nothing measured the thing we actually built.
The planner emits ranked formation changes and no number described how good that
ranking is, which is the first question a reader will ask.

The ground truth we have
------------------------
`triples.parquet` records, for each decision, the formation the team was in
(`f_src`) and the one it actually moved to (`f_tgt`). So the question we can
answer is: on decisions the model never trained on, how highly does the planner
rank the change the coach actually made?

That is a proxy, and we say so: what a coach did is not necessarily what was
optimal, and a planner that disagrees with a coach is not thereby wrong. But it
is the only observable signal available without a counterfactual, it is the one
used elsewhere in this literature, and a planner that cannot rank real decisions
above chance is not doing anything.

The evaluation deliberately does NOT score recommendations by the world model's
own predicted xG. That would ask the world model to mark its own homework, and
the world model is one of the things under test.

Two numbers before the ranking
------------------------------
The generator filters candidates by tactical grammar and returns at most
`candidate_k`. If the real transition is never proposed, the planner cannot rank
it and every ranking metric below is capped. So we report grammar coverage and
candidate-set coverage first; they bound everything else, and a low value is
itself a finding about the grammar rather than about the ranker.

Baselines, all scored on exactly the same candidate sets
--------------------------------------------------------
  random        shuffle the candidates
  frequency     rank by how often each target appears in training, ignoring state
  transition    rank by P(f_tgt | f_src) estimated on training decisions -- the
                strong baseline, and the one persistence is a special case of
  prior         the generator's own hand-written prior, with no critics and no
                opponent model: what the pipeline scores before it does anything
  planner       the whole thing

    python eval_planner.py --wm-dir ../world_model --n 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ramfp.data.ppda_impute import impute_out_of_fold
import torch
from sklearn.model_selection import GroupKFold

from ramfp.config import Config
from ramfp.inference import GameState, RAMFPPlanner, TransitionGenerator
from ramfp.models.opponent import OpponentCoBehaviorModel
from ramfp.vocab import FORMATIONS
from ramfp.eval.planner_diagnostics import AgreementTracker, effective_weights_as_used

K_AT = (1, 3, 5)


def state_from_row(row):
    def seq(col):
        v = row[col]
        return list(v) if v is not None and len(np.atleast_1d(v)) else []
    return GameState(
        f_src=int(row["f_src"]), opp_formation=int(row["opp_formation"]),
        opp_style=int(row["opp_style"]), goal_diff=int(row["goal_diff"]),
        minute=int(row["minute"]), opp_ppda=float(row["opp_ppda_pre"]),
        phase=str(row["phase"]),
        event_type=seq("event_type"), event_zone=seq("event_zone"),
        event_team=seq("event_team"), event_minute=seq("event_minute"),
        event_outcome=seq("event_outcome"),
    )


def rank_of(ordered_targets, truth):
    """1-based rank of `truth` in a list of distinct targets, or None."""
    for i, t in enumerate(ordered_targets, start=1):
        if t == truth:
            return i
    return None


def dedup(pairs):
    """(target, score) -> targets ordered by best score, each once."""
    best = {}
    for t, s in pairs:
        if t not in best or s > best[t]:
            best[t] = s
    return [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])]


def summarise(ranks, n_total):
    out = {}
    for k in K_AT:
        out[f"top{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n_total
    out["mrr"] = sum(1.0 / r for r in ranks if r is not None) / n_total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/processed/triples.parquet")
    ap.add_argument("--opponent", default="models/opponent.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--wm-dir", default=None, help="omit to run with the stub world model")
    ap.add_argument("--crossfit-fold0", action="store_true",
                    help="use the world model, C_res table and stability critic trained "
                         "WITHOUT fold 0's matches (run_crossfit.sh outputs). Without this "
                         "the main world model has seen most of the evaluated matches.")
    ap.add_argument("--n", type=int, default=300, help="held-out decisions to evaluate")
    ap.add_argument("--out", default="results/planner_eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    df = impute_out_of_fold(pd.read_parquet(args.triples), cfg.train.n_folds)

    gkf = GroupKFold(n_splits=cfg.train.n_folds)
    tr_idx, te_idx = next(iter(gkf.split(df, groups=df["match_id"].to_numpy())))
    train, held = df.iloc[tr_idx], df.iloc[te_idx].reset_index(drop=True)
    # The conformal terminator is calibrated on the first half of fold 0's
    # matches (run_planner.fit_terminator). Evaluate only on the OTHER half,
    # so no evaluated decision was used to calibrate the stopping rule.
    _m = np.unique(held["match_id"].to_numpy())
    _cal = set(_m[: len(_m) // 2])
    held = held[~held["match_id"].isin(_cal)].reset_index(drop=True)
    rows = held.sample(min(args.n, len(held)), random_state=0).reset_index(drop=True)
    print(f"{len(df):,} decisions | fold-0 held out: {len(held):,} "
          f"| evaluating {len(rows):,} (model never saw these)\n")

    # ---- baselines estimated on the training decisions only ----------------
    freq = Counter(train["f_tgt"].astype(int))
    trans = defaultdict(Counter)
    for a, b in zip(train["f_src"].astype(int), train["f_tgt"].astype(int)):
        trans[a][b] += 1

    # ---- coverage: can the generator even propose the real answer? ---------
    gen = TransitionGenerator()
    in_vocab = in_grammar = in_candidates = 0
    vocab = set(FORMATIONS)
    for _, r in rows.iterrows():
        truth = int(r["f_tgt"])
        if truth in vocab:
            in_vocab += 1
        st = state_from_row(r)
        full = {c.f_tgt for c in gen.generate(st, k=10_000)}
        topk = {c.f_tgt for c in gen.generate(st, k=cfg.inference.candidate_k)}
        in_grammar += truth in full
        in_candidates += truth in topk

    n = len(rows)
    print("Coverage — these cap every ranking number below")
    print(f"  real target is in the vocabulary        {in_vocab}/{n}  ({in_vocab/n:.1%})")
    print(f"  ... and survives the tactical grammar   {in_grammar}/{n}  ({in_grammar/n:.1%})")
    print(f"  ... and is inside the top-{cfg.inference.candidate_k} candidates  "
          f"{in_candidates}/{n}  ({in_candidates/n:.1%})")
    print()

    # ---- world model + planner ---------------------------------------------
    opponent = OpponentCoBehaviorModel.load(args.opponent, map_location=args.device)
    wm = None
    if args.wm_dir:
        here, wmd = os.getcwd(), os.path.abspath(os.path.expanduser(args.wm_dir))
        sys.path.insert(0, wmd); os.chdir(wmd)
        try:
            from world_model_adapter import TrainedWorldModel
            wm = TrainedWorldModel(
                checkpoint_path="checkpoints/world_model_best_excl0.pt"
                if args.crossfit_fold0 else "checkpoints/world_model_best.pt")
        finally:
            os.chdir(here)
        print(f"world model: trained (val_mse_total {wm.provenance()['val_mse_total']:.4f})\n")
    else:
        print("world model: STUB — planner numbers below are plumbing, not a result\n")

    # Give the planner its best shot: fitted terminator, all three critics.
    # Evaluating it half-wired would make the comparison unfair to it.
    from run_planner import fit_terminator, load_counter_resistance, load_stability_critic
    term = fit_terminator(opponent, df, cfg, args.device,
                          cfg.inference.conformal_alpha, cfg.inference.conformal_threshold)
    critics = None
    if args.wm_dir:
        from ramfp.inference import StructuralStabilityCritic, XGCritic
        if args.crossfit_fold0:
            cres = load_counter_resistance(args.wm_dir, "results/counter_resistance_lookup_excl0.json")
            stab = load_stability_critic(args.wm_dir, "results/stability_logistic_excl0.json")
        else:
            cres = load_counter_resistance(args.wm_dir)
            stab = load_stability_critic(args.wm_dir)
        if cres is not None:
            critics = [XGCritic(wm), stab, cres]

    planner = RAMFPPlanner(opponent, world_model=wm, critics=critics,
                           cfg=cfg.inference, terminator=term, device=args.device)
    print("effective critic weights: "
          + ", ".join(f"{k} {v:.3f}" for k, v in planner.effective_critic_weights().items()))
    print()

    METHODS = ["random", "frequency", "transition", "prior", "planner"]
    results = {k: [] for k in METHODS}
    critic_results = defaultdict(list)   # one ranking per critic, alone
    agreement = AgreementTracker()
    eff_w = defaultdict(list)          # effective weight per decision, as aggregated
    n_targets = []                     # distinct formations ranked per decision
    N_REVISE = 3                       # candidates revised by Algorithm 1 (not the list length)
    rng = np.random.default_rng(0)

    for i, (_, r) in enumerate(rows.iterrows(), 1):
        st = state_from_row(r)
        truth = int(r["f_tgt"])
        cands = gen.generate(st, k=cfg.inference.candidate_k)
        if not cands:
            for key in results:
                results[key].append(None)
            continue
        targets = list({c.f_tgt for c in cands})

        order = list(targets); rng.shuffle(order)
        results["random"].append(rank_of(order, truth))
        results["frequency"].append(rank_of(
            sorted(targets, key=lambda t: -freq.get(t, 0)), truth))
        row_t = trans.get(int(r["f_src"]), Counter())
        results["transition"].append(rank_of(
            sorted(targets, key=lambda t: -row_t.get(t, 0)), truth))
        results["prior"].append(rank_of(
            dedup([(c.f_tgt, c.prior) for c in cands]), truth))

        # plan() returns EVERY candidate ranked; top_n only says how many of
        # them go through the revision loop. The audit flagged top-5 as invalid
        # because "only top 3 are generated" -- that is the revision count, not
        # the list length. We check it explicitly on every decision instead of
        # assuming it: the planner must rank exactly the targets the baselines rank.
        recs = planner.plan(st, top_n=N_REVISE)
        ranked = {x.f_tgt for x in recs}
        if ranked != set(targets):
            raise RuntimeError(f"planner ranked {len(ranked)} targets but the candidate set "
                               f"has {len(targets)}; top-k would not be comparable")
        n_targets.append(len(ranked))
        results["planner"].append(rank_of(
            dedup([(x.f_tgt, x.score) for x in recs]), truth))

        # each critic alone, so a single bad one cannot hide inside the sum
        for cname in (recs[0].critic_scores if recs else {}):
            critic_results[cname].append(rank_of(
                dedup([(x.f_tgt, x.critic_scores[cname]) for x in recs]), truth))

        # diagnostics from the planner's real critic outputs (see planner_diagnostics)
        if recs:
            agreement.add(recs)
            for k, v in effective_weights_as_used(
                    recs, planner._normalise_critics,
                    planner.cfg.critic_weights).items():
                eff_w[k].append(v)

        if i % 50 == 0:
            print(f"  {i}/{n} decisions planned", flush=True)

    print(f"\n{'method':<14}{'top-1':>9}{'top-3':>9}{'top-5':>9}{'MRR':>9}")
    print("-" * 50)
    summary = {}
    for name in METHODS:
        m = summarise(results[name], n)
        summary[name] = m
        print(f"{name:<14}{m['top1']:>9.3f}{m['top3']:>9.3f}{m['top5']:>9.3f}{m['mrr']:>9.3f}")
    if critic_results:
        print("-" * 50)
        for cname, ranks in critic_results.items():
            m = summarise(ranks, n)
            summary[f"critic_only:{cname}"] = m
            print(f"{('  only ' + cname):<14}{m['top1']:>9.3f}{m['top3']:>9.3f}"
                  f"{m['top5']:>9.3f}{m['mrr']:>9.3f}")
    print("-" * 50)
    print(f"{'ceiling':<14}{in_candidates/n:>9.3f}{in_candidates/n:>9.3f}"
          f"{in_candidates/n:>9.3f}{'':>9}")

    # --- top-k validity -------------------------------------------------------
    nt = np.array(n_targets) if n_targets else np.array([0])
    short = {k: int((nt < k).sum()) for k in K_AT}
    print(f"\ndistinct formations ranked per decision: median {np.median(nt):.0f}, "
          f"min {nt.min()}, max {nt.max()}")
    for k in K_AT:
        print(f"  decisions with fewer than {k} targets ranked: {short[k]} of {len(nt)}")
    summary["ranked_targets"] = {"median": float(np.median(nt)), "min": int(nt.min()),
                                 "max": int(nt.max()), "fewer_than_k": short,
                                 "n_revised_per_decision": N_REVISE}

    # --- critic diagnostics, measured on what the planner actually used -------
    if eff_w:
        print(f"\n{'critic':<22}{'effective weight (as aggregated)':>34}{'configured':>12}")
        print("-" * 68)
        conf = planner.effective_critic_weights()
        for k in sorted(eff_w):
            print(f"{k:<22}{np.mean(eff_w[k]):>33.1%}{conf.get(k, 0):>12.1%}")
        summary["critic_effective_weights"] = {k: float(np.mean(v)) for k, v in eff_w.items()}
    ag = agreement.summary()
    summary["critic_agreement"] = ag
    print(f"\ncritic agreement: {ag['decisions']} decisions, status {ag['status']}")
    if ag["dead_critics"]:
        print(f"  DEAD (constant in every decision, carries no ranking): {ag['dead_critics']}")
    elif ag["constant_in_decisions"]:
        print(f"  constant in some decisions: {ag['constant_in_decisions']}")
    if ag["pairs"]:
        print(f"{'critic pair (Spearman rho per decision, by target)':<52}{'mean rho':>10}{'n':>6}")
        print("-" * 68)
        for pair, v in ag["pairs"].items():
            print(f"{pair.replace('|', '  vs  '):<52}{v['mean_rho']:>10.3f}{v['decisions']:>6}")
    else:
        print("  no decision had two non-constant critics: agreement is NOT measured")

    gap = summary["planner"]["top1"] - summary["transition"]["top1"]
    print(f"\nplanner minus transition-matrix baseline, top-1: {gap:+.3f}")
    print("The transition matrix is the bar: it uses the current formation and "
          "nothing else.\nBeating 'random' only says the ranking is not arbitrary.")

    summary["coverage"] = {
        "in_vocabulary": in_vocab / n, "in_grammar": in_grammar / n,
        "in_candidate_set": in_candidates / n, "n_evaluated": n,
    }
    summary["world_model"] = "trained" if wm is not None else "stub"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
