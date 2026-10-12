#!/usr/bin/env python3
"""
run_planner.py -- the whole pipeline in one command.

Loads the trained opponent co-behaviour model and the trained bipartite world
model, wires them into RAMFPPlanner in place of StubWorldModel, and plans on
real decision points taken from the triples table.

This is the only configuration whose output may be reported. With the stub in
place, every critic score is a deterministic function of formation geometry,
and `Recommendation.explain()` prints "[STUB WORLD MODEL -- not a result]" on
every line to say so. If you see that flag, the world model did not load and
nothing below it means anything.

    python run_planner.py \
        --opponent models/opponent.pt \
        --triples data/processed/triples.parquet \
        --wm-dir ~/user/data/processed/world_model \
        --n 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


def fit_terminator(opponent, df, cfg, device, alpha, threshold):
    """Fit the conformal terminator on data the opponent model never trained on.

    Left unfitted, `ConformalTerminator.qhat` is None, `set_size()` returns 0
    and `should_stop()` is always False -- the termination rule sits in the
    loop switched off, and every run silently spends its full revision budget.
    Every planner run before this did exactly that.

    The shipped checkpoint is seed 0, fold 0 of a deterministic GroupKFold(5)
    over match_id, so fold 0's held-out rows are recoverable exactly. Half of
    those matches calibrate; the planner then runs on decisions from anywhere,
    which is the intended use.
    """
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader

    from ramfp.data.datasets import TripleDataset, collate
    from ramfp.inference import ConformalTerminator
    from ramfp.train.train_opponent import predict

    device = next(opponent.parameters()).device   # trust the model, not a string
    gkf = GroupKFold(n_splits=cfg.train.n_folds)
    _, te_idx = next(iter(gkf.split(df, groups=df["match_id"].to_numpy())))
    held = df.iloc[te_idx].reset_index(drop=True)
    matches = np.unique(held["match_id"].to_numpy())
    cal_matches = set(matches[: len(matches) // 2])
    cal = held[held["match_id"].isin(cal_matches)]

    dl = DataLoader(TripleDataset(cal, cfg.encoder.seq_len, mirror=False, knn=cfg.encoder.knn),
                    batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate)
    _, mixture, _, y, _ = predict(opponent, dl, device)

    term = ConformalTerminator(alpha, threshold).fit(mixture, y)
    print(f"conformal  : fitted on {len(cal):,} held-out calibration decisions "
          f"({len(cal_matches):,} matches), alpha={alpha}, qhat={term.qhat:.4f}, "
          f"theta_conf={threshold}")
    return term


def load_counter_resistance(wm_dir, lookup_path="results/counter_resistance_lookup.json"):
    """C_res, the third critic in Algorithm 1, which was never wired in.

    Without it the planner runs two critics against three configured weights,
    silently renormalising to 0.5/0.5 -- see RAMFPPlanner._aggregate. Returns
    None if the lookup has not been built, so the planner still runs and the
    warning still fires rather than the run failing.
    """
    wm_dir = os.path.abspath(os.path.expanduser(wm_dir))
    path = lookup_path if os.path.isabs(lookup_path) else os.path.join(wm_dir, lookup_path)
    if not os.path.exists(path):
        print(f"C_res      : lookup not found at {path}; running with two critics")
        return None

    here = os.getcwd()
    sys.path.insert(0, wm_dir)
    os.chdir(wm_dir)
    try:
        from counter_resistance import CounterResistance, load as load_lookup
        lookup, counts = load_lookup(path)
        critic = CounterResistance(lookup, counts)
    finally:
        os.chdir(here)
    cov = critic.coverage()
    print(f"C_res      : {cov['cells']} structural matchup cells")
    thin = [v for k, v in cov.items() if k.startswith("thin_cells")][0]
    if thin:
        print(f"             thin (<30 windows): {', '.join(thin)}")
    return critic


def load_stability_critic(wm_dir, filename="results/stability_logistic.json"):
    """C_stab. Team-lead audit ("stability critic capacity"): use the simpler
    baseline -- a logistic regression on structural distance, fitted by
    User's train_stability.py -- instead of the unlearned -distance/100
    placeholder, whose sign assumption scored below chance. Falls back to the
    placeholder (and says so) when the fitted critic has not been built."""
    from ramfp.inference import StructuralStabilityCritic
    if wm_dir:
        wm_dir = os.path.abspath(os.path.expanduser(wm_dir))
        here = os.getcwd()
        sys.path.insert(0, wm_dir)
        os.chdir(wm_dir)
        try:
            from stability_critic import load as _load
            critic = _load(wm_dir, filename)
        except ImportError:
            critic = None
        finally:
            os.chdir(here)
        if critic is not None:
            print(f"C_stab     : logistic on structural distance ({critic.path}), "
                  f"P(concede)=sigmoid({critic.a:+.3f} {critic.b:+.3f}*d)")
            return critic
    print("C_stab     : fitted critic not found; using the -distance/100 placeholder (stub)")
    return StructuralStabilityCritic()


def load_world_model(wm_dir, checkpoint=None):
    """Import User's adapter from its own directory.

    Its modules resolve as `models.world_model.*` and it reads
    data/processed/ relative to the working directory, so we enter that
    directory for the import and the construction, then come back.
    """
    wm_dir = os.path.abspath(os.path.expanduser(wm_dir))
    if not os.path.isdir(wm_dir):
        raise SystemExit(f"--wm-dir {wm_dir} does not exist")

    here = os.getcwd()
    sys.path.insert(0, wm_dir)
    os.chdir(wm_dir)
    try:
        from world_model_adapter import TrainedWorldModel
        wm = TrainedWorldModel(
            checkpoint_path=checkpoint or "checkpoints/world_model_best.pt"
        )
    finally:
        os.chdir(here)
    return wm


def state_from_row(row, GameState):
    def seq(col):
        v = row[col]
        return list(v) if v is not None and len(np.atleast_1d(v)) else []

    return GameState(
        f_src=int(row["f_src"]),
        opp_formation=int(row["opp_formation"]),
        opp_style=int(row["opp_style"]),
        goal_diff=int(row["goal_diff"]),
        minute=int(row["minute"]),
        opp_ppda=float(row["opp_ppda_pre"]),
        phase=str(row["phase"]),
        event_type=seq("event_type"), event_zone=seq("event_zone"),
        event_team=seq("event_team"), event_minute=seq("event_minute"),
        event_outcome=seq("event_outcome"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="models/opponent.pt")
    ap.add_argument("--triples", default="data/processed/triples.parquet")
    ap.add_argument("--wm-dir", default=None,
                    help="directory holding world_model_adapter.py; omit to run with the stub")
    ap.add_argument("--wm-checkpoint", default=None)
    ap.add_argument("--n", type=int, default=5, help="decision points to plan on")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--no-cres", action="store_true",
                    help="run without the counter-resistance critic")
    ap.add_argument("--no-conformal", action="store_true",
                    help="leave the terminator unfitted (the old behaviour)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from ramfp.inference import GameState, RAMFPPlanner
    from ramfp.models.opponent import OpponentCoBehaviorModel

    print(f"device: {args.device}")
    from ramfp.config import Config
    from ramfp.data.ppda_impute import impute_out_of_fold
    df = impute_out_of_fold(pd.read_parquet(args.triples), Config.load(args.config).train.n_folds)
    opponent = OpponentCoBehaviorModel.load(args.opponent, map_location=args.device)
    print(f"opponent model: {args.opponent}")

    wm = None
    if args.wm_dir:
        wm = load_world_model(args.wm_dir, args.wm_checkpoint)
        prov = wm.provenance()
        print(f"world model   : trained, val_mse_total={prov['val_mse_total']:.4f}")
        for t, r in (prov["val_pearson_per_target"] or {}).items():
            print(f"                {t:<16} val r = {r:.3f}")
        print()
        wm.consistency_check()
    else:
        print("world model   : STUB -- output below is plumbing, not a result")
    print()

    # plan on decisions where the opponent actually responded, since those are
    # the only ones where the prediction is being asked a real question
    moved = df[df["opp_response"] != df["opp_formation"]]
    pool = moved if len(moved) >= args.n else df
    rows = pool.sample(min(args.n, len(pool)), random_state=0)
    print(f"planning on {len(rows)} decision points "
          f"({'opponent did shift' if len(moved) >= args.n else 'all decisions'})\n")

    critics = None
    if args.wm_dir and not args.no_cres:
        from ramfp.inference import StructuralStabilityCritic, XGCritic
        cres = load_counter_resistance(args.wm_dir)
        if cres is not None:
            critics = [XGCritic(wm), load_stability_critic(args.wm_dir), cres]

    from ramfp.config import Config
    cfg = Config.load(args.config)
    term = None
    if not args.no_conformal:
        term = fit_terminator(opponent, df, cfg, args.device,
                              cfg.inference.conformal_alpha,
                              cfg.inference.conformal_threshold)

    planner = RAMFPPlanner(opponent, world_model=wm, critics=critics,
                           cfg=cfg.inference, terminator=term, device=args.device)
    print(f"effective critic weights: "
          + ", ".join(f"{k} {v:.3f}" for k, v in planner.effective_critic_weights().items()))
    print()

    for _, row in rows.iterrows():
        state = state_from_row(row, GameState)
        recs = planner.plan(state, top_n=args.top)
        print(f"--- match {int(row['match_id'])}, minute {int(row['minute'])}, "
              f"{row['f_src']} vs {row['opp_formation']} (gd {int(row['goal_diff']):+d}, "
              f"style {int(row['opp_style'])})")
        print(f"    what the opponent actually did: {int(row['opp_response'])}")
        for r in recs:
            print("    " + r.explain().replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
