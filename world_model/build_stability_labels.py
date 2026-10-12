#!/usr/bin/env python3
"""
build_stability_labels.py -- labels for the stability critic (paper Eq. 25).

StabilityMLP is specified as predicting concession probability from
[emb(f_src); emb(f_tgt); delta_struct], but nothing in the repository ever
produced a label for it, so the critic in the inference loop is still the
hand-written placeholder `-structural_distance / 100` and every run is flagged
as stubbed.

The label is read off the data, not invented: for each formation change in the
natural-experiments table, did the changing team concede at least one goal in
the window that followed?

Censoring matters here and is easy to get wrong. A change made in the 88th
minute has only a couple of minutes of match left, so "did not concede" says
much less about it than the same observation after a full window. Rows are
therefore kept only when the full window was actually available, and the
script reports how many were dropped for that reason rather than quietly
treating a short window as a clean negative.

    python build_stability_labels.py \
        --experiments ../measurements/processed-data/natural_experiments.csv \
        --events data/processed/master_events.parquet \
        --window 600 \
        --out data/processed/stability_labels.parquet
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="../preprocessing/processed-data/natural_experiments.csv")
    ap.add_argument("--events", default="data/processed/master_events.parquet")
    ap.add_argument("--window", type=float, default=600.0, help="seconds after the change")
    ap.add_argument("--out", default="data/processed/stability_labels.parquet")
    args = ap.parse_args()

    exp = pd.read_csv(args.experiments)
    ev = pd.read_parquet(args.events, columns=["match_id", "team_id", "timestamp_sec", "is_goal"])
    goals = ev[ev["is_goal"]].copy()
    print(f"{len(exp):,} formation changes | {len(goals):,} goals in {ev['match_id'].nunique():,} matches")

    exp["window_available"] = (exp["match_end_seconds"] - exp["change_time"]).clip(lower=0)
    full = exp["window_available"] >= args.window
    print(f"full {args.window:.0f}s window available for {int(full.sum()):,} of {len(exp):,} "
          f"changes ({100 * full.mean():.1f}%); the rest are right-censored and dropped")
    exp = exp[full].copy()

    by_match = {mid: g for mid, g in goals.groupby("match_id")}

    conceded = []
    for _, r in exp.iterrows():
        g = by_match.get(int(r["game_id"]))
        if g is None:
            conceded.append(0)
            continue
        lo, hi = float(r["change_time"]), float(r["change_time"]) + args.window
        against = g[(g["timestamp_sec"] > lo) & (g["timestamp_sec"] <= hi)
                    & (g["team_id"] != int(r["team_id"]))]
        conceded.append(int(len(against) > 0))

    out = pd.DataFrame({
        "match_id": exp["game_id"].astype(int).values,
        "team_id": exp["team_id"].astype(int).values,
        "f_src": exp["old_formation_code"].astype(int).values,
        "f_tgt": exp["new_formation_code"].astype(int).values,
        "change_time": exp["change_time"].values,
        "minute": (exp["change_time"].values / 60.0),
        "score_difference": exp["score_difference"].values,
        "opp_style_cluster": exp["opponent_response_formation_code"].values * 0,  # placeholder column removed below
        "conceded": conceded,
    }).drop(columns=["opp_style_cluster"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)

    rate = out["conceded"].mean()
    print(f"\nlabelled {len(out):,} transitions | base rate {rate:.4f} "
          f"({int(out['conceded'].sum()):,} conceded)")
    print(f"distinct (f_src, f_tgt) pairs: {out.groupby(['f_src','f_tgt']).ngroups}")
    print(f"matches: {out['match_id'].nunique():,}")
    if rate < 0.05 or rate > 0.95:
        print("WARNING: base rate is extreme; accuracy will be meaningless, read AUROC/AP")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
