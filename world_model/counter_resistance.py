#!/usr/bin/env python3
"""
counter_resistance.py -- build C_res (paper Eq. 26) and expose it through the
Critic protocol the inference loop expects.

Why this file exists rather than calling critics.py directly
------------------------------------------------------------
`build_counter_resistance_lookup` in critics.py has two problems that would
have been invisible in the output:

1. It reads `output[0].item()` straight off the model. The world model is
   trained on z-scored targets, so that is a normalised value, not an xG
   delta. The table would have been in standard-deviation units while being
   documented, and read by the critic, as "mean historical xG delta". Here
   the value is denormalised with the checkpoint's own statistics.

2. It runs one forward pass per window over all 164,755 stable windows. The
   table has at most 6 x 6 = 36 cells, so the mean in each is long settled
   after a few hundred windows. A per-cell cap gives the same table in a
   fraction of the time, and the per-cell counts are reported so a cell
   estimated from a handful of windows is visible rather than quietly
   averaged in.

It also extends STRUCTURAL_GROUPS, which covered only the 15 hand-specified
formations. 3412, 41212 and 3511 fell through to "unknown" and scored exactly
0.0 -- a neutral-looking number that is really "no data". Their groups follow
the definitions already in the docstring (back line, then midfield shape):
3412 and 3511 play a back three; 41212 is a 4-back with a lone attacking
midfielder behind the strikers, like 4312/4321.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from critics import CounterResistanceCritic
from models.world_model.graph_builder import build_bipartite_graph

EXTRA_GROUPS = {
    # the three formations missing from the hand taxonomy
    3412:  "3back",
    3511:  "3back",
    41212: "4back_attack_mid",   # lone pivot, one attacking mid behind two forwards, as 4312/4321

    # the remaining five codes in the corpus vocabulary, 0.24% of all formation
    # appearances between them. The back line is unambiguous, so the three
    # back-three shapes join "3back". The two back-four shapes do NOT get forced
    # into an existing four-back family: the hand taxonomy is not self-consistent
    # on its second axis (4141, a lone pivot, sits in "4back_flat_mid" while 4132,
    # also a lone pivot, sits in "4back_double_pivot"), so there is no rule to
    # follow and guessing would put real windows into the wrong cell. They get
    # their own group, which is honest and, at this frequency, harmless.
    32221: "3back",
    3142:  "3back",
    32122: "3back",
    41221: "4back_other",
    42211: "4back_other",
}

TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]


def structural_groups():
    g = dict(CounterResistanceCritic.STRUCTURAL_GROUPS)
    g.update(EXTRA_GROUPS)
    return g


def build_lookup(world_model, windows, rms_df, sty_df, vocab, norm_stats,
                 per_cell_cap=400, seed=0, device="cpu"):
    """{(tgt_group, opp_group): mean xg_delta} plus the per-cell counts."""
    groups = structural_groups()
    vocab_set = set(vocab)
    rng = np.random.default_rng(seed)

    w = windows.copy()
    w["_tgt"] = [groups.get(int(c), "unknown") for c in w["home_formation"]]
    w["_opp"] = [groups.get(int(c), "unknown") for c in w["away_formation"]]
    w = w[(w["home_formation"].isin(vocab_set)) & (w["away_formation"].isin(vocab_set))]

    # Sample by index rather than groupby.apply: from pandas 2.2 apply drops
    # the grouping columns from what the function sees, so the returned frame
    # came back without _tgt/_opp and the next line raised KeyError('_tgt').
    # Taking indices keeps every column and does not depend on that behaviour.
    keep = []
    for _, g in w.groupby(["_tgt", "_opp"], sort=False):
        idx = g.index
        if len(idx) > per_cell_cap:
            idx = rng.choice(idx.to_numpy(), size=per_cell_cap, replace=False)
        keep.extend(list(idx))
    sampled = w.loc[keep]
    print(f"lookup built from {len(sampled):,} of {len(w):,} windows "
          f"(cap {per_cell_cap} per cell)")

    acc = defaultdict(list)
    failures = 0
    world_model.eval()
    with torch.no_grad():
        for _, row in sampled.iterrows():
            ctx = {
                "minute": row.get("minute_bucket", 45),
                "scoreline_home": row.get("scoreline_home", 0),
                "scoreline_away": row.get("scoreline_away", 0),
                "ppda": row.get("ppda", None),
                "home_team_id": row.get("home_team_id", None),
                "away_team_id": row.get("away_team_id", None),
            }
            try:
                data = build_bipartite_graph(int(row["home_formation"]),
                                             int(row["away_formation"]),
                                             ctx, rms_df, sty_df, vocab)
                out = world_model(data.to(device)).reshape(-1)
                # denormalise: the model predicts z-scored targets
                xg = float(out[0]) * norm_stats["xg_delta"]["std"] + norm_stats["xg_delta"]["mean"]
            except Exception:
                failures += 1
                continue
            acc[(row["_tgt"], row["_opp"])].append(xg)

    if failures:
        print(f"  {failures:,} windows failed to build a graph and were skipped")

    lookup = {k: float(np.mean(v)) for k, v in acc.items()}
    counts = {k: len(v) for k, v in acc.items()}
    return lookup, counts


def save(lookup, counts, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({
        "lookup": [{"tgt_group": k[0], "opp_group": k[1], "mean_xg_delta": v,
                    "n_windows": counts[k]} for k, v in sorted(lookup.items())],
        "note": "mean predicted xg_delta in real units (denormalised), by structural family pair",
    }, open(path, "w"), indent=2)


def load(path):
    raw = json.load(open(path))["lookup"]
    return ({(r["tgt_group"], r["opp_group"]): r["mean_xg_delta"] for r in raw},
            {(r["tgt_group"], r["opp_group"]): r["n_windows"] for r in raw})


class CounterResistance:
    """C_res behind the inference loop's Critic protocol.

    critics.CounterResistanceCritic.score takes (f_tgt, f_opp); the protocol
    passes (f_src, f_tgt, f_opp, state, wm_out). This adapts between them and
    refuses to pretend a missing cell is a neutral 0.0.
    """

    name = "counter_resistance"
    is_stub = False

    def __init__(self, lookup, counts=None, min_windows=30):
        self._inner = CounterResistanceCritic(lookup)
        self._inner.STRUCTURAL_GROUPS = structural_groups()
        self.counts = counts or {}
        self.min_windows = min_windows
        self.thin_cells = {k for k, n in self.counts.items() if n < min_windows}

    def score(self, f_src, f_tgt, f_opp, state, wm_out):
        groups = structural_groups()
        key = (groups.get(int(f_tgt), "unknown"), groups.get(int(f_opp), "unknown"))
        if key not in self._inner.lookup:
            # no evidence for this matchup: 0.0 would read as "neutral", which
            # is a claim. Return the table's own mean instead and say nothing
            # stronger than the data supports.
            return float(np.mean(list(self._inner.lookup.values()))) if self._inner.lookup else 0.0
        return float(self._inner.lookup[key])

    def coverage(self):
        return {
            "cells": len(self._inner.lookup),
            "thin_cells_under_%d_windows" % self.min_windows: sorted(
                f"{a}->{b}" for a, b in self.thin_cells),
        }


def main():
    import argparse
    import yaml
    from world_model_adapter import TrainedWorldModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world_model.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/world_model_best.pt")
    ap.add_argument("--per-cell-cap", type=int, default=400)
    ap.add_argument("--out", default="results/counter_resistance_lookup.json")
    ap.add_argument("--exclude-matches", default=None,
                    help="file of match_ids whose windows must not enter the table "
                         "(cross-fitting: use the same list the checkpoint was trained without)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    wm = TrainedWorldModel(checkpoint_path=args.checkpoint)
    windows = pd.read_parquet(cfg["stable_windows_path"])
    if args.exclude_matches:
        with open(args.exclude_matches) as f:
            ex = {int(x) for x in (ln.split("#")[0].strip() for ln in f) if x}
        n0 = len(windows)
        windows = windows[~windows["match_id"].isin(ex)].reset_index(drop=True)
        print(f"CROSS-FIT: excluded {len(ex)} matches, {n0:,} -> {len(windows):,} windows")
    # the role-matchup table the checkpoint was trained with (see the adapter)
    rms = wm.rms
    sty = pd.read_parquet(cfg["style_clusters_path"])
    vocab = {int(c): i for i, c in enumerate(json.load(open(cfg["formation_vocab_path"]))["vocab"])}

    lookup, counts = build_lookup(wm.model, windows, rms, sty, vocab, wm.norm,
                                  per_cell_cap=args.per_cell_cap,
                                  device=str(wm.device))
    save(lookup, counts, args.out)

    print(f"\n{'tgt group':<24}{'opp group':<24}{'mean xg_delta':>15}{'n':>8}")
    for (a, b), v in sorted(lookup.items(), key=lambda kv: -kv[1]):
        print(f"{a:<24}{b:<24}{v:>15.4f}{counts[(a,b)]:>8}")
    thin = [k for k, n in counts.items() if n < 30]
    if thin:
        print(f"\n{len(thin)} cell(s) estimated from fewer than 30 windows: "
              + ", ".join(f"{a}->{b}" for a, b in thin))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
