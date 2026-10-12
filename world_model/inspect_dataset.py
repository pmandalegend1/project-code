#!/usr/bin/env python3
"""
inspect_dataset.py -- Step 1a: lightweight audit of the world model's inputs.

Reports: window schema, target distributions and raw-value bounds, formation-pair
coverage (the 462 pairs behind E-B), rare formations (E-C), match counts (the
clustering unit for the bootstrap / E-A subsampling), and, if the repo's graph
builder is importable, the tensor shapes and context dimensionality of one graph.

    python inspect_dataset.py --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]
RARE_E_C = [3412, 41212, 3511]


def hdr(s):
    print(f"\n=== {s} " + "=" * max(0, 70 - len(s)))


def inspect_windows(w):
    hdr("stable_windows")
    print(f"rows={len(w):,}  cols={len(w.columns)}  matches={w['match_id'].nunique():,}")
    print("columns:", list(w.columns))
    per_match = w.groupby("match_id").size()
    print(f"windows/match: mean={per_match.mean():.1f} min={per_match.min()} max={per_match.max()}")

    hdr("targets (raw units)")
    print(f"{'target':<16}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}{'NaN':>9}")
    for t in TARGETS:
        if t not in w.columns:
            print(f"{t:<16} MISSING")
            continue
        c = w[t].astype(float)
        print(f"{t:<16}{c.mean():>10.3f}{c.std():>10.3f}{c.min():>10.3f}{c.max():>10.3f}{int(c.isna().sum()):>9,}")
    if "possession_pct" in w:
        p = w["possession_pct"].astype(float)
        scale = "FRACTION 0-1 (centre 0.5)" if p.max() <= 1.0 else "PERCENT 0-100 (centre 50)"
        print(f"possession_pct scale: {scale}")
    if "xg_delta" in w:
        x = w["xg_delta"].astype(float)
        print(f"xg_delta mean={x.mean():+.4f} (antisymmetry is only sensible if ~0), "
              f"frac exactly 0 = {(x == 0).mean():.1%}")

    hdr("context-ish columns")
    ctx_cols = [c for c in ("minute_bucket", "period", "scoreline_home", "scoreline_away",
                            "ppda", "competition_id", "season_id", "match_date") if c in w.columns]
    print("available:", ctx_cols)
    if "minute_bucket" in w:
        print(f"minute_bucket (window start, minutes): min={w['minute_bucket'].min()} max={w['minute_bucket'].max()}")
    if {"scoreline_home", "scoreline_away"} <= set(w.columns):
        d = (w["scoreline_home"] - w["scoreline_away"]).clip(-3, 3)
        print("score diff (clipped ±3):", d.value_counts().sort_index().to_dict())

    hdr("formations / pairs (E-B, E-C)")
    hf, af = w["home_formation"].astype(int), w["away_formation"].astype(int)
    forms = pd.concat([hf, af]).value_counts()
    print(f"{len(forms)} distinct formations; top 8: {forms.head(8).to_dict()}")
    pairs = pd.Series(list(zip(hf, af))).value_counts()
    print(f"{len(pairs)} distinct ordered (home, away) pairs; "
          f"windows/pair median={int(pairs.median())} min={int(pairs.min())} max={int(pairs.max())}")
    print(f"pairs with <30 windows: {(pairs < 30).sum()}  (noisy cell means for the pair-cell loss)")
    for f in RARE_E_C:
        n = int((forms.get(f, 0)))
        in_pairs = int(((hf == f) | (af == f)).sum())
        print(f"  E-C candidate {f}: {n:,} appearances, {in_pairs:,} windows")
    if "minute_bucket" in w:
        blk = (w["minute_bucket"].astype(float) // 15).astype(int)
        cells = w.assign(hp=list(zip(hf, af)), blk=blk).groupby(["hp", "blk"]).size()
        print(f"(pair, 15-min block) cells: {len(cells):,}, median size {int(cells.median())}, "
              f"cells with <5 windows: {(cells < 5).sum()}")
    print("all formations:", sorted(forms.index.tolist()))
    _check_bank(sorted(forms.index.tolist()))

    # same-formation windows: needed for swap sanity (f(A,A) should give exact 0 xg / 50 poss)
    print(f"mirror matchups (home==away formation): {(hf == af).mean():.1%} of windows")


def _check_bank(codes):
    hdr("formation bank (model inputs)")
    try:
        sys.path.insert(0, "models")
        sys.path.insert(0, "models/world_model")
        from dense_data import FormationBank
        canon = json.load(open("models/world_model/canonical_roles.json"))
        bank = FormationBank(codes, canon)
        fb = [c for c, s in bank.source.items() if s == "fallback"]
        print(f"OK: {len(codes)} formations built; hand-specified: {len(codes) - len(fb)}, fallback: {fb}")
    except Exception as e:  # noqa: BLE001
        print(f"PROBLEM building formation bank: {type(e).__name__}: {e}")


def inspect_graph(args, w):
    hdr("graph builder (optional)")
    try:
        import torch  # noqa: F401
        from models.world_model.graph_builder import build_bipartite_graph
    except Exception as e:  # noqa: BLE001
        print(f"skipped: cannot import graph_builder ({type(e).__name__}: {e})")
        return
    rms = pd.read_parquet(os.path.join(args.data_dir, "role_matchup_stats.parquet"))
    sty = pd.read_parquet(os.path.join(args.data_dir, "team_style_clusters.parquet"))
    vocab = json.load(open(os.path.join(args.data_dir, "formation_vocab.json")))["vocab"]
    vmap = {int(c): i for i, c in enumerate(vocab)}
    r = w.iloc[0]
    ctx = {"minute": r.get("minute_bucket", 45), "scoreline_home": r.get("scoreline_home", 0),
           "scoreline_away": r.get("scoreline_away", 0), "ppda": None,
           "home_team_id": r.get("home_team_id"), "away_team_id": r.get("away_team_id")}
    g = build_bipartite_graph(int(r.home_formation), int(r.away_formation), ctx, rms, sty, vmap)
    for k in g.keys() if callable(getattr(g, "keys", None)) else dir(g):
        v = getattr(g, k, None)
        if hasattr(v, "shape"):
            print(f"  g.{k:<14} shape={tuple(v.shape)} dtype={v.dtype}")
    x = getattr(g, "x", None)
    if x is not None:
        print(f"node_feat_dim={x.shape[-1]} (config says 80). Inspect which columns are context "
              f"(constant across the 22 nodes):")
        const = (x.std(dim=0) == 0).sum().item()
        print(f"  {const}/{x.shape[-1]} feature columns are constant across nodes "
              f"-> candidates for broadcast context leakage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    args = ap.parse_args()
    path = os.path.join(args.data_dir, "stable_windows.parquet")
    if not os.path.exists(path):
        print(f"missing {path}")
        return 1
    w = pd.read_parquet(path)
    inspect_windows(w)
    for name in ("role_matchup_stats.parquet", "team_style_clusters.parquet"):
        p = os.path.join(args.data_dir, name)
        if os.path.exists(p):
            d = pd.read_parquet(p)
            hdr(name)
            print(f"rows={len(d):,} cols={list(d.columns)}")
    inspect_graph(args, w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
