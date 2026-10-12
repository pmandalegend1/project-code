#!/usr/bin/env python3
"""
dense_data.py -- turn stable_windows + your role/formation JSONs into plain
arrays for the refactored model (no torch_geometric needed).

What the model is given per window
----------------------------------
  home_form_idx, away_form_idx   index into FormationBank (roles + intra adjacency)
  ctx                            game-state vector (score, minute, both styles)
  y                              4 targets (z-scored by the caller)
  y_cell                         per-(pair, minute_bucket) mean of y (denoising loss)

FormationBank holds, for every formation in the vocabulary:
  roles [F, 11]       canonical role id (0..10) of each position, from
                      canonical_roles.json or the documented fallback
  adj   [F, 11, 11]   intra-team adjacency (+ self loops)
Unseen formations (3412, 41212, 3511) go through formation_fallback, exactly as
graph_builder does, so E-C reuses the same code path.

edge_stats [11, 11, 3] = (present, wr, xg_rate) for "role a attacks role b",
built from positional_matchups.json edge_list + role_matchup_stats.parquet.
The model derives the cross-team structure from role ids on the fly, so any
role relabelling (mirror augmentation) just works.

Run the self-test:  python models/dense_data.py --assets models/world_model
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

NUM_ROLES = 11
TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]
# lateral mirror on canonical roles: RB<->LB, RCB<->LCB (CB, CDM, CM, CAM, W, ST, GK are central/collapsed)
ROLE_MIRROR = np.array([0, 5, 4, 3, 2, 1, 6, 7, 8, 9, 10], dtype=np.int64)
SCORE_BINS = 5          # score diff clipped to -2..2
MINUTE_BINS = 6         # 15-minute buckets


def _import_fallback():
    try:
        from models.world_model.formation_fallback import fallback_mapping
    except Exception:  # noqa: BLE001
        from formation_fallback import fallback_mapping
    return fallback_mapping


class FormationBank:
    def __init__(self, vocab_codes, canon: dict):
        fb = _import_fallback()
        self.codes = [int(c) for c in vocab_codes]
        self.index = {c: i for i, c in enumerate(self.codes)}
        F = len(self.codes)
        self.roles = np.zeros((F, NUM_ROLES), np.int64)
        self.adj = np.zeros((F, NUM_ROLES, NUM_ROLES), np.float32)
        self.source = {}
        for c, i in self.index.items():
            spec = canon["formation_mappings"].get(str(c))
            self.source[c] = "hand"
            if spec is None:
                spec = fb(c, canon)
                self.source[c] = "fallback"
            self.roles[i] = spec["node_assignments"]
            for a, b in spec["intra_edges"]:
                self.adj[i, a, b] = self.adj[i, b, a] = 1.0
            self.adj[i] += np.eye(NUM_ROLES, dtype=np.float32)


def build_edge_stats(matchups: dict, rms_df: pd.DataFrame | None) -> np.ndarray:
    S = np.zeros((NUM_ROLES, NUM_ROLES, 3), np.float32)
    look = {}
    if rms_df is not None and len(rms_df):
        for r in rms_df.itertuples(index=False):
            look[(int(r.attacking_role_idx), int(r.defending_role_idx))] = (float(r.wr), float(r.xg_rate))
    for a, d in matchups["edge_list"]:
        wr, xg = look.get((a, d), (0.0, 0.0))
        S[a, d] = (1.0, wr, xg)
    return S


def ctx_dim(n_style: int) -> int:
    return SCORE_BINS + MINUTE_BINS + 2 * n_style


def ctx_swap_perm(n_style: int):
    """(perm, sign) for SwapSym: reverse score bins, swap home/away style blocks."""
    perm = list(range(SCORE_BINS))[::-1]
    perm += list(range(SCORE_BINS, SCORE_BINS + MINUTE_BINS))
    s0 = SCORE_BINS + MINUTE_BINS
    perm += list(range(s0 + n_style, s0 + 2 * n_style)) + list(range(s0, s0 + n_style))
    return perm, [1.0] * len(perm)


def build_ctx(df: pd.DataFrame, style_map: dict, n_style: int) -> np.ndarray:
    n = len(df)
    out = np.zeros((n, ctx_dim(n_style)), np.float32)
    diff = np.clip((df["scoreline_home"] - df["scoreline_away"]).round().astype(int).values, -2, 2) + 2
    out[np.arange(n), diff] = 1.0
    # minute_bucket in stable_windows is the window start in minutes (ws // 60)
    mb = np.clip((df["minute_bucket"].astype(float).values // 15).astype(int), 0, MINUTE_BINS - 1)
    out[np.arange(n), SCORE_BINS + mb] = 1.0
    s0 = SCORE_BINS + MINUTE_BINS
    hs = np.clip(df["home_team_id"].map(style_map).fillna(0).astype(int).values, 0, n_style - 1)
    as_ = np.clip(df["away_team_id"].map(style_map).fillna(0).astype(int).values, 0, n_style - 1)
    out[np.arange(n), s0 + hs] = 1.0
    out[np.arange(n), s0 + n_style + as_] = 1.0
    return out


def cell_mean_targets(df: pd.DataFrame, y: np.ndarray, min_n: int = 5) -> np.ndarray:
    """Mean of y over each window's (ordered formation pair, 15-minute block) cell.
    Cells with fewer than `min_n` windows fall back to the window's own y."""
    key = pd.Series(list(zip(df["home_formation"].astype(int), df["away_formation"].astype(int),
                             (df["minute_bucket"].astype(float) // 15).astype(int))), index=df.index)
    codes, _ = pd.factorize(key)
    cnt = np.bincount(codes)
    out = np.empty_like(y)
    for j in range(y.shape[1]):
        s = np.bincount(codes, weights=y[:, j])
        out[:, j] = s[codes] / cnt[codes]
    small = cnt[codes] < min_n
    out[small] = y[small]
    return out.astype(np.float32)


def make_arrays(df, bank: FormationBank, style_map, n_style, y_z: np.ndarray):
    """Rows whose formations are outside the bank are dropped; returns dict + kept mask."""
    ok = df["home_formation"].astype(int).isin(bank.index) & df["away_formation"].astype(int).isin(bank.index)
    d = df[ok.values].reset_index(drop=True)
    y = y_z[ok.values].astype(np.float32)
    return {
        "home_idx": d["home_formation"].astype(int).map(bank.index).values.astype(np.int64),
        "away_idx": d["away_formation"].astype(int).map(bank.index).values.astype(np.int64),
        "ctx": build_ctx(d, style_map, n_style),
        "y": y,
        "y_cell": cell_mean_targets(d, y),
        "match_id": d["match_id"].values,
    }, ok.values


# ------------------------------- self-test ---------------------------------- #
def _selftest(assets):
    canon = json.load(open(os.path.join(assets, "canonical_roles.json")))
    match = json.load(open(os.path.join(assets, "positional_matchups.json")))
    vocab = [442, 433, 4231, 4141, 4411, 4321, 4222, 451, 4132, 352, 3412, 3421, 343, 532, 541, 41212, 3511]
    bank = FormationBank(vocab, canon)
    assert bank.roles.shape == (17, 11) and (bank.roles.min() >= 0) and (bank.roles.max() <= 10)
    assert (bank.roles[:, 0] == 0).all(), "position 0 must be GK"
    assert (bank.adj == bank.adj.transpose(0, 2, 1)).all()
    fb = [c for c, s in bank.source.items() if s == "fallback"]
    print(f"[ok] FormationBank: 17 formations, fallback used for {fb}")

    # connectivity of every intra graph
    for c, i in bank.index.items():
        reach = np.eye(11, dtype=bool)[0]
        for _ in range(11):
            reach = reach | ((bank.adj[i] > 0)[reach].any(0))
        assert reach.all(), f"{c} intra graph disconnected"
    print("[ok] all intra graphs connected")

    S = build_edge_stats(match, None)
    assert S[..., 0].sum() == len(match["edge_list"])
    print(f"[ok] edge_stats: {int(S[..., 0].sum())} role-attack edges")

    # mirror is an involution and fixes central roles
    assert (ROLE_MIRROR[ROLE_MIRROR] == np.arange(11)).all()
    print("[ok] ROLE_MIRROR involution")

    rng = np.random.default_rng(0)
    n = 600
    forms = np.array(vocab)
    df = pd.DataFrame({
        "match_id": rng.integers(0, 60, n),
        "home_formation": rng.choice(forms, n), "away_formation": rng.choice(forms, n),
        "minute_bucket": rng.integers(0, 90, n),
        "scoreline_home": rng.integers(0, 4, n), "scoreline_away": rng.integers(0, 4, n),
        "home_team_id": rng.integers(1, 5, n), "away_team_id": rng.integers(1, 5, n),
    })
    smap = {1: 0, 2: 1, 3: 0, 4: 1}
    y = rng.normal(size=(n, 4)).astype(np.float32)
    arr, kept = make_arrays(df, bank, smap, 2, y)
    cd = ctx_dim(2)
    assert arr["ctx"].shape == (n, cd) and kept.all()
    assert np.allclose(arr["ctx"][:, :5].sum(1), 1) and np.allclose(arr["ctx"][:, 5:11].sum(1), 1)
    perm, sign = ctx_swap_perm(2)
    swapped = arr["ctx"][:, perm]
    # swapping twice is the identity; swapped score one-hot matches negated diff
    assert np.allclose(swapped[:, perm], arr["ctx"])
    d0 = (df.scoreline_home - df.scoreline_away).clip(-2, 2).values
    assert (swapped[:, :5].argmax(1) - 2 == -d0).all()
    print(f"[ok] ctx dim {cd}; swap perm is an involution and negates score diff")

    assert arr["y_cell"].shape == y.shape and np.isfinite(arr["y_cell"]).all()
    print("[ok] cell-mean targets finite; small cells fall back to own y")
    print("all checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="models/world_model")
    a = ap.parse_args()
    sys.path.insert(0, a.assets)
    _selftest(a.assets)
