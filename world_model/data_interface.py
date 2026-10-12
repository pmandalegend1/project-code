#!/usr/bin/env python3
"""
data_interface.py -- preflight check on the world model's inputs.

What this replaced
------------------
The previous version validated five parquet files -- natural_experiment_
triples, ppda_per_match, ppda_temporal, team_style_clusters, tactical_shifts
-- against hardcoded row counts (3282, 338, 3290) and a hardcoded "expected
15 formations".  None of those five files is produced by any script in this
repository, and no module in this repository imports this one, so the check
described a handoff that did not exist and could not have passed.  The row
counts came from an earlier, smaller corpus, so they would have failed on the
real data even if the files had been there.

What it checks now
------------------
The four artefacts train_world_model.py actually reads, with every number
derived from the data rather than asserted:

    data/processed/stable_windows.parquet
    data/processed/role_matchup_stats.parquet
    data/processed/team_style_clusters.parquet
    data/processed/formation_vocab.json

and the three failure modes that produce a model that trains cleanly and
means nothing:

    * xg_delta identically zero  (master_events built without shot xG)
    * a formation vocabulary that does not cover the windows
    * a train/test split with an empty test side

Run:  python data_interface.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

DATA_DIR = os.environ.get("RAMFP_DATA_DIR", "data/processed")

STABLE_WINDOWS = os.path.join(DATA_DIR, "stable_windows.parquet")
ROLE_MATCHUPS  = os.path.join(DATA_DIR, "role_matchup_stats.parquet")
STYLE_CLUSTERS = os.path.join(DATA_DIR, "team_style_clusters.parquet")
FORMATION_VOCAB = os.path.join(DATA_DIR, "formation_vocab.json")

WINDOW_COLUMNS = [
    "match_id", "window_start_sec", "window_end_sec",
    "home_team_id", "away_team_id", "home_formation", "away_formation",
    "xg_delta", "possession_pct", "ppda", "shots_on_target",
    "scoreline_home", "scoreline_away", "minute_bucket",
]
TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]


def load_formation_vocab(path=FORMATION_VOCAB):
    with open(path) as f:
        raw = json.load(f)
    vocab = [int(c) for c in raw["vocab"]]
    return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}


def load_stable_windows(path=STABLE_WINDOWS):
    return pd.read_parquet(path)


def load_role_matchups(path=ROLE_MATCHUPS):
    return pd.read_parquet(path)


def load_style_clusters(path=STYLE_CLUSTERS):
    return pd.read_parquet(path)


def validate(data_dir=DATA_DIR):
    errors, warnings = [], []

    def problem(msg):
        errors.append(msg)

    def warn(msg):
        warnings.append(msg)

    # ---- formation vocabulary -------------------------------------------
    vocab_path = os.path.join(data_dir, "formation_vocab.json")
    code_to_idx = {}
    try:
        code_to_idx, _ = load_formation_vocab(vocab_path)
        print(f"[ok]   formation_vocab.json   {len(code_to_idx)} formations")
    except FileNotFoundError:
        problem(f"formation_vocab.json not found under {data_dir}/ -- run prepare_inputs.py")
    except Exception as e:
        problem(f"formation_vocab.json: {e}")

    # ---- stable windows --------------------------------------------------
    windows = None
    try:
        windows = load_stable_windows(os.path.join(data_dir, "stable_windows.parquet"))
        missing = [c for c in WINDOW_COLUMNS if c not in windows.columns]
        if missing:
            problem(f"stable_windows.parquet missing columns {missing}")
        print(f"[ok]   stable_windows         {len(windows):,} windows x "
              f"{len(windows.columns)} cols, {windows['match_id'].nunique():,} matches")
    except FileNotFoundError:
        problem(f"stable_windows.parquet not found -- run extract_stable_windows.py")
    except Exception as e:
        problem(f"stable_windows.parquet: {e}")

    if windows is not None and not windows.empty:
        # targets must vary; a constant target trains to its mean and says nothing
        for t in TARGETS:
            if t not in windows.columns:
                continue
            col = windows[t].astype(float)
            if col.notna().sum() == 0:
                problem(f"target '{t}' is entirely NaN")
            elif float(col.std(skipna=True) or 0.0) == 0.0:
                problem(f"target '{t}' is constant at {col.dropna().iloc[0]} -- "
                        f"the model cannot learn anything from it")
            else:
                print(f"       {t:<16} mean={col.mean(skipna=True):+.4f} "
                      f"std={col.std(skipna=True):.4f} "
                      f"NaN={int(col.isna().sum()):,}")

        # vocabulary coverage
        if code_to_idx:
            seen = pd.concat([windows["home_formation"], windows["away_formation"]]).dropna().astype(int)
            covered = seen.isin(set(code_to_idx)).mean()
            n_oov = int((~seen.isin(set(code_to_idx))).sum())
            print(f"[ok]   vocabulary covers      {100 * covered:.2f}% of formation appearances "
                  f"({n_oov:,} out of vocabulary)")
            if covered < 0.95:
                problem(f"vocabulary covers only {100 * covered:.1f}% of formation appearances; "
                        f"more than one window in twenty would be dropped")

        # the split must leave something to test on
        if "competition_id" in windows.columns and windows["competition_id"].notna().any():
            pairs = windows.dropna(subset=["competition_id", "season_id"])
            pairs = set(map(tuple, pairs[["competition_id", "season_id"]].drop_duplicates().values))
            multi = [c for c in {c for c, _ in pairs}
                     if len({s for cc, s in pairs if cc == c}) > 1]
            if multi:
                print(f"[ok]   season holdout         {len(multi)} competition(s) have more than "
                      f"one season, so a season can be held out")
            else:
                warn("no competition has more than one season; the split falls back to "
                     "a chronological 80/20 by match")
        else:
            warn("stable_windows has no competition metadata; the split falls back to "
                 "a chronological 80/20 by match")

    # ---- role matchup statistics ----------------------------------------
    try:
        rms = load_role_matchups(os.path.join(data_dir, "role_matchup_stats.parquet"))
        need = {"attacking_role_idx", "defending_role_idx"}
        if not need.issubset(rms.columns):
            problem(f"role_matchup_stats.parquet missing {need - set(rms.columns)}")
        print(f"[ok]   role_matchup_stats     {len(rms):,} role pairs")
        if "n" in rms.columns:
            thin = int((rms["n"] < 30).sum())
            if thin:
                warn(f"{thin} of {len(rms)} role pairs are estimated from fewer than "
                     f"30 windows; their statistics are noise")
    except FileNotFoundError:
        problem("role_matchup_stats.parquet not found -- run compute_role_matchup_stats.py")
    except Exception as e:
        problem(f"role_matchup_stats.parquet: {e}")

    # ---- style clusters --------------------------------------------------
    try:
        sty = load_style_clusters(os.path.join(data_dir, "team_style_clusters.parquet"))
        if "team_id" not in sty.columns or "style_cluster" not in sty.columns:
            problem("team_style_clusters.parquet needs team_id and style_cluster")
        else:
            counts = sty["style_cluster"].value_counts().sort_index().to_dict()
            print(f"[ok]   team_style_clusters    {len(sty):,} teams, clusters {counts}")
            if sty["style_cluster"].nunique() < 2:
                problem("only one style cluster -- the style-conditioning ablation "
                        "has nothing to ablate")
            if windows is not None and not windows.empty:
                teams = set(windows["home_team_id"].dropna().astype(int)) | \
                        set(windows["away_team_id"].dropna().astype(int))
                known = teams & set(sty["team_id"].astype(int))
                frac = len(known) / max(len(teams), 1)
                print(f"       style known for        {len(known)}/{len(teams)} teams "
                      f"({100 * frac:.1f}%)")
                if frac < 0.9:
                    warn(f"style cluster unknown for {len(teams) - len(known)} teams; "
                         f"those default to cluster 0, which is not a neutral value")
    except FileNotFoundError:
        problem("team_style_clusters.parquet not found -- run prepare_inputs.py")
    except Exception as e:
        problem(f"team_style_clusters.parquet: {e}")

    # ---- verdict ---------------------------------------------------------
    print()
    for w in warnings:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[FAIL] {e}")
    if errors:
        print(f"\n{len(errors)} problem(s) -- fix these before training.")
        return 1
    print("inputs look consistent; safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
