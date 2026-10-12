import json
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("RAMFP_DATA_DIR", "data/processed")
STABLE_WINDOWS_PATH  = os.path.join(DATA_DIR, "stable_windows.parquet")
MATCHUPS_PATH        = "models/world_model/positional_matchups.json"
CANON_ROLES_PATH     = "models/world_model/canonical_roles.json"
OUTPUT_PATH          = os.path.join(DATA_DIR, "role_matchup_stats.parquet")

# formation_code -> list of 11 canonical role indices (position-list order)
# loaded from canonical_roles.json formation_mappings[code]["node_assignments"]
class _RoleMap(dict):
    """Role assignments per formation, hand-specified first, derived otherwise.

    canonical_roles.json hand-specifies 15 formations; the corpus contains
    more.  Reading only formation_mappings meant every window whose formation
    had no hand entry was dropped here and again in graph_builder, silently.
    A miss now goes through the same documented canonical fallback the graph
    builder uses, so both stages describe the same formation the same way.
    """

    def __init__(self, canon):
        super().__init__()
        self._canon = canon
        for code_str, fm in canon["formation_mappings"].items():
            self[int(code_str)] = fm["node_assignments"]

    def __missing__(self, code):
        from models.world_model.formation_fallback import fallback_mapping
        roles = fallback_mapping(code, self._canon)["node_assignments"]
        self[code] = roles
        return roles


def _build_formation_role_map(canon_path=CANON_ROLES_PATH):
    with open(canon_path) as f:
        d = json.load(f)
    return _RoleMap(d)


def _load_matchup_edges(matchups_path=MATCHUPS_PATH):
    with open(matchups_path) as f:
        d = json.load(f)
    # list of [attacking_canonical_idx, defending_canonical_idx]
    return d["edge_list"]


def compute_role_matchup_stats(
    stable_windows_path=STABLE_WINDOWS_PATH,
    matchups_path=MATCHUPS_PATH,
    canon_path=CANON_ROLES_PATH,
    output_path=OUTPUT_PATH,
    exclude_matches=None,
):
    print(f"Loading stable windows from {stable_windows_path} ...")
    windows = pd.read_parquet(stable_windows_path)
    print(f"  {len(windows):,} windows loaded")
    if exclude_matches:
        # cross-fitting: these statistics feed the world model's edges, so a
        # clean world model needs them computed without the held-out matches too
        with open(exclude_matches) as f:
            ex = {int(x) for x in (ln.split("#")[0].strip() for ln in f) if x}
        before_ex = len(windows)
        windows = windows[~windows["match_id"].isin(ex)]
        print(f"  CROSS-FIT: excluded {len(ex)} matches -> {before_ex:,} -> {len(windows):,} windows")

    formation_role_map = _build_formation_role_map(canon_path)
    edge_list = _load_matchup_edges(matchups_path)

    known_formations = set(formation_role_map.keys())
    before = len(windows)
    windows = windows[
        windows["home_formation"].isin(known_formations)
        & windows["away_formation"].isin(known_formations)
    ]
    dropped = before - len(windows)
    if dropped:
        print(f"  Dropped {dropped} windows with out-of-vocab formations")
    print(f"  {len(windows):,} windows remaining after vocab filter")

    # accumulate xg_delta per edge across all windows
    # key: (atk_canonical_idx, def_canonical_idx)
    # value: list of xg_delta values from windows where this edge was active
    edge_xg: dict[tuple, list] = {(a, d): [] for a, d in edge_list}

    home_formations = windows["home_formation"].values
    away_formations = windows["away_formation"].values
    xg_deltas       = windows["xg_delta"].values

    for i in range(len(windows)):
        hf = int(home_formations[i])
        af = int(away_formations[i])
        xg = float(xg_deltas[i])

        # home attacks (canonical attacking roles from home formation)
        # away defends (canonical defending roles from away formation)
        home_roles = formation_role_map[hf]   # 11 canonical idx, home team
        away_roles = formation_role_map[af]   # 11 canonical idx, away team

        for atk_pos, def_pos in edge_list:
            # atk_pos: canonical idx that the home attacker occupies
            # def_pos: canonical idx that the away defender occupies
            # edge fires if both roles are present in their respective formations
            if atk_pos in home_roles and def_pos in away_roles:
                edge_xg[(atk_pos, def_pos)].append(xg)

    records = []
    for (atk_idx, def_idx), xg_values in edge_xg.items():
        n = len(xg_values)
        if n == 0:
            wr      = 0.0
            xg_rate = 0.0
        else:
            arr     = np.array(xg_values, dtype=np.float32)
            wr      = float(np.mean(arr))          # mean xg_delta (positive = attacking advantage)
            xg_rate = float(np.mean(np.clip(arr, 0, None)))  # mean xG generated (home attacker)
        records.append({
            "attacking_role_idx": atk_idx,
            "defending_role_idx": def_idx,
            "wr":                 wr,
            "xg_rate":            xg_rate,
            "n_windows":          n,
        })

    out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"\nRole matchup stats — {len(out)} edges")
    print(out.to_string(index=False))
    print(f"\nSaved to {output_path}")
    return out


if __name__ == "__main__":
    def smoke_test():
        import tempfile, shutil

        print("--- compute_role_matchup_stats smoke test ---")

        # build minimal stable_windows with all 15 vocab formations
        vocab = [4231,433,442,4141,4312,4321,352,343,532,541,4411,4132,3421,451,4222]
        rng = np.random.default_rng(42)

        n = 200
        home_f = rng.choice(vocab, size=n)
        away_f = rng.choice(vocab, size=n)
        xg     = rng.normal(0, 0.15, size=n).astype(np.float32)

        windows = pd.DataFrame({
            "match_id":        np.arange(n),
            "window_start_sec": rng.uniform(180, 3600, size=n),
            "window_end_sec":   rng.uniform(3780, 5400, size=n),
            "home_team_id":     rng.integers(0, 50, size=n),
            "away_team_id":     rng.integers(50, 100, size=n),
            "home_formation":   home_f,
            "away_formation":   away_f,
            "xg_delta":         xg,
            "possession_pct":   rng.uniform(0.3, 0.7, size=n),
            "ppda":             rng.uniform(5, 15, size=n),
            "shots_on_target":  rng.integers(0, 4, size=n),
            "scoreline_home":   rng.integers(0, 3, size=n),
            "scoreline_away":   rng.integers(0, 3, size=n),
            "minute_bucket":    rng.integers(3, 90, size=n),
            "period":           rng.choice([1, 2], size=n),
        })

        with tempfile.TemporaryDirectory() as tmp:
            sw_path  = os.path.join(tmp, "stable_windows.parquet")
            out_path = os.path.join(tmp, "role_matchup_stats.parquet")
            windows.to_parquet(sw_path, index=False)

            result = compute_role_matchup_stats(
                stable_windows_path=sw_path,
                matchups_path=MATCHUPS_PATH,
                canon_path=CANON_ROLES_PATH,
                output_path=out_path,
            )

            # load matchup edge list to know expected row count
            with open(MATCHUPS_PATH) as f:
                edge_list = json.load(f)["edge_list"]
            expected_rows = len(edge_list)

            assert len(result) == expected_rows, \
                f"Expected {expected_rows} rows (one per edge), got {len(result)}"

            expected_cols = {"attacking_role_idx","defending_role_idx","wr","xg_rate","n_windows"}
            assert expected_cols.issubset(set(result.columns)), \
                f"Missing columns: {expected_cols - set(result.columns)}"

            assert result["attacking_role_idx"].between(0, 10).all()
            assert result["defending_role_idx"].between(0, 10).all()
            assert result["xg_rate"].ge(0).all(), "xg_rate must be non-negative"
            assert result["n_windows"].gt(0).all(), \
                "All edges should fire for at least one window given full vocab coverage"

            print(f"\nAssertions passed: {len(result)} edge rows, all columns present")
            print("--- smoke test passed ---")

    import argparse
    _ap = argparse.ArgumentParser(description="Role-matchup statistics from the stable windows.")
    _ap.add_argument("--windows", default=STABLE_WINDOWS_PATH)
    _ap.add_argument("--out", default=OUTPUT_PATH)
    _ap.add_argument("--smoke-test", action="store_true")
    _ap.add_argument("--exclude-matches", default=None,
                     help="file of match_ids to leave out (cross-fitting)")
    _a = _ap.parse_args()
    if _a.smoke_test:
        smoke_test()
    else:
        compute_role_matchup_stats(stable_windows_path=_a.windows, output_path=_a.out,
                                   exclude_matches=_a.exclude_matches)
