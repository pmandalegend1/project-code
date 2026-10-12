import os
import warnings
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("RAMFP_DATA_DIR", "data/processed")
MASTER_EVENTS_PATH = os.path.join(DATA_DIR, "master_events.parquet")
OUTPUT_PATH = os.path.join(DATA_DIR, "stable_windows.parquet")

WINDOW_SECONDS = 600       # 10 minutes
MIN_WINDOW_SECONDS = 300   # drop windows with < 5 min of events (period end truncation)
PITCH_LENGTH = 120.0
PPDA_X_THRESHOLD = 40.0
# B9: some matches (older La Liga, some women's competitions) have no
# Starting XI event with a formation. Every window in those matches is
# silently skipped by _get_formation_at_time returning None. Count them,
# print one line per match, and summarise at the end.
matches_dropped_no_formation = set()

# Opta's published PPDA definition. Four things were wrong before:
#
# 1. The names were lowercase ("pressure", "tackle", ...) while StatsBomb
#    writes "Pressure", "Interception", "Block". .isin() matched NOTHING, so
#    n_actions was 0 for every window and _compute_ppda returned NaN every
#    single time. The world model's ppda target was entirely empty.
#
# 2. "tackle" is not a StatsBomb event type at all. A tackle is a Duel with
#    duel.type.name == "Tackle"; on 200 matches there are 0 events of type
#    "Tackle" and 8,412 tackles inside Duel. It comes in via the is_tackle
#    column that build_master_events.py precomputes.
#
# 3. Pressure was counted as a defensive action. It is not in the Opta
#    definition, and StatsBomb records a great many of them (7.6 per 5-min
#    window against 2.0 real defensive actions), which drives PPDA to ~2.7
#    against a published range of 8-15. Excluded, this corpus gives 11.28.
#
# 4. Both halves of the ratio filtered on x < 40. They have to be MIRRORED.
#    StatsBomb records every event with the acting team attacking towards
#    x = 120, so the pressing team's high actions are x > 40 while the
#    pressed team's own build-up is x < 120 - 40 = 80. Filtering the
#    pressing team at x < 40 counted actions in their OWN defensive third --
#    the opposite of pressing.
PPDA_DEFENSIVE_ACTIONS = frozenset({
    "Interception", "Foul Committed", "Block", "Clearance",
})


def _parse_location_x(loc):
    if loc is None:
        return np.nan
    if isinstance(loc, (list, tuple)):
        return float(loc[0])
    return np.nan


def _compute_ppda(events_df, pressing_team_id, pressed_team_id):
    """Opponent passes per defensive action, Opta definition.

    Returns NaN when the pressing team made no defensive action in the
    window. That is a real football situation, not a fault: it happens in
    roughly 14% of 5-minute windows under this definition. Keep the raw
    counts if you need an estimator that can use those windows.
    """
    x = pd.to_numeric(events_df["location_x"], errors="coerce")
    pressed_x_threshold = PITCH_LENGTH - PPDA_X_THRESHOLD      # 80.0

    pressed_passes = events_df[
        (events_df["team_id"] == pressed_team_id)
        & (events_df["type_name"] == "Pass")
        & x.lt(pressed_x_threshold)
        & x.notna()
    ]

    defensive = events_df["type_name"].isin(PPDA_DEFENSIVE_ACTIONS)
    if "is_tackle" in events_df.columns:
        defensive = defensive | (
            events_df["is_tackle"].fillna(False).astype(bool)
        )
    else:
        warnings.warn(
            "no is_tackle column: tackles are being dropped from PPDA. "
            "Rebuild master_events.parquet with "
            "pipeline/build_master_events.py.",
            RuntimeWarning,
        )

    pressing_actions = events_df[
        (events_df["team_id"] == pressing_team_id)
        & defensive
        & x.gt(PPDA_X_THRESHOLD)
        & x.notna()
    ]

    n_passes = len(pressed_passes)
    n_actions = len(pressing_actions)
    if n_actions == 0:
        return np.nan
    return n_passes / n_actions


def _get_formation_at_time(team_events, timestamp_sec):
    shifts = team_events[
        (team_events["type_name"].isin(["Tactical Shift", "Starting XI"]))
        & (team_events["timestamp_sec"] <= timestamp_sec)
    ].sort_values("timestamp_sec")
    if shifts.empty:
        return None
    last = shifts.iloc[-1]
    return last.get("formation_code", None)


def _scoreline_at_time(match_events, timestamp_sec):
    goals = match_events[
        (match_events["type_name"] == "Shot")
        & (match_events["shot_outcome"] == "Goal")
        & (match_events["timestamp_sec"] <= timestamp_sec)
    ]
    home_id = match_events["home_team_id"].iloc[0]
    home_goals = int((goals["team_id"] == home_id).sum())
    away_goals = int((goals["team_id"] != home_id).sum())
    return home_goals, away_goals



# Columns this module reads. The full corpus is 9.4 million events; reading
# every column at default widths needs several GB and is killed outright on a
# laptop or a small VM. Reading only what is used, with narrow dtypes and
# categoricals for the two low-cardinality strings, brings it under 500 MB.
_USED_COLUMNS = [
    "match_id", "team_id", "type_name", "timestamp_sec", "location_x",
    "period", "minute", "is_tackle", "home_team_id", "formation",
    "formation_code", "shot_statsbomb_xg", "shot_outcome",
    "competition_id", "season_id", "match_date",
]

_NARROW = {
    "match_id": "int32", "team_id": "int32", "timestamp_sec": "float32",
    "location_x": "float32", "period": "int8", "minute": "int16",
    "formation": "float32", "formation_code": "float32",
    "shot_statsbomb_xg": "float32", "home_team_id": "float32",
    "competition_id": "float32", "season_id": "float32",
}


def _load_events(path):
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in _USED_COLUMNS if c in available]
    df = pd.read_parquet(path, columns=cols)

    for col in ("type_name", "shot_outcome", "match_date"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col, dtype in _NARROW.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (TypeError, ValueError):
                pass
    return df

def extract_stable_windows(master_events_path=MASTER_EVENTS_PATH, output_path=OUTPUT_PATH,
                           start=0, count=None):
    """Extract stable windows, optionally for a slice of the matches.

    start/count exist so the full corpus can be built in pieces on a machine
    with a per-process time limit: each call handles `count` matches from
    `start` and writes its own shard, and the shards are concatenated after.
    Matches are taken in sorted order, so the slices are disjoint and stable
    across calls.
    """
    print(f"Loading {master_events_path} ...")
    events = _load_events(master_events_path)
    print(f"  Loaded {len(events):,} events across {events['match_id'].nunique():,} matches "
          f"({events.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory)")

    # normalise column names to snake_case expected names
    events.columns = [c.lower().replace(" ", "_") for c in events.columns]

    # parse location_x if stored as list/object
    if "location_x" not in events.columns and "location" in events.columns:
        events["location_x"] = events["location"].apply(_parse_location_x)

    # timestamp in seconds
    if "timestamp_sec" not in events.columns:
        if "timestamp" in events.columns:
            def _ts_to_sec(ts):
                if pd.isna(ts):
                    return np.nan
                try:
                    parts = str(ts).split(":")
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                except Exception:
                    return np.nan
            events["timestamp_sec"] = events["timestamp"].apply(_ts_to_sec)
        elif "minute" in events.columns and "second" in events.columns:
            events["timestamp_sec"] = events["minute"] * 60 + events["second"].fillna(0)
        else:
            raise ValueError("Cannot derive timestamp_sec: need 'timestamp' or ('minute','second') columns")

    if "type_name" not in events.columns and "type" in events.columns:
        events["type_name"] = events["type"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else str(x)
        )

    if "team_id" not in events.columns and "team" in events.columns:
        events["team_id"] = events["team"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else x
        )

    if "shot_statsbomb_xg" not in events.columns:
        events["shot_statsbomb_xg"] = np.nan

    if "shot_outcome" not in events.columns:
        events["shot_outcome"] = None

    if "formation_code" not in events.columns and "formation" in events.columns:
        # master_events.parquet calls this column "formation"; everything
        # downstream reads "formation_code". The old code fell through to the
        # "tactics" branch, found no such column, set formation_code to None
        # for every row, and produced zero stable windows without erroring.
        events["formation_code"] = events["formation"]

    if "formation_code" not in events.columns:
        if "tactics" in events.columns:
            events["formation_code"] = events["tactics"].apply(
                lambda x: x.get("formation") if isinstance(x, dict) else None
            )
        else:
            events["formation_code"] = None

    records = []
    match_ids = sorted(events["match_id"].unique())
    matches_dropped_no_formation = set()
    if start or count is not None:
        selected = match_ids[start: (start + count) if count else None]
        events = events[events["match_id"].isin(set(selected))]
        match_ids = selected
        print(f"Slice: matches [{start}, {start + len(selected)}) of the corpus")
    print(f"Processing {len(match_ids):,} matches ...")

    # One boolean scan of the whole table per match is O(matches x events):
    # on the full corpus that is 2,651 passes over 9.4 million rows, which
    # takes tens of minutes and looks like a hang. groupby partitions once.
    done = 0
    for match_id, mev in events.groupby("match_id", sort=False):
        mev = mev.copy()
        done += 1
        if done % 250 == 0:
            print(f"  {done:,}/{len(match_ids):,} matches -> {len(records):,} windows", flush=True)

        teams = mev["team_id"].dropna().unique()
        if len(teams) < 2:
            continue

        # Home/away came from the order teams happened to appear in the
        # array, which is not who was at home. master_events.parquet now
        # carries the real home_team_id from the match list; fall back to
        # the old behaviour only if it is missing, and say so.
        home_team_id = None
        if "home_team_id" in mev.columns:
            known = mev["home_team_id"].dropna().unique()
            if len(known) == 1:
                home_team_id = type(teams[0])(known[0])
        if home_team_id is None or home_team_id not in set(teams):
            warnings.warn(
                f"match {match_id}: no usable home_team_id, falling back to "
                f"array order; home/away labels may be swapped.",
                RuntimeWarning,
            )
            home_team_id = teams[0]
        away_team_id = [t for t in teams if t != home_team_id][0]

        def _one(col):
            if col not in mev.columns:
                return None
            vals = mev[col].dropna().unique()
            return vals[0] if len(vals) == 1 else None

        comp_id    = _one("competition_id")
        season_id  = _one("season_id")
        match_date = _one("match_date")

        mev["home_team_id"] = home_team_id

        # identify tactical shift timestamps for both teams
        shift_mask = mev["type_name"].isin(["Tactical Shift", "Starting XI"])
        shift_times = set(mev.loc[shift_mask, "timestamp_sec"].dropna().tolist())

        max_ts = mev["timestamp_sec"].max()

        # candidate window starts: every 60s, skip kickoff (<180s)
        window_starts = np.arange(180, max(180, max_ts - WINDOW_SECONDS + 1), 60)

        for ws in window_starts:
            we = ws + WINDOW_SECONDS
            win = mev[(mev["timestamp_sec"] >= ws) & (mev["timestamp_sec"] < we)]

            if len(win) < 5:
                continue

            # stability check: no tactical shift from either team within window
            if any(ws <= t < we for t in shift_times):
                continue

            # window must span at least MIN_WINDOW_SECONDS of actual events
            actual_span = win["timestamp_sec"].max() - win["timestamp_sec"].min()
            if actual_span < MIN_WINDOW_SECONDS:
                continue

            # period and minute bucket
            period = int(win["period"].mode().iloc[0]) if "period" in win.columns else 1
            minute_bucket = int(ws // 60)

            # formations at window start
            home_events = mev[mev["team_id"] == home_team_id]
            away_events = mev[mev["team_id"] == away_team_id]
            home_formation = _get_formation_at_time(home_events, ws)
            away_formation = _get_formation_at_time(away_events, ws)

            if home_formation is None or away_formation is None:
                # B9: log once per match, then skip. Appending only the first
                # time keeps this from spamming for every 10-min window.
                if match_id not in matches_dropped_no_formation:
                    matches_dropped_no_formation.add(match_id)
                    missing = []
                    if home_formation is None: missing.append("home")
                    if away_formation is None: missing.append("away")
                    print(f"  match {match_id}: no Starting XI formation for "
                          f"{'/'.join(missing)} -> all windows dropped",
                          flush=True)
                continue

            # ground truth targets
            home_shots = win[
                (win["team_id"] == home_team_id) & (win["type_name"] == "Shot")
            ]
            away_shots = win[
                (win["team_id"] == away_team_id) & (win["type_name"] == "Shot")
            ]
            xg_home = home_shots["shot_statsbomb_xg"].fillna(0).sum()
            xg_away = away_shots["shot_statsbomb_xg"].fillna(0).sum()
            xg_delta = float(xg_home - xg_away)

            home_passes = int(((win["team_id"] == home_team_id) & (win["type_name"] == "Pass")).sum())
            away_passes = int(((win["team_id"] == away_team_id) & (win["type_name"] == "Pass")).sum())
            total_passes = home_passes + away_passes
            possession_pct = float(home_passes / total_passes) if total_passes > 0 else 0.5

            # Both directions. Only one was computed before, so every
            # window contributed half the observation it could have.
            # "ppda" keeps its old meaning so nothing downstream shifts.
            ppda = _compute_ppda(win, away_team_id, home_team_id)
            ppda_home_pressing = _compute_ppda(win, home_team_id, away_team_id)

            sot_outcomes = {"Saved", "Goal"}
            shots_on_target = int(
                win[
                    (win["type_name"] == "Shot")
                    & (win["shot_outcome"].isin(sot_outcomes))
                ].shape[0]
            )

            scoreline_home, scoreline_away = _scoreline_at_time(mev, ws)

            records.append({
                "match_id":        match_id,
                "window_start_sec": float(ws),
                "window_end_sec":   float(we),
                "home_team_id":     home_team_id,
                "away_team_id":     away_team_id,
                "home_formation":   int(home_formation),
                "away_formation":   int(away_formation),
                "xg_delta":         xg_delta,
                "possession_pct":   possession_pct,
                "ppda":             ppda,                 # away pressing home
                "ppda_away_pressing": ppda,
                "ppda_home_pressing": ppda_home_pressing,
                "shots_on_target":  shots_on_target,
                "scoreline_home":   scoreline_home,
                "scoreline_away":   scoreline_away,
                "minute_bucket":    minute_bucket,
                "period":           period,
                # carried through so train_world_model.py can hold out whole
                # seasons. Without these it falls back to sorting the whole
                # corpus by window_start_sec, which puts early-match windows
                # in train and late-match windows in test -- a split by
                # minute of play, not by time.
                "competition_id":   comp_id,
                "season_id":        season_id,
                "match_date":       match_date,
            })

    if matches_dropped_no_formation:
        n_dropped = len(matches_dropped_no_formation)
        pct = 100.0 * n_dropped / max(len(match_ids), 1)
        print(f"\nDropped {n_dropped:,} of {len(match_ids):,} matches ({pct:.1f}%) "
              f"for missing Starting XI formation.")
        if pct > 5:
            print("  WARNING: more than 5% of matches had no formation on the "
                  "Starting XI event; check the master_events schema.")

    out = pd.DataFrame(records)
    if out.empty:
        raise SystemExit(
            "No stable windows were produced. The usual cause is that no event "
            "carried a formation: check that master_events.parquet has a "
            "'formation' or 'formation_code' column with non-null values for "
            "Starting XI and Tactical Shift events."
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"Saved {len(out):,} stable windows to {output_path}")
    print(f"  xg_delta:       mean={out['xg_delta'].mean():.4f}  std={out['xg_delta'].std():.4f}")
    print(f"  possession_pct: mean={out['possession_pct'].mean():.3f}")
    print(f"  ppda:           mean={out['ppda'].mean():.2f}  (NaN={out['ppda'].isna().sum()})")
    print(f"  shots_on_target mean={out['shots_on_target'].mean():.2f}")
    if "competition_id" in out.columns:
        known = out["competition_id"].notna().sum()
        print(f"  competition/season known for {known:,}/{len(out):,} windows "
              f"across {out.dropna(subset=['competition_id']).groupby(['competition_id', 'season_id']).ngroups} season(s)")
    if out["xg_delta"].abs().sum() == 0:
        print("  WARNING: every xg_delta is exactly 0 -- master_events.parquet "
              "is missing shot_statsbomb_xg; rebuild it before training.")
    return out


if __name__ == "__main__":
    def smoke_test():
        import json, tempfile
        print("--- extract_stable_windows smoke test ---")

        # build minimal synthetic master_events
        rng = np.random.default_rng(42)
        n_matches = 3
        rows = []
        for match_id in range(n_matches):
            teams = [match_id * 10, match_id * 10 + 1]
            # Starting XI at t=0 for both teams
            for team_id in teams:
                rows.append({
                    "match_id": match_id, "team_id": team_id,
                    "type_name": "Starting XI", "timestamp_sec": 0.0,
                    "formation_code": 4231, "period": 1,
                    "location_x": 60.0, "shot_statsbomb_xg": None,
                    "shot_outcome": None,
                })
            # passes and pressures over 20 minutes
            for t in range(200, 1400, 30):
                for team_id in teams:
                    rows.append({
                        "match_id": match_id, "team_id": team_id,
                        "type_name": "Pass", "timestamp_sec": float(t),
                        "formation_code": None, "period": 1 if t < 2700 else 2,
                        "location_x": float(rng.uniform(10, 110)),
                        "shot_statsbomb_xg": None, "shot_outcome": None,
                    })
            # a couple of shots
            rows.append({
                "match_id": match_id, "team_id": teams[0],
                "type_name": "Shot", "timestamp_sec": 500.0,
                "formation_code": None, "period": 1,
                "location_x": 100.0, "shot_statsbomb_xg": 0.12,
                "shot_outcome": "Saved",
            })
            rows.append({
                "match_id": match_id, "team_id": teams[1],
                "type_name": "Pressure", "timestamp_sec": 520.0,
                "formation_code": None, "period": 1,
                "location_x": 25.0, "shot_statsbomb_xg": None,
                "shot_outcome": None,
            })

        ev = pd.DataFrame(rows)
        ev["home_team_id"] = ev.groupby("match_id")["team_id"].transform("first")

        with tempfile.TemporaryDirectory() as tmp:
            events_path = os.path.join(tmp, "master_events.parquet")
            out_path    = os.path.join(tmp, "stable_windows.parquet")
            ev.to_parquet(events_path, index=False)

            result = extract_stable_windows(events_path, out_path)

            assert os.path.exists(out_path), "Output parquet not created"
            expected_cols = [
                "match_id", "window_start_sec", "window_end_sec",
                "home_team_id", "away_team_id", "home_formation", "away_formation",
                "xg_delta", "possession_pct", "ppda", "shots_on_target",
                "scoreline_home", "scoreline_away", "minute_bucket", "period",
            ]
            missing = [c for c in expected_cols if c not in result.columns]
            assert not missing, f"Missing output columns: {missing}"
            assert len(result) > 0, "No stable windows extracted from synthetic data"
            assert result["xg_delta"].dtype in [float, np.float64, np.float32]
            assert result["possession_pct"].between(0, 1).all(), "possession_pct out of [0,1]"
            print(f"Extracted {len(result)} stable windows from {n_matches} synthetic matches")
            print(f"Output columns: {list(result.columns)}")
            print("--- smoke test passed ---")

    import argparse
    _ap = argparse.ArgumentParser(description="Extract stable formation windows from master events.")
    _ap.add_argument("--master-events", default=MASTER_EVENTS_PATH)
    _ap.add_argument("--out", default=OUTPUT_PATH)
    _ap.add_argument("--smoke-test", action="store_true")
    _ap.add_argument("--start", type=int, default=0, help="first match index of this shard")
    _ap.add_argument("--count", type=int, default=None, help="number of matches in this shard")
    _a = _ap.parse_args()
    if _a.smoke_test:
        smoke_test()
    else:
        extract_stable_windows(master_events_path=_a.master_events, output_path=_a.out,
                               start=_a.start, count=_a.count)
