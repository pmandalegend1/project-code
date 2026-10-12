"""
PPDA computation — CORRECTED.

Drop-in replacement for src/analysis/ppda.py. Same public API:
    compute_ppda, compute_team_match_ppda, compute_15_minute_ppda

Three bugs fixed, each verified against real StatsBomb open data
(9 La Liga matches, 2018-2020). Evidence is in README_FIXES.md.

FIX 1 — tackles were never counted
    "Tackle" is not a StatsBomb event type. Tackles live inside `Duel`
    with duel.type.name == "Tackle". The old DEFENSIVE_ACTIONS set
    contained the literal string "Tackle", which matched zero rows in all
    4,712 windows (see ppda_window_diagnostics.csv: tackle_events max = 0).

FIX 2 — numerator and denominator were at opposite ends of the pitch
    StatsBomb records every event with the ACTING team attacking towards
    x = 120. So `x < 40` means "in the acting team's own defensive third".
    Applying it to both the opponent's passes AND the presser's defensive
    actions puts them 60+ metres apart. The zone must be MIRRORED:
    presser acts at x > t, opponent passes at x < 120 - t.

FIX 3 — `Pressure` does not belong in the denominator
    StatsBomb logs ~117 Pressure events per team per match against ~34
    real defensive actions. Including them drives PPDA to ~3.6, which is
    not comparable to any published PPDA. Measured on the same 9 matches:

        Tackle+Intercept+Foul                mean 18.04   median 15.61
        + Block + Clearance  (DEFAULT)       mean 12.72   median 11.24  <-- 8-15
        Paper spec, with Pressure            mean  3.65   median  3.41

    `include_pressure=True` restores the draft's definition. If you use it,
    do NOT call the metric "PPDA" in the paper without a footnote saying it
    includes StatsBomb pressure events and is not comparable to Opta PPDA.
"""

from __future__ import annotations

import math
import warnings

import pandas as pd


# Opta-comparable defensive actions. Tackles are handled separately because
# they are not their own StatsBomb event type.
DEFENSIVE_ACTIONS = frozenset(
    {
        "Interception",
        "Foul Committed",
        "Block",
        "Clearance",
    }
)

# Kept only so the draft's definition can be reproduced for an ablation.
PRESSURE_ACTIONS = frozenset({"Pressure"})

DEFAULT_X_THRESHOLD = 40.0
PITCH_LENGTH = 120.0


def _defensive_mask(df: pd.DataFrame, include_pressure: bool) -> pd.Series:
    """
    Boolean mask of defensive actions, counting tackles correctly.

    Prefers the upstream `is_tackle` flag (see patch_preprocessing.md, which
    rebuilds it from the `extra` column). Falls back to `duel_type`, then
    warns loudly if neither is present — silently losing tackles again is
    exactly the failure this module exists to prevent.
    """
    types = DEFENSIVE_ACTIONS | (PRESSURE_ACTIONS if include_pressure else frozenset())
    mask = df["event_type"].isin(types)

    if "is_tackle" in df.columns:
        tackles = df["is_tackle"].fillna(False).astype(bool)
    elif "duel_type" in df.columns:
        tackles = (df["event_type"].eq("Duel") & df["duel_type"].eq("Tackle"))
    else:
        warnings.warn(
            "Neither 'is_tackle' nor 'duel_type' is present: tackles cannot be "
            "counted and PPDA will be biased upward. Apply "
            "patch_preprocessing.md before trusting these numbers.",
            RuntimeWarning,
            stacklevel=2,
        )
        tackles = pd.Series(False, index=df.index)

    return mask | tackles


def compute_ppda(
    events: pd.DataFrame,
    pressing_team_id: int,
    opponent_team_id: int,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    x_threshold: float = DEFAULT_X_THRESHOLD,
    include_pressure: bool = False,
) -> dict[str, float | int]:
    """
    PPDA = opponent passes in the pressing zone / pressing team's defensive
    actions in that SAME physical zone.

    `x_threshold` is expressed in the PRESSING team's frame: actions count
    when x > x_threshold (outside their own defensive third). The opponent's
    passes are counted in the mirrored band, x < 120 - x_threshold, which is
    the same strip of grass.
    """
    _validate_events(events)

    df = events
    if start_seconds is not None:
        df = df[df["event_seconds"] >= float(start_seconds)]
    if end_seconds is not None:
        df = df[df["event_seconds"] < float(end_seconds)]

    x = pd.to_numeric(df["x"], errors="coerce")
    opponent_x_threshold = PITCH_LENGTH - float(x_threshold)

    opponent_passes = df[
        df["team_id"].eq(int(opponent_team_id))
        & df["event_type"].eq("Pass")
        & x.lt(opponent_x_threshold)
        & x.notna()
    ]

    defensive_actions = df[
        df["team_id"].eq(int(pressing_team_id))
        & _defensive_mask(df, include_pressure)
        & x.gt(float(x_threshold))
        & x.notna()
    ]

    pass_count = int(len(opponent_passes))
    defensive_count = int(len(defensive_actions))
    value = math.nan if defensive_count == 0 else pass_count / defensive_count

    return {
        "pressing_team_id": int(pressing_team_id),
        "opponent_team_id": int(opponent_team_id),
        "passes": pass_count,
        "defensive_actions": defensive_count,
        "ppda": value,
    }


def compute_team_match_ppda(
    events: pd.DataFrame,
    include_pressure: bool = False,
) -> pd.DataFrame:
    _validate_events(events)

    teams = events["team_id"].dropna().astype(int).unique()
    if len(teams) != 2:
        raise ValueError("Expected exactly two teams in a match.")

    team_a, team_b = int(teams[0]), int(teams[1])
    team_names = (
        events[["team_id", "team"]]
        .drop_duplicates("team_id")
        .set_index("team_id")["team"]
        .to_dict()
    )

    rows = []
    for pressing, opponent in ((team_a, team_b), (team_b, team_a)):
        result = compute_ppda(
            events,
            pressing_team_id=pressing,
            opponent_team_id=opponent,
            include_pressure=include_pressure,
        )
        result["team"] = team_names.get(pressing)
        result["opponent"] = team_names.get(opponent)
        rows.append(result)

    return pd.DataFrame(rows)


def compute_15_minute_ppda(
    events: pd.DataFrame,
    include_pressure: bool = False,
) -> pd.DataFrame:
    _validate_events(events)

    teams = events["team_id"].dropna().astype(int).unique()
    if len(teams) != 2:
        raise ValueError("Expected exactly two teams in a match.")

    rows = []
    for team_id in teams:
        team_id = int(team_id)
        opponent_id = next(int(o) for o in teams if int(o) != team_id)
        team_name = events.loc[events["team_id"] == team_id, "team"].dropna().iloc[0]

        for start_minute in range(0, 90, 15):
            end_minute = start_minute + 15
            result = compute_ppda(
                events,
                pressing_team_id=team_id,
                opponent_team_id=opponent_id,
                start_seconds=start_minute * 60,
                end_seconds=end_minute * 60,
                include_pressure=include_pressure,
            )
            rows.append({
                "game_id": int(events["game_id"].iloc[0]),
                "team_id": team_id,
                "team": team_name,
                "window_start_minute": start_minute,
                "window_end_minute": end_minute,
                "passes": result["passes"],
                "defensive_actions": result["defensive_actions"],
                "ppda": result["ppda"],
            })

    return pd.DataFrame(rows)


def _validate_events(events: pd.DataFrame) -> None:
    required = {"game_id", "event_seconds", "team_id", "event_type", "x"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError("Missing PPDA columns: " + ", ".join(sorted(missing)))
