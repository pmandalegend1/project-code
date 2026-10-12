from __future__ import annotations

import math

import pandas as pd

from analysis.formation import (
    get_formation_at_time,
    format_formation,
)
from analysis.ppda import compute_ppda


# Two different physical quantities, previously collapsed into one constant.
#
# DID_WINDOW_SECONDS  -- how long pressing intensity takes to respond to a
#   formation change.  Pressing adjusts within a couple of possessions, so a
#   5-minute pre/post window is appropriate and is what the DiD uses.
#
# RESPONSE_WINDOW_SECONDS -- how long an opposing coach takes to observe a
#   formation change, decide, and execute a shift or substitution.  Five
#   minutes is far too short for that; 15 minutes is declared primary.
DID_WINDOW_SECONDS = 300.0
RESPONSE_WINDOW_SECONDS = 900.0

# Bilateral isolation window.  A formation change is only a clean treatment
# if no OTHER formation change happens close enough to confound the before/
# after comparison.  300 s on either side matches the DiD window itself.
ISOLATION_WINDOW_SECONDS = 300.0

# Deprecated alias. Nothing should import this; it exists so that any
# forgotten external caller fails loudly on review rather than silently
# using the wrong window.
WINDOW_SECONDS = DID_WINDOW_SECONDS


OUTPUT_COLUMNS = [
    "game_id",
    "team_id",
    "team",
    "opponent_team_id",
    "opponent_team",
    "change_event_id",
    "change_time",
    "period_id",
    "is_halftime_shift",
    "old_formation_code",
    "new_formation_code",
    "old_formation",
    "new_formation",
    "before_start",
    "before_end",
    "after_start",
    "after_end",
    "before_ppda",
    "after_ppda",
    "before_passes",
    "after_passes",
    "before_defensive_actions",
    "after_defensive_actions",
    "score_for",
    "score_against",
    "score_difference",
    "opponent_response_formation_code",
    "opponent_response_formation",
    "opponent_response_is_shift",
    "opponent_response_is_halftime_shift",
    # --- censoring provenance (see README_FIXES.md) ----------------------
    "match_end_seconds",
    "did_window_available_seconds",
    "response_window_available_seconds",
    "response_window_complete",
    # --- confound provenance ---------------------------------------------
    # These flag, per treatment, whether a plausible confound sits inside
    # the before/after window.  They do NOT drop rows.  `did.py` decides
    # which subset to estimate on, so nothing is silently excluded and
    # every exclusion is one line in a robustness table.
    #
    #   is_first_change_for_team_match
    #       False if this team had already changed formation earlier in the
    #       match, in which case the "before" window may not be a clean
    #       baseline for the team's original shape.
    #
    #   other_formation_change_within_isolation_window
    #       True if ANY other formation change by EITHER team occurred
    #       within +/-ISOLATION_WINDOW_SECONDS of this change.  This is the
    #       bilateral-isolation filter the audit asked for.
    #
    #   seconds_to_nearest_other_change
    #       Signed absolute distance to the nearest other formation change
    #       in the same match, NaN if none.  Feeds sensitivity analyses.
    #
    #   substitutions_within_did_window
    #   goals_within_did_window
    #       Counts inside [t-DID, t+DID).  Substitutions and goals are the
    #       two events most likely to move pressing intensity independently
    #       of any formation change, so they are recorded rather than
    #       assumed away.
    "is_first_change_for_team_match",
    "other_formation_change_within_isolation_window",
    "seconds_to_nearest_other_change",
    "substitutions_within_did_window",
    "goals_within_did_window",
]


def extract_natural_experiments(
    events: pd.DataFrame,
    formation_timeline: pd.DataFrame,
) -> pd.DataFrame:

    _validate_inputs(
        events,
        formation_timeline,
    )

    changes = formation_timeline[
        formation_timeline[
            "is_eligible_natural_experiment"
        ]
    ].copy()

    if changes.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    # Pre-filter the timeline once: only real formation changes matter for
    # the isolation and first-change checks.  Everything below compares
    # against this frame, not against every Starting XI / Tactical Shift.
    if "is_formation_change" in formation_timeline.columns:
        changes_only = formation_timeline[
            formation_timeline[
                "is_formation_change"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()
    else:
        changes_only = formation_timeline.iloc[0:0].copy()

    rows = []

    for _, change in changes.iterrows():

        game_id = int(change["game_id"])
        team_id = int(change["team_id"])
        change_time = float(change["event_seconds"])

        match_events = events[
            events["game_id"] == game_id
        ].copy()

        match_end_seconds = float(
            match_events["event_seconds"].max()
        )

        seconds_remaining = max(
            0.0,
            match_end_seconds - change_time,
        )

        did_window_available = min(
            DID_WINDOW_SECONDS, seconds_remaining,
        )
        response_window_available = min(
            RESPONSE_WINDOW_SECONDS, seconds_remaining,
        )

        teams = (
            match_events["team_id"]
            .dropna()
            .astype(int)
            .unique()
        )
        opponents = [int(t) for t in teams if int(t) != team_id]
        if len(opponents) != 1:
            continue
        opponent_id = opponents[0]

        opponent_names = (
            match_events.loc[
                match_events["team_id"] == opponent_id, "team"
            ]
            .dropna()
            .unique()
        )
        if len(opponent_names) != 1:
            continue
        opponent_name = str(opponent_names[0])
        team_name = str(change["team"])

        before_start = change_time - DID_WINDOW_SECONDS
        before_end = change_time
        after_start = change_time
        after_end = change_time + DID_WINDOW_SECONDS

        before = compute_ppda(
            match_events,
            pressing_team_id=opponent_id,
            opponent_team_id=team_id,
            start_seconds=before_start,
            end_seconds=before_end,
        )
        after = compute_ppda(
            match_events,
            pressing_team_id=opponent_id,
            opponent_team_id=team_id,
            start_seconds=after_start,
            end_seconds=after_end,
        )

        score_for, score_against = _score_at_time(
            match_events, team_id, change_time,
        )

        response = _opponent_response(
            formation_timeline, opponent_id, change_time,
        )

        confounds = _compute_confound_flags(
            match_events=match_events,
            changes_only=changes_only,
            game_id=game_id,
            change_time=change_time,
            change_event_id=change["event_id"],
            team_id=team_id,
        )

        rows.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "team": team_name,
                "opponent_team_id": opponent_id,
                "opponent_team": opponent_name,
                "change_event_id": change["event_id"],
                "change_time": change_time,
                "period_id": int(change["period_id"]),
                "is_halftime_shift": bool(
                    change["is_halftime_shift"]
                ),
                "old_formation_code": change["old_formation_code"],
                "new_formation_code": change["new_formation_code"],
                "old_formation": change["old_formation"],
                "new_formation": change["new_formation"],
                "before_start": before_start,
                "before_end": before_end,
                "after_start": after_start,
                "after_end": after_end,
                "before_ppda": before["ppda"],
                "after_ppda": after["ppda"],
                "before_passes": before["passes"],
                "after_passes": after["passes"],
                "before_defensive_actions": before[
                    "defensive_actions"
                ],
                "after_defensive_actions": after[
                    "defensive_actions"
                ],
                "score_for": score_for,
                "score_against": score_against,
                "score_difference": score_for - score_against,
                "opponent_response_formation_code": response[
                    "formation_code"
                ],
                "opponent_response_formation": response[
                    "formation"
                ],
                "opponent_response_is_shift": response["is_shift"],
                "opponent_response_is_halftime_shift": response[
                    "is_halftime_shift"
                ],
                "match_end_seconds": match_end_seconds,
                "did_window_available_seconds": (
                    did_window_available
                ),
                "response_window_available_seconds": (
                    response_window_available
                ),
                "response_window_complete": (
                    response_window_available
                    >= RESPONSE_WINDOW_SECONDS
                ),
                **confounds,
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Confound flags
# ---------------------------------------------------------------------------

def _compute_confound_flags(
    *,
    match_events: pd.DataFrame,
    changes_only: pd.DataFrame,
    game_id: int,
    change_time: float,
    change_event_id,
    team_id: int,
) -> dict:

    # --- is this the first formation change by this team in this match? ---
    if changes_only.empty:
        is_first = True
    else:
        earlier_same_team = changes_only[
            (changes_only["game_id"] == game_id)
            & (changes_only["team_id"] == team_id)
            & (changes_only["event_seconds"] <= change_time)
            & (changes_only["event_id"] != change_event_id)
        ]
        is_first = earlier_same_team.empty

    # --- is there any OTHER formation change within the isolation window? --
    iso_start = change_time - ISOLATION_WINDOW_SECONDS
    iso_end = change_time + ISOLATION_WINDOW_SECONDS

    if changes_only.empty:
        has_other_change = False
        seconds_to_nearest = float("nan")
    else:
        other_changes = changes_only[
            (changes_only["game_id"] == game_id)
            & (changes_only["event_id"] != change_event_id)
            & (changes_only["event_seconds"] > iso_start)
            & (changes_only["event_seconds"] < iso_end)
        ]
        has_other_change = not other_changes.empty
        if has_other_change:
            diffs = (
                other_changes["event_seconds"] - change_time
            ).abs()
            seconds_to_nearest = float(diffs.min())
        else:
            seconds_to_nearest = float("nan")

    # --- substitutions and goals in the +/- DID window --------------------
    win_start = change_time - DID_WINDOW_SECONDS
    win_end = change_time + DID_WINDOW_SECONDS

    window = match_events[
        (match_events["event_seconds"] >= win_start)
        & (match_events["event_seconds"] < win_end)
    ]

    sub_count = 0
    goal_count = 0
    if not window.empty:
        if "event_type" in window.columns:
            sub_count = int(
                (window["event_type"] == "Substitution").sum()
            )
        if "is_goal" in window.columns:
            goal_count = int(
                window["is_goal"].fillna(False).astype(bool).sum()
            )

    return {
        "is_first_change_for_team_match": bool(is_first),
        "other_formation_change_within_isolation_window": bool(
            has_other_change
        ),
        "seconds_to_nearest_other_change": seconds_to_nearest,
        "substitutions_within_did_window": sub_count,
        "goals_within_did_window": goal_count,
    }


# ---------------------------------------------------------------------------
# Response and score helpers (unchanged in behaviour)
# ---------------------------------------------------------------------------

def _opponent_response(
    timeline: pd.DataFrame,
    opponent_id: int,
    change_time: float,
) -> dict:

    upper = change_time + RESPONSE_WINDOW_SECONDS

    shifts = timeline[
        (timeline["team_id"] == opponent_id)
        & (timeline["event_type"] == "Tactical Shift")
        & (timeline["event_seconds"] >= change_time)
        & (timeline["event_seconds"] < upper)
    ].sort_values(["event_seconds"], kind="stable")

    if not shifts.empty:
        row = shifts.iloc[0]
        return {
            "formation_code": int(row["new_formation_code"]),
            "formation": row["new_formation"],
            "is_shift": True,
            "is_halftime_shift": bool(row["is_halftime_shift"]),
        }

    formation_code = get_formation_at_time(
        timeline, opponent_id, upper,
    )
    if formation_code is None:
        return {
            "formation_code": None,
            "formation": None,
            "is_shift": False,
            "is_halftime_shift": False,
        }

    return {
        "formation_code": formation_code,
        "formation": format_formation(formation_code),
        "is_shift": False,
        "is_halftime_shift": False,
    }


def _score_at_time(
    events: pd.DataFrame,
    team_id: int,
    time_seconds: float,
) -> tuple[int, int]:

    relevant = events[events["event_seconds"] < time_seconds]

    goals = relevant[
        relevant["is_goal"].fillna(False).astype(bool)
    ]

    score_for = int((goals["team_id"] == team_id).sum())
    score_against = int((goals["team_id"] != team_id).sum())
    return score_for, score_against


def _validate_inputs(
    events: pd.DataFrame,
    timeline: pd.DataFrame,
) -> None:

    event_columns = {
        "game_id", "event_seconds", "team_id", "team",
        "event_type", "is_goal",
    }
    timeline_columns = {
        "game_id", "team_id", "event_seconds", "period_id",
        "event_type", "is_formation_change", "is_halftime_shift",
        "is_eligible_natural_experiment",
    }

    missing_events = event_columns - set(events.columns)
    missing_timeline = timeline_columns - set(timeline.columns)

    if missing_events:
        raise ValueError(
            "Missing event columns: "
            + ", ".join(sorted(missing_events))
        )
    if missing_timeline:
        raise ValueError(
            "Missing formation timeline columns: "
            + ", ".join(sorted(missing_timeline))
        )