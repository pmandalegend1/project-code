from __future__ import annotations

import pandas as pd


FORMATION_EVENT_TYPES = frozenset(
    {
        "Starting XI",
        "Tactical Shift",
    }
)

MIN_FORMATION_CHANGE_SECONDS = 180.0


def format_formation(
    formation_code: int | float | None,
) -> str | None:

    if formation_code is None:
        return None

    try:
        digits = str(
            int(formation_code)
        )
    except (TypeError, ValueError):
        return None

    if len(digits) < 3:
        return None

    return "-".join(digits)


def build_formation_timeline(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the formation timeline for both teams.

    A Tactical Shift is not automatically a formation change.

    It becomes an actual formation change only when:
        old formation != new formation

    Halftime Tactical Shift events are retained and flagged.
    """

    required = {
        "game_id",
        "event_id",
        "event_seconds",
        "period_id",
        "index",
        "team_id",
        "team",
        "event_type",
        "formation_code",
        "formation",
        "is_halftime_shift",
    }

    missing = required - set(
        events.columns
    )

    if missing:
        raise ValueError(
            "Missing formation columns: "
            + ", ".join(sorted(missing))
        )

    source = events[
        events["event_type"].isin(
            FORMATION_EVENT_TYPES
        )
        & events["formation_code"].notna()
    ].copy()

    source = source.sort_values(
        [
            "team_id",
            "event_seconds",
            "index",
        ],
        kind="stable",
    )

    rows: list[dict] = []

    for team_id, team_events in source.groupby(
        "team_id",
        sort=False,
    ):

        current_formation = None

        for _, event in team_events.iterrows():

            new_formation = int(
                event["formation_code"]
            )

            old_formation = (
                current_formation
            )

            is_change = (
                event["event_type"]
                == "Tactical Shift"
                and old_formation is not None
                and old_formation != new_formation
            )

            eligible = (
                is_change
                and float(
                    event["event_seconds"]
                )
                >= MIN_FORMATION_CHANGE_SECONDS
            )

            rows.append(
                {
                    "game_id": int(
                        event["game_id"]
                    ),
                    "team_id": int(
                        team_id
                    ),
                    "team": event["team"],
                    "event_id": event[
                        "event_id"
                    ],
                    "event_seconds": float(
                        event["event_seconds"]
                    ),
                    "period_id": int(
                        event["period_id"]
                    ),
                    "event_type": event[
                        "event_type"
                    ],
                    "old_formation_code": (
                        old_formation
                    ),
                    "old_formation": (
                        format_formation(
                            old_formation
                        )
                        if old_formation
                        is not None
                        else None
                    ),
                    "new_formation_code": (
                        new_formation
                    ),
                    "new_formation": (
                        format_formation(
                            new_formation
                        )
                    ),
                    "is_formation_change": (
                        is_change
                    ),
                    "is_halftime_shift": bool(
                        event[
                            "is_halftime_shift"
                        ]
                    ),
                    "is_eligible_natural_experiment": (
                        eligible
                    ),
                }
            )

            current_formation = (
                new_formation
            )

    return pd.DataFrame(rows)


def get_formation_at_time(
    timeline: pd.DataFrame,
    team_id: int,
    event_seconds: float,
) -> int | None:

    if timeline.empty:
        return None

    team_timeline = timeline[
        timeline["team_id"]
        == team_id
    ]

    team_timeline = team_timeline[
        team_timeline["event_seconds"]
        <= event_seconds
    ]

    if team_timeline.empty:
        return None

    team_timeline = team_timeline.sort_values(
        [
            "event_seconds",
        ],
        kind="stable",
    )

    row = team_timeline.iloc[-1]

    value = row[
        "new_formation_code"
    ]

    if pd.isna(value):
        return None

    return int(value)