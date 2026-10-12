from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from data.preprocessing import EventPreprocessor
from data.statsbomb_loader import StatsBombDataLoader
from analysis.formation import build_formation_timeline
from analysis.ppda import compute_ppda


MAIN_MATCH_SECONDS = 90.0 * 60.0
EXTRA_TIME_MATCH_SECONDS = 120.0 * 60.0
STYLE_FILE = "team_style_clusters.csv"


@dataclass(frozen=True)
class DiDConfig:
    """Configuration for one SportsM5 DiD specification."""

    window_minutes: int = 5
    control_search_minutes: int = 15
    control_grid_seconds: int = 60
    score_tolerance: int = 1
    exclude_halftime: bool = True
    only_halftime: bool = False
    allowed_periods: tuple[int, ...] = (1, 2)
    max_match_seconds: float = MAIN_MATCH_SECONDS

    first_change_per_team_match: bool = True
    exclude_opponent_responses: bool = False
    winsorize_95: bool = False
    minimum_change_gap_minutes: int | None = None

    # --- confound filters -------------------------------------------------
    # Each of these consumes a column emitted by extract_natural_experiments
    # (see src/analysis/natural_experiments.py).  They are OFF by default so
    # the primary specification is untouched; each one is turned on in one
    # robustness row, so every exclusion is on the record.
    #
    #   exclude_bilateral_confound
    #       drop treatments where another formation change by EITHER team
    #       sits within +/-ISOLATION_WINDOW_SECONDS.  Without this, the
    #       "after" window can straddle a second change, and the DiD
    #       estimate is a sum of two effects.
    #
    #   exclude_any_substitution_nearby
    #   exclude_any_goal_nearby
    #       drop treatments where a substitution or a goal occurs inside
    #       [t - DID, t + DID).  Substitutions and goals both move pressing
    #       intensity independently of any formation change, so they are the
    #       two most plausible remaining confounds inside the window.
    exclude_bilateral_confound: bool = False
    exclude_any_substitution_nearby: bool = False
    exclude_any_goal_nearby: bool = False

    @property
    def window_seconds(self) -> float:
        return self.window_minutes * 60.0

    @property
    def search_seconds(self) -> float:
        return self.control_search_minutes * 60.0


def _require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(missing)
        )


def _numeric(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )
    return df


def _score_at_time(
    events: pd.DataFrame,
    team_id: int,
    time_seconds: float,
) -> tuple[int, int, int]:
    """Return score-for, score-against and signed score difference at t."""
    relevant = events[
        events["event_seconds"] < float(time_seconds)
    ]

    goals = relevant[
        relevant["is_goal"].fillna(False).astype(bool)
    ]

    score_for = int(
        (goals["team_id"] == int(team_id)).sum()
    )
    score_against = int(
        (goals["team_id"] != int(team_id)).sum()
    )

    return (
        score_for,
        score_against,
        score_for - score_against,
    )


def _window_periods(
    events: pd.DataFrame,
    start: float,
    end: float,
) -> set[int]:
    window = events[
        (events["event_seconds"] >= float(start))
        & (events["event_seconds"] < float(end))
    ]

    return set(
        pd.to_numeric(
            window["period_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )


def _window_is_valid(
    events: pd.DataFrame,
    start: float,
    end: float,
    allowed_periods: tuple[int, ...],
    max_match_seconds: float = MAIN_MATCH_SECONDS,
) -> bool:
    if start < 0.0 or end > max_match_seconds:
        return False

    periods = _window_periods(
        events,
        start,
        end,
    )

    return (
        len(periods) == 1
        and next(iter(periods)) in allowed_periods
    )


def _formation_changes_in_window(
    timeline: pd.DataFrame,
    start: float,
    end: float,
) -> pd.DataFrame:
    mask = (
        timeline["is_formation_change"]
        .fillna(False)
        .astype(bool)
        & (timeline["event_seconds"] >= float(start))
        & (timeline["event_seconds"] < float(end))
    )
    return timeline.loc[mask].copy()


def _opponent_id(
    events: pd.DataFrame,
    team_id: int,
) -> int:
    teams = (
        events["team_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    opponents = [
        value for value in teams
        if value != int(team_id)
    ]

    if len(opponents) != 1:
        raise ValueError(
            f"Expected exactly one opponent for team {team_id}; "
            f"found {opponents}."
        )

    return int(opponents[0])


def _team_name(
    events: pd.DataFrame,
    team_id: int,
) -> str:
    values = (
        events.loc[
            events["team_id"] == int(team_id),
            "team",
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    return str(values[0]) if len(values) else str(team_id)


def _ppda_pair(
    events: pd.DataFrame,
    focal_team_id: int,
    start: float,
    window_seconds: float,
) -> dict:
    """Opponent PPDA against focal_team_id before and after start."""
    opponent_id = _opponent_id(
        events,
        focal_team_id,
    )

    before = compute_ppda(
        events,
        pressing_team_id=opponent_id,
        opponent_team_id=focal_team_id,
        start_seconds=start - window_seconds,
        end_seconds=start,
    )

    after = compute_ppda(
        events,
        pressing_team_id=opponent_id,
        opponent_team_id=focal_team_id,
        start_seconds=start,
        end_seconds=start + window_seconds,
    )

    return {
        "before_ppda": before["ppda"],
        "after_ppda": after["ppda"],
        "before_passes": before["passes"],
        "after_passes": after["passes"],
        "before_defensive_actions": before["defensive_actions"],
        "after_defensive_actions": after["defensive_actions"],
    }


def _prepare_game(
    loader: StatsBombDataLoader,
    game: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_id = int(game["game_id"])

    if pd.isna(game.get("home_team_id")):
        raise ValueError(
            f"Game {game_id} has no home_team_id."
        )

    events = loader.get_events(game_id)

    if events.empty:
        raise ValueError(
            f"Game {game_id} contains no events."
        )

    canonical = EventPreprocessor(
        events,
        home_team_id=int(game["home_team_id"]),
    ).transform()

    timeline = build_formation_timeline(
        canonical
    )

    return canonical, timeline


def _load_style_clusters(
    processed_dir: Path,
) -> pd.DataFrame:
    path = processed_dir / STYLE_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Required style-cluster file not found:\n{path}\n"
            "Run run_style_clustering.py first."
        )

    style = pd.read_csv(path)

    _require_columns(
        style,
        {"team_id", "style_cluster"},
        STYLE_FILE,
    )

    style = _numeric(
        style[["team_id", "style_cluster"]].copy(),
        ["team_id", "style_cluster"],
    ).dropna(
        subset=["team_id", "style_cluster"]
    )

    style["team_id"] = style["team_id"].astype(int)
    style["style_cluster"] = style["style_cluster"].astype(int)

    if style["team_id"].duplicated().any():
        duplicates = (
            style.loc[
                style["team_id"].duplicated(keep=False),
                "team_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "team_style_clusters.csv must contain one row per team_id. "
            f"Duplicates: {duplicates[:20]}"
        )

    return style


def _prepare_treatments(
    experiments: pd.DataFrame,
    inventory: pd.DataFrame,
    config: DiDConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "game_id",
        "team_id",
        "team",
        "opponent_team_id",
        "opponent_team",
        "change_event_id",
        "change_time",
        "period_id",
        "is_halftime_shift",
        "before_ppda",
        "after_ppda",
        "score_for",
        "score_against",
        "score_difference",
        "old_formation",
        "new_formation",
        "opponent_response_is_shift",
    }

    _require_columns(
        experiments,
        required,
        "natural_experiments.csv",
    )

    df = _numeric(
        experiments.copy(),
        [
            "game_id",
            "team_id",
            "opponent_team_id",
            "change_time",
            "period_id",
            "before_ppda",
            "after_ppda",
            "score_for",
            "score_against",
            "score_difference",
        ],
    )

    df["is_halftime_shift"] = (
        df["is_halftime_shift"]
        .fillna(False)
        .astype(bool)
    )

    df["opponent_response_is_shift"] = (
        df["opponent_response_is_shift"]
        .fillna(False)
        .astype(bool)
    )

    counts = {
        "input_natural_experiments": len(df)
    }

    if config.only_halftime:
        df = df[df["is_halftime_shift"]].copy()
    elif config.exclude_halftime:
        df = df[~df["is_halftime_shift"]].copy()

    counts["after_halftime_filter"] = len(df)

    df = df[
        df["period_id"].isin(config.allowed_periods)
    ].copy()

    counts["after_period_filter"] = len(df)

    df = df.dropna(
        subset=[
            "game_id",
            "team_id",
            "opponent_team_id",
            "change_event_id",
            "change_time",
            "period_id",
            "score_difference",
        ]
    ).copy()

    df = df.sort_values(
        [
            "game_id",
            "team_id",
            "change_time",
            "change_event_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    if config.first_change_per_team_match:
        df = (
            df.drop_duplicates(
                subset=["game_id", "team_id"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    counts["after_first_change_per_team_match"] = len(df)

    if config.minimum_change_gap_minutes is not None:
        gap_seconds = float(config.minimum_change_gap_minutes * 60)
        previous_change = df.groupby(
            ["game_id", "team_id"],
            sort=False,
        )["change_time"].shift(1)
        gap_ok = previous_change.isna() | (
            df["change_time"] - previous_change >= gap_seconds
        )
        df = df.loc[gap_ok].copy().reset_index(drop=True)

    counts["after_minimum_change_gap"] = len(df)

    if config.exclude_opponent_responses:
        df = df[
            ~df["opponent_response_is_shift"]
        ].copy()

    counts["after_opponent_response_filter"] = len(df)

    # --- confound filters (see DiDConfig docstring) -----------------------
    # Each one is skipped quietly if its source column is absent, which
    # happens when natural_experiments.csv was built before
    # extract_natural_experiments gained these columns.  The DiD remains
    # estimable; the robustness row simply falls back to "column absent".
    if config.exclude_bilateral_confound:
        if "other_formation_change_within_isolation_window" in df.columns:
            df = df[
                ~df["other_formation_change_within_isolation_window"]
                .fillna(False)
                .astype(bool)
            ].copy()
        counts["after_bilateral_isolation_filter"] = len(df)

    if config.exclude_any_substitution_nearby:
        if "substitutions_within_did_window" in df.columns:
            df = df[
                pd.to_numeric(
                    df["substitutions_within_did_window"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                == 0
            ].copy()
        counts["after_substitution_filter"] = len(df)

    if config.exclude_any_goal_nearby:
        if "goals_within_did_window" in df.columns:
            df = df[
                pd.to_numeric(
                    df["goals_within_did_window"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                == 0
            ].copy()
        counts["after_goal_filter"] = len(df)

    # `build_dataset.py` already attaches competition_id and season_id
    # to natural_experiments.csv.  Do NOT blindly merge those columns
    # again: pandas would create competition_id_x/competition_id_y and
    # season_id_x/season_id_y, leaving no plain competition_id/season_id.
    # This was the direct cause of the KeyError seen in the DiD run.
    if {"competition_id", "season_id"}.issubset(df.columns):
        df = _numeric(
            df,
            ["competition_id", "season_id"],
        )
    else:
        inventory_meta = (
            inventory[
                ["game_id", "competition_id", "season_id"]
            ]
            .copy()
        )

        inventory_meta = _numeric(
            inventory_meta,
            [
                "game_id",
                "competition_id",
                "season_id",
            ],
        ).dropna(
            subset=[
                "game_id",
                "competition_id",
                "season_id",
            ]
        ).drop_duplicates(
            subset=["game_id"]
        )

        df = df.merge(
            inventory_meta,
            on="game_id",
            how="left",
            validate="many_to_one",
        )

    if df[
        ["competition_id", "season_id"]
    ].isna().any().any():
        raise ValueError(
            "Some treatment games have no competition/season metadata "
            "in natural_experiments.csv or dataset_inventory.csv."
        )

    df["minute"] = (
        df["change_time"] / 60.0
    )

    df["minute_bucket"] = np.floor(
        df["change_time"] / 300.0
    ).astype(int)

    if df.empty:
        raise RuntimeError(
            "No treatment observations remain after primary filters."
        )

    return (
        df.reset_index(drop=True),
        counts,
    )


def _candidate_starts(
    treatment_time: float,
    config: DiDConfig,
) -> list[float]:
    """
    Generate deterministic candidate starts on a one-minute grid.

    The team specified a +/-15 minute search radius but did not
    specify sub-minute control-start resolution. One-minute
    resolution is therefore an explicit reproducibility choice.
    """
    lower = max(
        config.window_seconds,
        treatment_time - config.search_seconds,
    )

    upper = min(
        config.max_match_seconds - config.window_seconds,
        treatment_time + config.search_seconds,
    )

    if upper < lower:
        return []

    step = float(config.control_grid_seconds)

    first = np.ceil(
        lower / step
    ) * step

    last = np.floor(
        upper / step
    ) * step

    if last < first:
        return []

    starts = np.arange(
        first,
        last + step / 2.0,
        step,
    )

    values = [
        float(round(value, 6))
        for value in starts
    ]

    values.sort(
        key=lambda value: (
            abs(value - treatment_time),
            value,
        )
    )

    return values


def _find_control(
    canonical: pd.DataFrame,
    timeline: pd.DataFrame,
    treatment: dict,
    config: DiDConfig,
) -> dict | None:
    """
    Find one untreated same-match control window for the same focal team.

    The control must:
      * be in the same match;
      * be in the same period;
      * start within +/-15 minutes of treatment;
      * have no formation change by either team during the window
        or within 5 minutes on either side;
      * have score difference within +/-1 at its start;
      * have complete window geometry.

    The control uses the same focal team as the treatment. This keeps
    the outcome definition identical: opponent PPDA against that team.
    """
    treatment_time = float(
        treatment["change_time"]
    )

    treatment_period = int(
        treatment["period_id"]
    )

    treatment_score_difference = int(
        treatment["score_difference"]
    )

    focal_team_id = int(
        treatment["team_id"]
    )

    for start in _candidate_starts(
        treatment_time,
        config,
    ):
        end = (
            start
            + config.window_seconds
        )

        full_start = (
            start
            - config.window_seconds
        )

        if full_start < 0.0 or end > config.max_match_seconds:
            continue

        if not _window_is_valid(
            canonical,
            full_start,
            end,
            config.allowed_periods,
            config.max_match_seconds,
        ):
            continue

        periods = _window_periods(
            canonical,
            full_start,
            end,
        )

        if periods != {treatment_period}:
            continue

        # No formation change during the control window or within
        # one outcome-window (5 minutes) on either side.  The control
        # window is [start-window, start+window], so its full +/-window
        # exclusion buffer is [start-2*window, start+2*window].
        buffer_start = start - (2.0 * config.window_seconds)
        buffer_end = start + (2.0 * config.window_seconds)

        if not _formation_changes_in_window(
            timeline,
            buffer_start,
            buffer_end,
        ).empty:
            continue

        (
            score_for,
            score_against,
            score_difference,
        ) = _score_at_time(
            canonical,
            focal_team_id,
            start,
        )

        if (
            abs(
                score_difference
                - treatment_score_difference
            )
            > config.score_tolerance
        ):
            continue

        pair = _ppda_pair(
            canonical,
            focal_team_id,
            start,
            config.window_seconds,
        )

        opponent_id = _opponent_id(
            canonical,
            focal_team_id,
        )

        return {
            "control_game_id": int(
                treatment["game_id"]
            ),
            "control_team_id": focal_team_id,
            "control_team": _team_name(
                canonical,
                focal_team_id,
            ),
            "control_opponent_team_id": opponent_id,
            "control_opponent_team": _team_name(
                canonical,
                opponent_id,
            ),
            "control_start": float(start),
            "control_period_id": treatment_period,
            "control_score_for": score_for,
            "control_score_against": score_against,
            "control_score_difference": score_difference,
            "control_before_ppda": pair[
                "before_ppda"
            ],
            "control_after_ppda": pair[
                "after_ppda"
            ],
            "control_before_passes": pair[
                "before_passes"
            ],
            "control_after_passes": pair[
                "after_passes"
            ],
            "control_before_defensive_actions": pair[
                "before_defensive_actions"
            ],
            "control_after_defensive_actions": pair[
                "after_defensive_actions"
            ],
            "control_distance_seconds": abs(
                start - treatment_time
            ),
        }

    return None


def _style_cluster(
    styles: pd.DataFrame,
    team_id: int,
) -> int | None:
    values = styles.loc[
        styles["team_id"] == int(team_id),
        "style_cluster",
    ]

    if values.empty:
        return None

    return int(values.iloc[0])


def _build_pairs(
    treatments: pd.DataFrame,
    game_cache: dict[
        int,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
    styles: pd.DataFrame,
    config: DiDConfig,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    unmatched: list[str] = []

    for treatment in treatments.to_dict(
        orient="records"
    ):
        game_id = int(
            treatment["game_id"]
        )

        event_id = str(
            treatment["change_event_id"]
        )

        if game_id not in game_cache:
            unmatched.append(event_id)
            continue

        canonical, timeline = game_cache[
            game_id
        ]

        treatment_time = float(
            treatment["change_time"]
        )
        treatment_period = int(
            treatment["period_id"]
        )

        # Recompute the treatment outcome from the event data using
        # the requested window length. This is essential for the
        # +/-3 and +/-10 minute robustness checks.
        treatment_ppda = _ppda_pair(
            canonical,
            int(treatment["team_id"]),
            treatment_time,
            config.window_seconds,
        )

        # Primary treatment windows must be fully contained in one
        # allowed period. A halftime-only robustness check is the
        # explicit exception because its treatment window straddles
        # the two halves by definition.
        if config.only_halftime:
            if not bool(treatment["is_halftime_shift"]):
                continue
            if treatment_period != 2:
                continue
            if (
                treatment_time - config.window_seconds < 0.0
                or treatment_time + config.window_seconds
                > config.max_match_seconds
            ):
                continue
        else:
            if not _window_is_valid(
                canonical,
                treatment_time - config.window_seconds,
                treatment_time + config.window_seconds,
                config.allowed_periods,
                config.max_match_seconds,
            ):
                continue

        control = _find_control(
            canonical,
            timeline,
            treatment,
            config,
        )

        if control is None:
            unmatched.append(event_id)
            continue

        treated_cluster = _style_cluster(
            styles,
            int(
                treatment[
                    "opponent_team_id"
                ]
            ),
        )

        control_cluster = _style_cluster(
            styles,
            int(
                control[
                    "control_opponent_team_id"
                ]
            ),
        )

        if (
            treated_cluster is None
            or control_cluster is None
        ):
            unmatched.append(event_id)
            continue

        rows.append(
            {
                "pair_id": (
                    f"{game_id}_"
                    f"{event_id}_"
                    f"{int(control['control_start'])}"
                ),
                "match_id": game_id,
                "treated_game_id": game_id,
                "treated_change_event_id": event_id,
                "treated_team_id": int(
                    treatment["team_id"]
                ),
                "treated_team": treatment["team"],
                "treated_opponent_team_id": int(
                    treatment[
                        "opponent_team_id"
                    ]
                ),
                "treated_opponent_team": treatment[
                    "opponent_team"
                ],
                "treated_opponent_style_cluster": (
                    treated_cluster
                ),
                "treated_change_time": float(
                    treatment["change_time"]
                ),
                "treated_minute": float(
                    treatment["minute"]
                ),
                "treated_period_id": int(
                    treatment["period_id"]
                ),
                "treated_minute_bucket": int(
                    treatment["minute_bucket"]
                ),
                "competition_id": int(
                    treatment["competition_id"]
                ),
                "season_id": int(
                    treatment["season_id"]
                ),
                "score_difference": int(
                    treatment["score_difference"]
                ),
                "old_formation": treatment[
                    "old_formation"
                ],
                "new_formation": treatment[
                    "new_formation"
                ],
                "is_halftime_shift": bool(
                    treatment["is_halftime_shift"]
                ),

                "opponent_response_is_shift": bool(
                    treatment[
                        "opponent_response_is_shift"
                    ]
                ),
                # --- confound provenance, carried through per pair ---------
                "treated_is_first_change": bool(
                    treatment.get(
                        "is_first_change_for_team_match", True
                    )
                ),
                "treated_other_change_within_isolation": bool(
                    treatment.get(
                        "other_formation_change_within_isolation_window",
                        False,
                    )
                ),
                "treated_seconds_to_nearest_other_change": (
                    float(
                        treatment.get("seconds_to_nearest_other_change")
                    )
                    if pd.notna(
                        treatment.get("seconds_to_nearest_other_change")
                    )
                    else np.nan
                ),
                "treated_substitutions_within_did_window": (
                    int(
                        treatment.get("substitutions_within_did_window", 0)
                    )
                    if pd.notna(
                        treatment.get("substitutions_within_did_window")
                    )
                    else 0
                ),
                "treated_goals_within_did_window": (
                    int(treatment.get("goals_within_did_window", 0))
                    if pd.notna(treatment.get("goals_within_did_window"))
                    else 0
                ),
                "control_game_id": int(
                    control["control_game_id"]
                ),
                
                "control_team_id": int(
                    control["control_team_id"]
                ),
                "control_team": control[
                    "control_team"
                ],
                "control_opponent_team_id": int(
                    control[
                        "control_opponent_team_id"
                    ]
                ),
                "control_opponent_team": control[
                    "control_opponent_team"
                ],
                "control_opponent_style_cluster": (
                    control_cluster
                ),
                "control_start": float(
                    control["control_start"]
                ),
                "control_minute": float(
                    control["control_start"]
                ) / 60.0,
                "control_period_id": int(
                    control["control_period_id"]
                ),
                "control_minute_bucket": int(
                    control["control_start"]
                    // 300
                ),
                "control_score_difference": int(
                    control[
                        "control_score_difference"
                    ]
                ),
                "control_distance_seconds": float(
                    control[
                        "control_distance_seconds"
                    ]
                ),
                "treated_before_ppda": treatment_ppda[
                    "before_ppda"
                ],
                "treated_after_ppda": treatment_ppda[
                    "after_ppda"
                ],
                "control_before_ppda": control[
                    "control_before_ppda"
                ],
                "control_after_ppda": control[
                    "control_after_ppda"
                ],
                # FIX: counts carried through so the DiD can be estimated as a
                # count model with an exposure offset, where a window with zero
                # defensive actions is a valid y = 0 rather than a dropped row.
                "treated_before_passes": treatment_ppda[
                    "before_passes"
                ],
                "treated_after_passes": treatment_ppda[
                    "after_passes"
                ],
                "treated_before_defensive_actions": treatment_ppda[
                    "before_defensive_actions"
                ],
                "treated_after_defensive_actions": treatment_ppda[
                    "after_defensive_actions"
                ],
                "control_before_passes": control[
                    "control_before_passes"
                ],
                "control_after_passes": control[
                    "control_after_passes"
                ],
                "control_before_defensive_actions": control[
                    "control_before_defensive_actions"
                ],
                "control_after_defensive_actions": control[
                    "control_after_defensive_actions"
                ],
            }
        )

    pairs = pd.DataFrame(rows)

    if pairs.empty:
        raise RuntimeError(
            "No treatment event could be matched to a valid "
            "same-match control window."
        )

    return pairs, unmatched


def _pairs_to_long(
    pairs: pd.DataFrame,
    winsorize_95: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []

    for row in pairs.to_dict(
        orient="records"
    ):
        common = {
            "pair_id": row["pair_id"],
            "match_id": row["match_id"],
            "treated_change_event_id": row[
                "treated_change_event_id"
            ],
            "competition_id": row[
                "competition_id"
            ],
            "season_id": row["season_id"],
        }

        for post, ppda, n_pass, n_act in (
            (0, row["treated_before_ppda"],
             row.get("treated_before_passes"),
             row.get("treated_before_defensive_actions")),
            (1, row["treated_after_ppda"],
             row.get("treated_after_passes"),
             row.get("treated_after_defensive_actions")),
        ):
            rows.append(
                {
                    **common,
                    "opponent_passes": n_pass,
                    "defensive_actions": n_act,
                    "game_id": row[
                        "treated_game_id"
                    ],
                    "source": "treated",
                    "treated": 1,
                    "post": post,
                    "ppda": ppda,
                    "minute": row[
                        "treated_minute"
                    ],
                    "period": row[
                        "treated_period_id"
                    ],
                    "minute_bucket": row[
                        "treated_minute_bucket"
                    ],
                    "score_difference": row[
                        "score_difference"
                    ],
                    "opponent_style_cluster": row[
                        "treated_opponent_style_cluster"
                    ],
                }
            )

        for post, ppda, n_pass, n_act in (
            (0, row["control_before_ppda"],
             row.get("control_before_passes"),
             row.get("control_before_defensive_actions")),
            (1, row["control_after_ppda"],
             row.get("control_after_passes"),
             row.get("control_after_defensive_actions")),
        ):
            rows.append(
                {
                    **common,
                    "opponent_passes": n_pass,
                    "defensive_actions": n_act,
                    "game_id": row[
                        "control_game_id"
                    ],
                    "source": "control",
                    "treated": 0,
                    "post": post,
                    "ppda": ppda,
                    "minute": row[
                        "control_minute"
                    ],
                    "period": row[
                        "control_period_id"
                    ],
                    "minute_bucket": row[
                        "control_minute_bucket"
                    ],
                    "score_difference": row[
                        "control_score_difference"
                    ],
                    "opponent_style_cluster": row[
                        "control_opponent_style_cluster"
                    ],
                }
            )

    long = pd.DataFrame(rows)

    if long.empty:
        raise RuntimeError(
            "No long-format DiD observations were created."
        )

    long["ppda"] = pd.to_numeric(
        long["ppda"],
        errors="coerce",
    )

    for _count_column in ("opponent_passes", "defensive_actions"):
        if _count_column in long.columns:
            long[_count_column] = pd.to_numeric(
                long[_count_column],
                errors="coerce",
            )

    if winsorize_95:
        observed = long["ppda"].dropna()

        if observed.empty:
            raise RuntimeError(
                "Cannot winsorize because all PPDA values are missing."
            )

        cutoff = float(
            observed.quantile(0.95)
        )

        long["ppda"] = long[
            "ppda"
        ].clip(
            upper=cutoff
        )

    return long


def _logistic_missingness(
    long: pd.DataFrame,
) -> tuple[object | None, pd.DataFrame]:
    """
    Test whether PPDA missingness differs by treatment status.

    This is intentionally performed before the complete-case
    restriction. One row represents one matched unit (treated or
    control); missingness is 1 if either before or after PPDA is missing.
    """
    required = {
        "pair_id",
        "match_id",
        "treated",
        "ppda",
        "minute",
        "period",
        "score_difference",
    }

    _require_columns(
        long,
        required,
        "DiD long data",
    )

    unit = (
        long.groupby(
            [
                "pair_id",
                "match_id",
                "treated",
                "minute",
                "period",
                "score_difference",
            ],
            as_index=False,
        )["ppda"]
        .apply(
            lambda values: int(
                values.isna().any()
            )
        )
    )

    if "ppda" not in unit.columns:
        raise RuntimeError(
            "Could not construct the missingness dataset."
        )

    unit = unit.rename(
        columns={
            "ppda": "ppda_missing"
        }
    )

    if (
        unit.empty
        or unit["ppda_missing"].nunique() < 2
        or unit["treated"].nunique() < 2
    ):
        return None, pd.DataFrame(
            [
                {
                    "n_units": len(unit),
                    "missing_rate": (
                        float(
                            unit["ppda_missing"].mean()
                        )
                        if len(unit)
                        else np.nan
                    ),
                    "treated_coefficient": np.nan,
                    "treated_p_value": np.nan,
                    "treatment_predicts_missingness": np.nan,
                    "status": "not_estimable",
                }
            ]
        )

    try:
        model = smf.logit(
            "ppda_missing ~ treated + score_difference "
            "+ minute + C(period)",
            data=unit,
        ).fit(
            disp=False
        )
    except (
        PerfectSeparationError,
        np.linalg.LinAlgError,
        ValueError,
    ) as exc:
        return None, pd.DataFrame(
            [
                {
                    "n_units": len(unit),
                    "missing_rate": float(
                        unit["ppda_missing"].mean()
                    ),
                    "treated_coefficient": np.nan,
                    "treated_p_value": np.nan,
                    "treatment_predicts_missingness": np.nan,
                    "status": (
                        "logistic_model_failed: "
                        f"{type(exc).__name__}"
                    ),
                }
            ]
        )

    p_value = float(
        model.pvalues.get(
            "treated",
            np.nan,
        )
    )

    result = pd.DataFrame(
        [
            {
                "n_units": len(unit),
                "missing_rate": float(
                    unit["ppda_missing"].mean()
                ),
                "treated_coefficient": float(
                    model.params.get(
                        "treated",
                        np.nan,
                    )
                ),
                "treated_p_value": p_value,
                "treatment_predicts_missingness": (
                    bool(
                        p_value <= 0.10
                    )
                    if np.isfinite(p_value)
                    else np.nan
                ),
                "status": (
                    "treatment_not_predictive_at_0_10"
                    if np.isfinite(p_value)
                    and p_value > 0.10
                    else "treatment_predicts_or_unavailable"
                ),
            }
        ]
    )

    return model, result


def _regression_frame(
    long: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "pair_id",
        "match_id",
        "treated",
        "post",
        "ppda",
        "score_difference",
        "minute",
        "period",
        "opponent_style_cluster",
        "competition_id",
        "season_id",
    }

    _require_columns(
        long,
        required,
        "DiD long data",
    )

    df = _numeric(
        long.copy(),
        [
            "match_id",
            "treated",
            "post",
            "ppda",
            "score_difference",
            "minute",
            "period",
            "opponent_style_cluster",
            "competition_id",
            "season_id",
        ],
    ).dropna(
        subset=list(required)
    ).copy()

    if df.empty:
        raise RuntimeError(
            "No complete observations remain for DiD regression."
        )

    df["period"] = df["period"].astype(int)
    df["opponent_style_cluster"] = (
        df["opponent_style_cluster"].astype(int)
    )
    df["match_id"] = df["match_id"].astype(int)

    return df


def run_did_regression(
    long: pd.DataFrame,
    *,
    include_style_cluster: bool = True,
    require_style_reference: bool = False,
) -> tuple[object, pd.DataFrame]:
    """Estimate the two-period DiD with match-clustered standard errors.

    The primary specification uses the highest-pressing cluster as the
    style-cluster reference (derived, not a hard-coded label id).
    Subgroup analyses may lack that level, so they use a valid observed
    reference level unless ``require_style_reference`` is True.
    """
    df = _regression_frame(long)

    terms = ["post", "treated", "post:treated"]

    if df["score_difference"].nunique(dropna=True) > 1:
        terms.append("score_difference")

    if df["minute"].nunique(dropna=True) > 1:
        terms.append("minute")

    if df["period"].nunique(dropna=True) > 1:
        terms.append("C(period)")

    style_reference = None
    if include_style_cluster:
        style_levels = sorted(
            df["opponent_style_cluster"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if len(style_levels) > 1:
            # The reference category is DERIVED, not hard-coded.
            #
            # Note first that this choice does not move the DiD estimate:
            # the reference level only relabels the style dummies, so
            # beta_3 (treated x after) is identical whichever level is
            # chosen.  What it changes is how the style coefficients read.
            #
            # This used to demand cluster 3, which only existed because the
            # clustering step happened to produce four labels.  The number
            # of pressing-regime clusters is now chosen by a silhouette
            # sweep (k = 2 on the 1200-match corpus), so a literal 3 is
            # neither guaranteed to exist nor meaningful if it does --
            # K-means label numbers are arbitrary and change with the seed.
            #
            # The stable, interpretable choice is the most heavily pressing
            # regime: ppda_intensity_rank == 1 is the lowest-PPDA cluster,
            # i.e. the high-press group.  Every style coefficient is then
            # read as "relative to facing a high-pressing opponent", which
            # is the same contrast regardless of k or of how K-means
            # happened to number its clusters this run.  If the rank column
            # is unavailable, fall back to the largest cluster, which keeps
            # the reference the best-estimated one.
            style_reference = None

            rank_col = "opponent_style_intensity_rank"
            if rank_col in df.columns:
                ranked = (
                    df[["opponent_style_cluster", rank_col]]
                    .dropna()
                    .astype(int)
                    .drop_duplicates()
                    .sort_values(rank_col)
                )
                if not ranked.empty:
                    style_reference = int(
                        ranked["opponent_style_cluster"].iloc[0]
                    )

            if style_reference is None:
                style_reference = int(
                    df["opponent_style_cluster"]
                    .dropna()
                    .astype(int)
                    .value_counts()
                    .idxmax()
                )

            if style_reference not in style_levels:
                if require_style_reference:
                    raise RuntimeError(
                        "Derived style reference "
                        f"{style_reference} is absent from the analysis "
                        f"sample (levels present: {style_levels})."
                    )
                style_reference = style_levels[0]

            terms.append(
                "C(opponent_style_cluster, "
                f"Treatment(reference={style_reference}))"
            )

    formula = "ppda ~ " + " + ".join(terms)

    model_spec = smf.ols(formula, data=df)
    exog = np.asarray(model_spec.exog, dtype=float)
    rank = int(np.linalg.matrix_rank(exog))
    n_columns = int(exog.shape[1])

    if rank < n_columns:
        raise RuntimeError(
            "DiD design matrix is rank-deficient: "
            f"rank={rank}, columns={n_columns}."
        )

    if df["match_id"].nunique() < 2:
        raise RuntimeError(
            "At least two match clusters are required for "
            "match-clustered standard errors."
        )

    model = model_spec.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df["match_id"],
        },
    )

    term = "post:treated"
    if term not in model.params.index:
        raise RuntimeError(
            "DiD interaction 'post:treated' was not estimable."
        )

    estimate = float(model.params[term])
    std_error = float(model.bse[term])
    p_value = float(model.pvalues[term])
    ci = model.conf_int().loc[term]

    if not all(np.isfinite([estimate, std_error, p_value])):
        raise RuntimeError(
            "DiD interaction produced a non-finite estimate, standard "
            "error, or p-value."
        )

    result = pd.DataFrame([
        {
            "term": term,
            "estimate_beta3": estimate,
            "std_error": std_error,
            "p_value": p_value,
            "conf_low_95": float(ci.iloc[0]),
            "conf_high_95": float(ci.iloc[1]),
            "n_long_rows": int(len(df)),
            "n_pairs": int(df["pair_id"].nunique()),
            "n_matches": int(df["match_id"].nunique()),
            "significant_p_lt_0_05": bool(p_value < 0.05),
        }
    ])

    return model, result


def run_did_count_regression(
    long: pd.DataFrame,
    cluster_column: str = "match_id",
) -> pd.DataFrame:
    """
    DiD estimated as a count model with an exposure offset.

        defensive_actions ~ NegativeBinomial,  offset = log(opponent_passes)

    beta3 on post:treated is the log RATE RATIO of the opponent's defensive
    actions per opponent pass — algebraically the same quantity PPDA measures
    (PPDA is its reciprocal), estimated on its natural scale.

    Why this and not the ratio
    --------------------------
    PPDA is undefined whenever a five-minute window contains zero defensive
    actions, which happened in 1,075 of 4,712 windows (22.81%) and forced
    930 of 1,617 matched pairs out of the complete-case regression — with
    treatment borderline-predictive of that missingness (p = 0.089).

    A window with zero defensive actions against forty opponent passes is not
    missing information. It is the strongest available evidence of NOT
    pressing. A count model represents it as y = 0, so every one of those
    windows re-enters the sample and the selection concern disappears with
    them. Nothing is imputed: PPDA's NaN is never filled, the outcome variable
    is simply a different, better-behaved quantity.

    The only requirement is opponent_passes > 0 for the offset. Windows with
    neither passes nor actions carry no information and are dropped explicitly,
    and the count is reported.
    """
    required = {
        "defensive_actions",
        "opponent_passes",
        "post",
        "treated",
        cluster_column,
    }
    _require_columns(long, required, "count-model DiD frame")

    df = long.copy()
    df = _numeric(
        df,
        [
            "defensive_actions",
            "opponent_passes",
            "post",
            "treated",
            "score_difference",
            "minute",
        ],
    )

    n_input = int(len(df))

    usable = (
        df["defensive_actions"].notna()
        & df["opponent_passes"].notna()
        & (df["opponent_passes"] > 0)
    )
    n_zero_exposure = int((~usable).sum())
    df = df.loc[usable].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "No rows with a positive pass exposure remain for the count model."
        )

    terms = ["post * treated"]
    if "score_difference" in df.columns and df["score_difference"].notna().any():
        terms.append("score_difference")
    if "minute" in df.columns and df["minute"].notna().any():
        terms.append("minute")
    if "competition_id" in df.columns and df["competition_id"].nunique() > 1:
        terms.append("C(competition_id)")

    formula = "defensive_actions ~ " + " + ".join(terms)
    offset = np.log(df["opponent_passes"].to_numpy(dtype=float))

    import statsmodels.api as sm

    # Estimate the negative-binomial dispersion instead of leaving it at the
    # statsmodels default of alpha = 1.0, which is an arbitrary assumption that
    # would show up directly in the standard errors. Cameron & Trivedi's
    # auxiliary regression: ((y - mu)^2 - y) / mu  ~  mu, through the origin.
    poisson_fit = smf.glm(
        formula,
        data=df,
        family=sm.families.Poisson(),
        offset=offset,
    ).fit()

    mu = np.asarray(poisson_fit.mu, dtype=float)
    y = df["defensive_actions"].to_numpy(dtype=float)
    aux_y = ((y - mu) ** 2 - y) / np.where(mu > 0, mu, np.nan)
    finite = np.isfinite(aux_y) & np.isfinite(mu)
    alpha = float(
        sm.OLS(aux_y[finite], mu[finite]).fit().params[0]
    )
    alpha = max(alpha, 1e-6)

    pearson_dispersion = float(
        poisson_fit.pearson_chi2 / poisson_fit.df_resid
    )

    rows: list[dict] = []
    for family_name, family in (
        ("negative_binomial", sm.families.NegativeBinomial(alpha=alpha)),
        ("poisson", sm.families.Poisson()),
    ):
        try:
            model = smf.glm(
                formula,
                data=df,
                family=family,
                offset=offset,
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": df[cluster_column]},
            )
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            rows.append(
                {
                    "family": family_name,
                    "status": f"failed: {type(error).__name__}",
                    "error": str(error),
                }
            )
            continue

        term = "post:treated"
        estimate = float(model.params[term])
        std_error = float(model.bse[term])
        conf = model.conf_int().loc[term]

        rows.append(
            {
                "term": term,
                "family": family_name,
                "status": "estimated",
                # log rate ratio
                "estimate_beta3_log_rr": estimate,
                "std_error": std_error,
                "p_value": float(model.pvalues[term]),
                "conf_low_95": float(conf[0]),
                "conf_high_95": float(conf[1]),
                # the same thing as a multiplicative effect, easier to read
                "rate_ratio": float(np.exp(estimate)),
                "rate_ratio_low_95": float(np.exp(conf[0])),
                "rate_ratio_high_95": float(np.exp(conf[1])),
                # PPDA is 1 / (actions per pass), so the implied PPDA ratio
                # is the reciprocal of the rate ratio
                "implied_ppda_ratio": float(np.exp(-estimate)),
                "n_rows_input": n_input,
                "n_rows_used": int(len(df)),
                "n_dropped_zero_exposure": n_zero_exposure,
                "n_clusters": int(df[cluster_column].nunique()),
                "significant_p_lt_0_05": bool(model.pvalues[term] < 0.05),
                # minimum detectable effect at 80% power, on the log scale
                "mde_80_power_log_rr": float(2.8016 * std_error),
                # dispersion diagnostics: Poisson assumes dispersion == 1
                "nb_alpha_estimated": alpha,
                "poisson_pearson_dispersion": pearson_dispersion,
            }
        )

    return pd.DataFrame(rows)


def _run_subgroup(
    long: pd.DataFrame,
    mask: pd.Series,
    label,
    column_name: str,
) -> dict:
    subset = long.loc[mask].copy()

    if subset.empty:
        return {
            column_name: label,
            "status": "no_observations",
            "estimate_beta3": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "n_long_rows": 0,
            "n_pairs": 0,
            "n_matches": 0,
        }

    try:
        _, result = run_did_regression(
            subset,
            include_style_cluster=(column_name != "opponent_style_cluster"),
            require_style_reference=False,
        )
        row = result.iloc[0].to_dict()
        row[column_name] = label
        row["status"] = "estimated"
        return row
    except (
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return {
            column_name: label,
            "status": (
                "not_estimable: "
                f"{type(exc).__name__}"
            ),
            "estimate_beta3": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "n_long_rows": len(subset),
            "n_pairs": subset[
                "pair_id"
            ].nunique(),
            "n_matches": subset[
                "match_id"
            ].nunique(),
        }


def run_score_state_heterogeneity(
    long: pd.DataFrame,
) -> pd.DataFrame:
    """Run separate DiDs for losing, drawing and winning states."""
    return pd.DataFrame(
        [
            _run_subgroup(
                long,
                long["score_difference"] < 0,
                "losing",
                "score_state",
            ),
            _run_subgroup(
                long,
                long["score_difference"] == 0,
                "drawing",
                "score_state",
            ),
            _run_subgroup(
                long,
                long["score_difference"] > 0,
                "winning",
                "score_state",
            ),
        ]
    )


def run_style_cluster_heterogeneity(
    long: pd.DataFrame,
) -> pd.DataFrame:
    """Run separate DiDs for each opponent style cluster."""
    clusters = sorted(
        pd.to_numeric(
            long["opponent_style_cluster"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    rows = []

    for cluster in clusters:
        rows.append(
            _run_subgroup(
                long,
                long["opponent_style_cluster"] == cluster,
                int(cluster),
                "opponent_style_cluster",
            )
        )

    return pd.DataFrame(rows)


def build_did_dataset(
    processed_dir: str | Path = "processed-data",
    config: DiDConfig = DiDConfig(),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build treatment/control pairs and the complete-case primary sample.

    Returns (pairs, long, construction, missingness, long_all).

    `long_all` is the long frame BEFORE the complete-case PPDA restriction —
    every matched pair, including the 930 that the ratio-based analysis has to
    discard because a five-minute window contained zero defensive actions.
    It is the frame `run_did_count_regression` is meant to consume: in a count
    model those windows are valid y = 0 observations, not missing data.
    """
    processed_dir = (
        Path(processed_dir)
        .expanduser()
        .resolve()
    )

    experiments = pd.read_csv(
        processed_dir
        / "natural_experiments.csv"
    )

    inventory = pd.read_csv(
        processed_dir
        / "dataset_inventory.csv"
    )

    styles = _load_style_clusters(
        processed_dir
    )

    treatments, counts = _prepare_treatments(
        experiments,
        inventory,
        config,
    )

    loader = StatsBombDataLoader()

    game_ids = sorted(
        treatments["game_id"]
        .astype(int)
        .unique()
        .tolist()
    )

    games = (
        inventory[
            inventory["game_id"]
            .astype(int)
            .isin(game_ids)
        ]
        .drop_duplicates(
            subset=["game_id"]
        )
    )

    game_cache = {}

    for position, game in enumerate(
        games.to_dict(
            orient="records"
        ),
        start=1,
    ):
        game_id = int(
            game["game_id"]
        )

        game_cache[game_id] = (
            _prepare_game(
                loader,
                game,
            )
        )

        if (
            position % 100 == 0
            or position == len(games)
        ):
            print(
                f"Prepared games: "
                f"{position}/{len(games)}"
            )

    pairs, unmatched = _build_pairs(
        treatments,
        game_cache,
        styles,
        config,
    )

    counts[
        "matched_treatments_before_complete_case"
    ] = int(
        len(pairs)
    )

    counts[
        "unmatched_treatments"
    ] = int(
        len(unmatched)
    )

    # Missingness check MUST happen before complete-case filtering.
    long_all = _pairs_to_long(
        pairs,
        winsorize_95=False,
    )

    _, missingness = _logistic_missingness(
        long_all
    )

    # Primary analysis: complete treatment AND control PPDA pairs.
    complete = (
        pairs["treated_before_ppda"].notna()
        & pairs["treated_after_ppda"].notna()
        & pairs["control_before_ppda"].notna()
        & pairs["control_after_ppda"].notna()
    )

    counts[
        "dropped_incomplete_pairs"
    ] = int(
        (~complete).sum()
    )

    pairs = pairs.loc[
        complete
    ].reset_index(
        drop=True
    )

    if pairs.empty:
        raise RuntimeError(
            "No complete treatment/control pairs remain "
            "after the complete-case PPDA restriction."
        )

    counts[
        "complete_treatment_control_pairs"
    ] = int(
        len(pairs)
    )

    long = _pairs_to_long(
        pairs,
        winsorize_95=config.winsorize_95,
    )

    counts[
        "primary_long_rows"
    ] = int(
        len(long)
    )

    construction = pd.DataFrame(
        [
            {
                "stage": key,
                "value": value,
            }
            for key, value in counts.items()
        ]
    )

    return (
        pairs,
        long,
        construction,
        missingness,
        long_all,
    )


def _save_analysis(
    output_dir: Path,
    pairs: pd.DataFrame,
    long: pd.DataFrame,
    construction: pd.DataFrame,
    missingness: pd.DataFrame,
    result: pd.DataFrame,
    score_state: pd.DataFrame,
    style_heterogeneity: pd.DataFrame,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs.to_csv(
        output_dir
        / "did_matched_pairs.csv",
        index=False,
    )

    long.to_csv(
        output_dir
        / "did_long.csv",
        index=False,
    )

    construction.to_csv(
        output_dir
        / "did_sample_construction.csv",
        index=False,
    )

    missingness.to_csv(
        output_dir
        / "did_missingness_check.csv",
        index=False,
    )

    result.to_csv(
        output_dir
        / "did_results.csv",
        index=False,
    )

    score_state.to_csv(
        output_dir
        / "did_score_state_heterogeneity.csv",
        index=False,
    )

    style_heterogeneity.to_csv(
        output_dir
        / "did_style_cluster_heterogeneity.csv",
        index=False,
    )


def _run_specification(
    processed_dir: Path,
    config: DiDConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    object,
]:
    pairs, long, construction, missingness, _long_all = (
        build_did_dataset(
            processed_dir,
            config,
        )
    )

    model, result = run_did_regression(
        long,
        include_style_cluster=False,
        require_style_reference=False,
    )

    return (
        pairs,
        long,
        construction,
        missingness,
        model,
    )


def _run_robustness(
    processed_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the robustness analyses requested by the team."""

    rows: list[dict] = []

    specifications = [
        (
            "window_3_minutes",
            DiDConfig(
                window_minutes=3,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
            ),
        ),
        (
            "window_10_minutes",
            DiDConfig(
                window_minutes=10,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
            ),
        ),
        (
            "minimum_10_minute_gap",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
                first_change_per_team_match=False,
                minimum_change_gap_minutes=10,
            ),
        ),
        (
            "winsorize_95",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
                winsorize_95=True,
            ),
        ),
        (
            "exclude_opponent_responses",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
                exclude_opponent_responses=True,
            ),
        ),
        (
            "halftime_only",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=False,
                only_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
            ),
        ),
        (
            "extra_time_only",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(3, 4),
                max_match_seconds=EXTRA_TIME_MATCH_SECONDS,
            ),
        ),
        # --- confound robustness grid (see DiDConfig) ---------------------
        # One row per filter so each exclusion is visible, plus a
        # joint row with all three switched on.  If the primary
        # estimate survives this grid, the causal claim is stronger;
        # if it does not, the sensitivity is reported as such.
        (
            "bilateral_isolation",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
                exclude_bilateral_confound=True,
            ),
        ),
        (
            "no_concurrent_substitution",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
                exclude_any_substitution_nearby=True,
            ),
        ),
        (
            "no_concurrent_goal",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
                exclude_any_goal_nearby=True,
            ),
        ),
        (
            "all_confounds_excluded",
            DiDConfig(
                window_minutes=5,
                control_search_minutes=15,
                control_grid_seconds=60,
                score_tolerance=1,
                exclude_halftime=True,
                allowed_periods=(1, 2),
                max_match_seconds=MAIN_MATCH_SECONDS,
                exclude_bilateral_confound=True,
                exclude_any_substitution_nearby=True,
                exclude_any_goal_nearby=True,
            ),
        ),
    ]

    for name, config in specifications:
        try:
            pairs, long, _, _, _ = build_did_dataset(
                processed_dir=processed_dir,
                config=config,
            )

            # Robustness specifications must remain estimable even if a
            # particular restricted sample lacks style cluster 3.  The
            # primary specification remains strict about cluster 3 below.
            _, result = run_did_regression(
                long,
                include_style_cluster=False,
                require_style_reference=False,
            )

            row = result.iloc[0].to_dict()
            row["analysis"] = name
            row["status"] = "estimated"
            row["matched_pairs"] = int(len(pairs))
            if config.winsorize_95:
                raw_long = _pairs_to_long(pairs, winsorize_95=False)
                raw_ppda = pd.to_numeric(
                    raw_long["ppda"], errors="coerce"
                ).dropna()
                row["winsorize_cutoff_95"] = (
                    float(raw_ppda.quantile(0.95))
                    if not raw_ppda.empty else np.nan
                )
            else:
                row["winsorize_cutoff_95"] = np.nan
            rows.append(row)

        except (
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            rows.append(
                {
                    "analysis": name,
                    "status": (
                        "failed: "
                        f"{type(exc).__name__}"
                    ),
                    "estimate_beta3": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "matched_pairs": 0,
                    "winsorize_cutoff_95": np.nan,
                    "error": str(exc),
                }
            )

    robustness = pd.DataFrame(rows)

    robustness.to_csv(
        output_dir / "did_robustness.csv",
        index=False,
    )

    return robustness


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SportsM5 Difference-in-Differences validation "
            "using same-match matched controls."
        )
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed-data"),
        help=(
            "Directory containing natural_experiments.csv, "
            "dataset_inventory.csv and team_style_clusters.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run only the primary DiD and skip robustness analyses.",
    )

    args = parser.parse_args()

    processed_dir = (
        args.processed_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        if args.output_dir is not None
        else processed_dir
    )

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed-data directory not found: {processed_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_config = DiDConfig(
        window_minutes=5,
        control_search_minutes=15,
        control_grid_seconds=60,
        score_tolerance=1,
        exclude_halftime=True,
        only_halftime=False,
        allowed_periods=(1, 2),
        first_change_per_team_match=True,
        exclude_opponent_responses=False,
        winsorize_95=False,
    )

    (
        pairs,
        long,
        construction,
        missingness,
        model,
    ) = _run_specification(
        processed_dir,
        primary_config,
    )

    primary_result = pd.DataFrame([
        {
            "term": "post:treated",
            "estimate_beta3": float(model.params["post:treated"]),
            "std_error": float(model.bse["post:treated"]),
            "p_value": float(model.pvalues["post:treated"]),
            "conf_low_95": float(model.conf_int().loc["post:treated"].iloc[0]),
            "conf_high_95": float(model.conf_int().loc["post:treated"].iloc[1]),
            "n_long_rows": int(len(long)),
            "n_pairs": int(long["pair_id"].nunique()),
            "n_matches": int(long["match_id"].nunique()),
            "significant_p_lt_0_05": bool(model.pvalues["post:treated"] < 0.05),
        }
    ])

    score_state = (
        run_score_state_heterogeneity(
            long
        )
    )

    style_heterogeneity = (
        run_style_cluster_heterogeneity(
            long
        )
    )

    _save_analysis(
        output_dir,
        pairs,
        long,
        construction,
        missingness,
        primary_result,
        score_state,
        style_heterogeneity,
    )

    print()
    print("=" * 80)
    print("SPORTSM5 DIFFERENCE-IN-DIFFERENCES")
    print("=" * 80)

    print(
        f"Primary matched pairs: "
        f"{len(pairs)}"
    )

    print(
        f"Primary long observations: "
        f"{len(long)}"
    )

    print(
        f"Match clusters: "
        f"{long['match_id'].nunique()}"
    )

    print()
    print("SAMPLE CONSTRUCTION")
    print(construction.to_string(index=False))

    print()
    print("MISSINGNESS CHECK")
    print(missingness.to_string(index=False))

    print()
    print("PRIMARY DiD RESULT")
    print(primary_result.to_string(index=False))

    print()
    print("SCORE-STATE HETEROGENEITY")
    print(score_state.to_string(index=False))

    print()
    print("OPPONENT STYLE-CLUSTER HETEROGENEITY")
    print(style_heterogeneity.to_string(index=False))

    if not args.primary_only:
        print()
        print("ROBUSTNESS CHECKS")
        robustness = _run_robustness(
            processed_dir,
            output_dir,
        )
        print(
            robustness.to_string(
                index=False
            )
        )

    print()
    print("MODEL SUMMARY")
    print(model.summary())

    print()
    print("OUTPUT DIRECTORY")
    print(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
