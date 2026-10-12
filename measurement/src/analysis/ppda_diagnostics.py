from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocessing import EventPreprocessor
from data.statsbomb_loader import StatsBombDataLoader
from analysis.ppda import compute_ppda


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
    relevant = events[
        events["event_seconds"] < float(time_seconds)
    ]

    goals = relevant[
        relevant["is_goal"]
        .fillna(False)
        .astype(bool)
    ]

    score_for = int(
        (
            goals["team_id"]
            == int(team_id)
        ).sum()
    )

    score_against = int(
        (
            goals["team_id"]
            != int(team_id)
        ).sum()
    )

    return (
        score_for,
        score_against,
        score_for - score_against,
    )


def _get_opponent_id(
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
        value
        for value in teams
        if value != int(team_id)
    ]

    if len(opponents) != 1:
        raise ValueError(
            f"Expected exactly one opponent for team "
            f"{team_id}; found {opponents}."
        )

    return int(opponents[0])


def _prepare_game(
    loader: StatsBombDataLoader,
    game: dict,
) -> pd.DataFrame:
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

    return canonical


def _window_diagnostics(
    events: pd.DataFrame,
    pressing_team_id: int,
    opponent_team_id: int,
    start_seconds: float,
    end_seconds: float,
) -> dict:
    result = compute_ppda(
        events,
        pressing_team_id=pressing_team_id,
        opponent_team_id=opponent_team_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    return {
        "passes": int(result["passes"]),
        "defensive_actions": int(
            result["defensive_actions"]
        ),
        "ppda": result["ppda"],
    }


def _event_type_counts(
    events: pd.DataFrame,
    pressing_team_id: int,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, int]:
    window = events[
        (events["event_seconds"] >= start_seconds)
        & (events["event_seconds"] < end_seconds)
        & (
            events["team_id"]
            == int(pressing_team_id)
        )
    ]

    counts = (
        window["event_type"]
        .value_counts()
        .to_dict()
    )

    out = {
        str(event_type): int(count)
        for event_type, count in counts.items()
    }

    # StatsBomb has no literal "Tackle" event type -- a tackle is a Duel
    # with duel.type.name == "Tackle".  Reading event_type == "Tackle"
    # therefore always returns 0, which is what this diagnostic used to
    # report even though compute_ppda was counting tackles correctly via
    # is_tackle.  A reader comparing "tackles found: 222" from the raw JSON
    # against "mean tackle_events = 0.00" here would conclude the fix had
    # not landed.  Count them the same way compute_ppda does.
    if "is_tackle" in window.columns:
        out["__tackles__"] = int(
            window["is_tackle"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
    elif "duel_type" in window.columns:
        out["__tackles__"] = int(
            (
                window["event_type"].eq("Duel")
                & window["duel_type"].eq("Tackle")
            ).sum()
        )
    else:
        out["__tackles__"] = 0

    return out


def _analyse_window(
    events: pd.DataFrame,
    pressing_team_id: int,
    opponent_team_id: int,
    start_seconds: float,
    end_seconds: float,
) -> dict:
    result = _window_diagnostics(
        events=events,
        pressing_team_id=pressing_team_id,
        opponent_team_id=opponent_team_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    event_counts = _event_type_counts(
        events=events,
        pressing_team_id=pressing_team_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    return {
        **result,
        "total_pressing_team_events": int(
            sum(
                count
                for key, count in event_counts.items()
                if not key.startswith("__")
            )
        ),
        "pressure_events": int(
            event_counts.get("Pressure", 0)
        ),
        "tackle_events": int(
            event_counts.get("__tackles__", 0)
        ),
        "interception_events": int(
            event_counts.get("Interception", 0)
        ),
        "block_events": int(
            event_counts.get("Block", 0)
        ),
        "foul_committed_events": int(
            event_counts.get("Foul Committed", 0)
        ),
        "ball_recovery_events": int(
            event_counts.get("Ball Recovery", 0)
        ),
        "clearance_events": int(
            event_counts.get("Clearance", 0)
        ),
        "dispossessed_events": int(
            event_counts.get("Dispossessed", 0)
        ),
        "duel_events": int(
            event_counts.get("Duel", 0)
        ),
    }


def _analyse_treatment_row(
    loader: StatsBombDataLoader,
    game: dict,
    treatment: dict,
    window_minutes: float,
) -> list[dict]:
    events = _prepare_game(
        loader,
        game,
    )

    treatment_time = float(
        treatment["change_time"]
    )

    focal_team_id = int(
        treatment["team_id"]
    )

    opponent_team_id = _get_opponent_id(
        events,
        focal_team_id,
    )

    window_seconds = (
        float(window_minutes) * 60.0
    )

    rows = []

    for label, start, end in (
        (
            "before",
            treatment_time - window_seconds,
            treatment_time,
        ),
        (
            "after",
            treatment_time,
            treatment_time + window_seconds,
        ),
    ):
        diagnostics = _analyse_window(
            events=events,
            pressing_team_id=opponent_team_id,
            opponent_team_id=focal_team_id,
            start_seconds=start,
            end_seconds=end,
        )

        (
            score_for,
            score_against,
            score_difference,
        ) = _score_at_time(
            events,
            focal_team_id,
            start,
        )

        rows.append(
            {
                "game_id": int(
                    treatment["game_id"]
                ),
                "team_id": focal_team_id,
                "team": treatment["team"],
                "opponent_team_id": opponent_team_id,
                "opponent_team": treatment[
                    "opponent_team"
                ],
                "change_event_id": str(
                    treatment["change_event_id"]
                ),
                "change_time_seconds": (
                    treatment_time
                ),
                "change_minute": (
                    treatment_time / 60.0
                ),
                "period_id": int(
                    treatment["period_id"]
                ),
                "window": label,
                "window_start_seconds": float(
                    start
                ),
                "window_end_seconds": float(
                    end
                ),
                "score_for": score_for,
                "score_against": score_against,
                "score_difference": (
                    score_difference
                ),
                **diagnostics,
            }
        )

    return rows


def _load_inputs(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiments_path = (
        processed_dir
        / "natural_experiments.csv"
    )

    inventory_path = (
        processed_dir
        / "dataset_inventory.csv"
    )

    if not experiments_path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{experiments_path}"
        )

    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{inventory_path}"
        )

    experiments = pd.read_csv(
        experiments_path
    )

    inventory = pd.read_csv(
        inventory_path
    )

    _require_columns(
        experiments,
        {
            "game_id",
            "team_id",
            "team",
            "opponent_team_id",
            "opponent_team",
            "change_event_id",
            "change_time",
            "period_id",
            "is_halftime_shift",
            "opponent_response_is_shift",
        },
        "natural_experiments.csv",
    )

    _require_columns(
        inventory,
        {
            "game_id",
            "home_team_id",
        },
        "dataset_inventory.csv",
    )

    experiments = _numeric(
        experiments,
        [
            "game_id",
            "team_id",
            "opponent_team_id",
            "change_time",
            "period_id",
        ],
    )

    inventory = _numeric(
        inventory,
        [
            "game_id",
            "home_team_id",
        ],
    )

    experiments["is_halftime_shift"] = (
        experiments[
            "is_halftime_shift"
        ]
        .fillna(False)
        .astype(bool)
    )

    experiments[
        "opponent_response_is_shift"
    ] = (
        experiments[
            "opponent_response_is_shift"
        ]
        .fillna(False)
        .astype(bool)
    )

    return experiments, inventory


def _select_primary_treatments(
    experiments: pd.DataFrame,
) -> pd.DataFrame:
    df = experiments.copy()

    # Primary DiD rules from the team:
    # - normal match play only
    # - exclude halftime shifts
    # - keep first eligible formation change
    #   per team per match
    df = df[
        ~df["is_halftime_shift"]
    ].copy()

    df = df[
        df["period_id"].isin([1, 2])
    ].copy()

    df = df.dropna(
        subset=[
            "game_id",
            "team_id",
            "change_event_id",
            "change_time",
            "period_id",
        ]
    ).copy()

    df = (
        df.sort_values(
            [
                "game_id",
                "team_id",
                "change_time",
                "change_event_id",
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "game_id",
                "team_id",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return df


def run_diagnostic(
    processed_dir: str | Path = "processed-data",
    output_dir: str | Path | None = None,
    window_minutes: float = 5.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    processed_dir = (
        Path(processed_dir)
        .expanduser()
        .resolve()
    )

    if output_dir is None:
        output_dir = processed_dir / "ppda-diagnostics"
    else:
        output_dir = (
            Path(output_dir)
            .expanduser()
            .resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiments, inventory = _load_inputs(
        processed_dir
    )

    treatments = _select_primary_treatments(
        experiments
    )

    inventory_by_game = (
        inventory[
            [
                "game_id",
                "home_team_id",
            ]
        ]
        .drop_duplicates(
            subset=["game_id"]
        )
        .set_index("game_id")
    )

    loader = StatsBombDataLoader()

    rows: list[dict] = []

    total = len(treatments)

    for position, treatment in enumerate(
        treatments.to_dict(
            orient="records"
        ),
        start=1,
    ):
        game_id = int(
            treatment["game_id"]
        )

        if game_id not in inventory_by_game.index:
            continue

        game = inventory_by_game.loc[
            game_id
        ].to_dict()

        game["game_id"] = game_id

        rows.extend(
            _analyse_treatment_row(
                loader=loader,
                game=game,
                treatment=treatment,
                window_minutes=window_minutes,
            )
        )

        if (
            position % 100 == 0
            or position == total
        ):
            print(
                f"Diagnosed treatment events: "
                f"{position}/{total}"
            )

    detail = pd.DataFrame(rows)

    if detail.empty:
        raise RuntimeError(
            "No PPDA diagnostic rows were produced."
        )

    detail["ppda_missing"] = (
        detail["ppda"].isna()
    )

    detail["zero_defensive_actions"] = (
        detail["defensive_actions"] == 0
    )

    detail["zero_passes"] = (
        detail["passes"] == 0
    )

    detail_path = (
        output_dir
        / "ppda_window_diagnostics.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    summary = (
        detail.groupby(
            [
                "window",
                "ppda_missing",
                "zero_defensive_actions",
            ],
            dropna=False,
        )
        .agg(
            windows=("game_id", "size"),
            mean_passes=("passes", "mean"),
            median_passes=("passes", "median"),
            mean_defensive_actions=(
                "defensive_actions",
                "mean",
            ),
            median_defensive_actions=(
                "defensive_actions",
                "median",
            ),
            mean_total_pressing_events=(
                "total_pressing_team_events",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        output_dir
        / "ppda_missingness_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return detail, summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose undefined PPDA windows "
            "without changing the PPDA formula."
        )
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed-data"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--window-minutes",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    detail, summary = run_diagnostic(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        window_minutes=args.window_minutes,
    )

    print()
    print("=" * 80)
    print("SPORTSM5 PPDA DIAGNOSTIC")
    print("=" * 80)

    print()
    print("WINDOW COUNTS")
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("OVERALL WINDOWS")
    print(
        f"Total windows: "
        f"{len(detail)}"
    )

    print(
        f"PPDA undefined: "
        f"{int(detail['ppda_missing'].sum())}"
    )

    print(
        f"PPDA undefined rate: "
        f"{detail['ppda_missing'].mean():.4f}"
    )

    print(
        f"Zero defensive-action windows: "
        f"{int(detail['zero_defensive_actions'].sum())}"
    )

    print()
    print("OUTPUT FILES")
    print(
        f"Detailed: "
        f"{args.output_dir or (Path(args.processed_dir) / 'ppda-diagnostics')}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())