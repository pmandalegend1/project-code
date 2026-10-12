from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from data.statsbomb_loader import (
    StatsBombDataLoader,
)

from data.preprocessing import (
    EventPreprocessor,
)

from analysis.formation import (
    build_formation_timeline,
)

from analysis.ppda import (
    compute_team_match_ppda,
    compute_15_minute_ppda,
)

from analysis.natural_experiments import (
    extract_natural_experiments,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Process every locally available "
            "StatsBomb Open Data match for SportsM5."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where processed CSV "
            "files will be written."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N matches. "
            "Use this for a small test run. "
            "Default: process everything."
        ),
    )

    parser.add_argument(
        "--no-lineup-check",
        action="store_true",
        help=(
            "Do not check whether a lineup JSON "
            "exists before processing."
        ),
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loader = StatsBombDataLoader()

    print()
    print("=" * 80)
    print("SPORTSM5 FULL DATASET PROCESSOR")
    print("=" * 80)

    competitions = (
        loader.get_competitions()
    )

    if competitions.empty:
        raise RuntimeError(
            "No competitions/seasons were found."
        )

    competition_pairs = (
        competitions[
            [
                "competition_id",
                "season_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        "Competition/season pairs found: "
        f"{len(competition_pairs)}"
    )

    # --------------------------------------------------------------
    # Build complete game inventory first.
    # --------------------------------------------------------------

    game_inventory = []

    for pair in competition_pairs.itertuples(
        index=False
    ):

        competition_id = int(
            pair.competition_id
        )

        season_id = int(
            pair.season_id
        )

        games = loader.get_games(
            competition_id,
            season_id,
        )

        for game in games.to_dict(
            orient="records"
        ):

            game_id = int(
                game["game_id"]
            )

            game_inventory.append(
                {
                    **game,
                    "competition_id": (
                        competition_id
                    ),
                    "season_id": (
                        season_id
                    ),
                    "event_file_exists": (
                        loader.event_file_exists(
                            game_id
                        )
                    ),
                    "lineup_file_exists": (
                        loader.lineup_file_exists(
                            game_id
                        )
                    ),
                }
            )

    inventory = pd.DataFrame(
        game_inventory
    )

    inventory_path = (
        output_dir
        / "dataset_inventory.csv"
    )

    inventory.to_csv(
        inventory_path,
        index=False,
    )

    print(
        "Games discovered: "
        f"{len(inventory)}"
    )

    print(
        "Games with event files: "
        f"{int(inventory['event_file_exists'].sum())}"
    )

    if not args.no_lineup_check:
        print(
            "Games with lineup files: "
            f"{int(inventory['lineup_file_exists'].sum())}"
        )

    print(
        f"Inventory saved to: {inventory_path}"
    )

    # --------------------------------------------------------------
    # Select games to process.
    # --------------------------------------------------------------

    games_to_process = inventory[
        inventory["event_file_exists"]
    ].copy()

    games_to_process = (
        games_to_process.sort_values(
            [
                "competition_id",
                "season_id",
                "game_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        games_to_process = (
            games_to_process.iloc[
                : args.limit
            ]
        )

    print(
        "Games selected for processing: "
        f"{len(games_to_process)}"
    )

    # --------------------------------------------------------------
    # Output accumulators.
    # --------------------------------------------------------------

    match_summary_rows = []
    match_ppda_rows = []
    ppda_15_rows = []
    experiment_rows = []
    error_rows = []

    total_events = 0
    total_canonical_events = 0
    total_tactical_shifts = 0
    total_halftime_shifts = 0
    total_formation_changes = 0
    total_eligible_changes = 0

    start_time = time.perf_counter()

    # --------------------------------------------------------------
    # Match-by-match processing.
    # --------------------------------------------------------------

    for position, game in enumerate(
        games_to_process.to_dict(
            orient="records"
        ),
        start=1,
    ):

        game_id = int(
            game["game_id"]
        )

        competition_id = int(
            game["competition_id"]
        )

        season_id = int(
            game["season_id"]
        )

        print(
            f"[{position}/{len(games_to_process)}] "
            f"Processing game {game_id}..."
        )

        try:

            if "home_team_id" not in game:
                raise ValueError(
                    "Game record does not contain "
                    "'home_team_id'."
                )

            home_team_id = int(
                game["home_team_id"]
            )

            events = loader.get_events(
                game_id
            )

            raw_event_count = len(
                events
            )

            canonical = (
                EventPreprocessor(
                    events,
                    home_team_id=home_team_id,
                ).transform()
            )

            canonical_event_count = len(
                canonical
            )

            if (
                raw_event_count
                != canonical_event_count
            ):
                raise ValueError(
                    "Canonical preprocessing "
                    "changed event count: "
                    f"{raw_event_count} -> "
                    f"{canonical_event_count}"
                )

            # ------------------------------------------------------
            # Formation
            # ------------------------------------------------------

            formation_timeline = (
                build_formation_timeline(
                    canonical
                )
            )

            tactical_shift_count = int(
                canonical[
                    "is_tactical_shift"
                ].sum()
            )

            halftime_shift_count = int(
                canonical[
                    "is_halftime_shift"
                ].sum()
            )

            formation_change_count = int(
                formation_timeline[
                    "is_formation_change"
                ].sum()
            )

            eligible_change_count = int(
                formation_timeline[
                    "is_eligible_natural_experiment"
                ].sum()
            )

            # ------------------------------------------------------
            # PPDA
            # ------------------------------------------------------

            match_ppda = (
                compute_team_match_ppda(
                    canonical
                )
            )

            ppda_15 = (
                compute_15_minute_ppda(
                    canonical
                )
            )

            # ------------------------------------------------------
            # Natural experiments
            # ------------------------------------------------------

            experiments = (
                extract_natural_experiments(
                    canonical,
                    formation_timeline,
                )
            )

            # ------------------------------------------------------
            # Attach dataset metadata
            # ------------------------------------------------------

            match_ppda = (
                match_ppda.copy()
            )

            match_ppda[
                "competition_id"
            ] = competition_id

            match_ppda[
                "season_id"
            ] = season_id

            ppda_15 = ppda_15.copy()

            ppda_15[
                "competition_id"
            ] = competition_id

            ppda_15[
                "season_id"
            ] = season_id

            experiments = (
                experiments.copy()
            )

            if not experiments.empty:

                experiments[
                    "competition_id"
                ] = competition_id

                experiments[
                    "season_id"
                ] = season_id

            # ------------------------------------------------------
            # Store rows
            # ------------------------------------------------------

            match_summary_rows.append(
                {
                    "competition_id": (
                        competition_id
                    ),
                    "season_id": season_id,
                    "game_id": game_id,
                    "home_team_id": (
                        home_team_id
                    ),
                    "raw_events": (
                        raw_event_count
                    ),
                    "canonical_events": (
                        canonical_event_count
                    ),
                    "tactical_shifts": (
                        tactical_shift_count
                    ),
                    "halftime_shifts": (
                        halftime_shift_count
                    ),
                    "formation_changes": (
                        formation_change_count
                    ),
                    "eligible_natural_experiments": (
                        eligible_change_count
                    ),
                    "natural_experiment_rows": (
                        len(experiments)
                    ),
                }
            )

            match_ppda_rows.append(
                match_ppda
            )

            ppda_15_rows.append(
                ppda_15
            )

            if not experiments.empty:
                experiment_rows.append(
                    experiments
                )

            total_events += (
                raw_event_count
            )

            total_canonical_events += (
                canonical_event_count
            )

            total_tactical_shifts += (
                tactical_shift_count
            )

            total_halftime_shifts += (
                halftime_shift_count
            )

            total_formation_changes += (
                formation_change_count
            )

            total_eligible_changes += (
                eligible_change_count
            )

        except Exception as exc:

            error_rows.append(
                {
                    "competition_id": (
                        competition_id
                    ),
                    "season_id": season_id,
                    "game_id": game_id,
                    "error_type": type(
                        exc
                    ).__name__,
                    "error_message": str(
                        exc
                    ),
                }
            )

            print(
                f"  ERROR: {type(exc).__name__}: "
                f"{exc}"
            )

    # --------------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------------

    summary = pd.DataFrame(
        match_summary_rows
    )

    errors = pd.DataFrame(
        error_rows
    )

    if match_ppda_rows:

        match_ppda_output = pd.concat(
            match_ppda_rows,
            ignore_index=True,
        )

    else:

        match_ppda_output = pd.DataFrame()

    if ppda_15_rows:

        ppda_15_output = pd.concat(
            ppda_15_rows,
            ignore_index=True,
        )

    else:

        ppda_15_output = pd.DataFrame()

    if experiment_rows:

        experiments_output = pd.concat(
            experiment_rows,
            ignore_index=True,
        )

    else:

        experiments_output = pd.DataFrame()

    summary_path = (
        output_dir
        / "match_summary.csv"
    )

    match_ppda_path = (
        output_dir
        / "match_ppda.csv"
    )

    ppda_15_path = (
        output_dir
        / "ppda_15min.csv"
    )

    experiments_path = (
        output_dir
        / "natural_experiments.csv"
    )

    errors_path = (
        output_dir
        / "processing_errors.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    match_ppda_output.to_csv(
        match_ppda_path,
        index=False,
    )

    ppda_15_output.to_csv(
        ppda_15_path,
        index=False,
    )

    experiments_output.to_csv(
        experiments_path,
        index=False,
    )

    errors.to_csv(
        errors_path,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------------
    # Final report.
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("DATASET PROCESSING COMPLETE")
    print("=" * 80)

    print(
        f"Competition/season pairs: "
        f"{len(competition_pairs)}"
    )

    print(
        f"Games discovered:          "
        f"{len(inventory)}"
    )

    print(
        f"Games processed:           "
        f"{len(match_summary_rows)}"
    )

    print(
        f"Games failed:              "
        f"{len(error_rows)}"
    )

    print(
        f"Raw events processed:      "
        f"{total_events}"
    )

    print(
        f"Canonical events:          "
        f"{total_canonical_events}"
    )

    print(
        f"Tactical shifts:           "
        f"{total_tactical_shifts}"
    )

    print(
        f"Halftime shifts:           "
        f"{total_halftime_shifts}"
    )

    print(
        f"Formation changes:         "
        f"{total_formation_changes}"
    )

    print(
        f"Eligible experiments:      "
        f"{total_eligible_changes}"
    )

    print(
        f"Natural experiment rows:   "
        f"{len(experiments_output)}"
    )

    print(
        f"Processing time:            "
        f"{elapsed:.2f} seconds"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 80)

    print(summary_path)
    print(match_ppda_path)
    print(ppda_15_path)
    print(experiments_path)
    print(errors_path)

    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )