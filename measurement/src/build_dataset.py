from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from data.statsbomb_loader import StatsBombDataLoader
from data.preprocessing import EventPreprocessor
from analysis.formation import build_formation_timeline
from analysis.ppda import compute_team_match_ppda, compute_15_minute_ppda
from analysis.natural_experiments import extract_natural_experiments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed-data"
CANONICAL_EVENTS_DIRNAME = "events"

# Event-level canonical CSVs are the actual processed event dataset.
# Set to False only if disk space is insufficient.
SAVE_CANONICAL_EVENTS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the complete SportsM5 processed dataset from local "
            "StatsBomb Open Data."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where processed-data files are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N games with event files. "
            "Use this for a test run. Default: process all available games."
        ),
    )
    parser.add_argument(
        "--no-canonical-events",
        action="store_true",
        help="Do not save event-level canonical CSVs under processed-data/events/.",
    )
    parser.add_argument(
        "--keep-existing-events",
        action="store_true",
        help=(
            "Do not clear processed-data/events/ before a new build. "
            "Normally the canonical event directory is rebuilt from scratch."
        ),
    )
    return parser.parse_args()


def _append_csv(
    frame: pd.DataFrame,
    path: Path,
    expected_columns: list[str] | None = None,
) -> None:
    """Append a DataFrame to CSV, writing the header only once."""
    if frame is None or frame.empty:
        return

    frame = frame.copy()

    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {missing}"
            )

    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def _prepare_inventory(
    loader: StatsBombDataLoader,
) -> tuple[pd.DataFrame, int]:
    """Discover every local competition/season/game before processing."""
    competitions = loader.get_competitions()

    if competitions.empty:
        raise RuntimeError("No competitions/seasons were found.")

    required = {"competition_id", "season_id"}
    missing = required - set(competitions.columns)
    if missing:
        raise ValueError(
            "Competition dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    pairs = (
        competitions[["competition_id", "season_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    inventory_rows: list[dict] = []

    for pair in pairs.itertuples(index=False):
        competition_id = int(pair.competition_id)
        season_id = int(pair.season_id)

        games = loader.get_games(
            competition_id=competition_id,
            season_id=season_id,
        )

        if games.empty:
            continue

        for game in games.to_dict(orient="records"):
            game_id = int(game["game_id"])

            inventory_rows.append(
                {
                    **game,
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "event_file_exists": bool(
                        loader.event_file_exists(game_id)
                    ),
                    "lineup_file_exists": bool(
                        loader.lineup_file_exists(game_id)
                    ),
                }
            )

    inventory = pd.DataFrame(inventory_rows)

    if inventory.empty:
        raise RuntimeError(
            "No games were discovered in the local StatsBomb data."
        )

    if inventory["game_id"].duplicated().any():
        duplicates = (
            inventory.loc[
                inventory["game_id"].duplicated(keep=False),
                "game_id",
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Duplicate game IDs were discovered in the inventory: "
            f"{duplicates[:20]}"
        )

    return inventory, len(pairs)


def _validate_canonical(
    raw_events: pd.DataFrame,
    canonical: pd.DataFrame,
    game_id: int,
) -> None:
    if canonical.empty:
        raise ValueError(f"Canonical event data is empty for game {game_id}.")

    if len(raw_events) != len(canonical):
        raise ValueError(
            "Canonical preprocessing changed event count: "
            f"{len(raw_events)} -> {len(canonical)}"
        )

    required = {
        "game_id",
        "event_id",
        "index",
        "period_id",
        "event_seconds",
        "team_id",
        "team",
        "event_type",
        "x",
        "y",
        "is_tactical_shift",
        "is_halftime_shift",
    }
    missing = required - set(canonical.columns)
    if missing:
        raise ValueError(
            "Canonical dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if not canonical["event_id"].is_unique:
        raise ValueError(
            f"Canonical event_id values are not unique for game {game_id}."
        )

    if not canonical["index"].is_unique:
        raise ValueError(
            f"Canonical StatsBomb indices are not unique for game {game_id}."
        )

    game_ids = pd.to_numeric(
        canonical["game_id"],
        errors="coerce",
    ).dropna()

    if not game_ids.eq(game_id).all():
        raise ValueError(
            f"Canonical events contain an unexpected game_id for {game_id}."
        )


def _add_metadata(
    frame: pd.DataFrame,
    *,
    competition_id: int,
    season_id: int,
    game_id: int,
) -> pd.DataFrame:
    """Attach stable identifiers to a derived analysis frame."""
    if frame is None:
        return pd.DataFrame()

    result = frame.copy()
    result["competition_id"] = int(competition_id)
    result["season_id"] = int(season_id)

    # Match-level PPDA does not return game_id; other analysis outputs may.
    if "game_id" not in result.columns:
        result["game_id"] = int(game_id)
    else:
        result["game_id"] = (
            pd.to_numeric(result["game_id"], errors="coerce")
            .fillna(game_id)
            .astype(int)
        )

    first = ["competition_id", "season_id", "game_id"]
    if "team_id" in result.columns:
        first.append("team_id")
    rest = [c for c in result.columns if c not in first]
    return result[first + rest]


def _enrich_natural_experiments(
    experiments: pd.DataFrame,
    canonical: pd.DataFrame,
    formation_timeline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the event identifiers required by the current DiD input contract.

    The current natural-experiment extractor supplies the five-minute PPDA,
    score state and opponent-response fields, but its output does not always
    include change_event_id, period_id, or is_halftime_shift. These values are
    deterministic properties of the already-identified formation-change event,
    so they are recovered from the canonical/formation records here.
    """
    if experiments is None:
        return pd.DataFrame()

    result = experiments.copy()

    required_defaults = {
        "change_event_id": pd.Series(pd.NA, index=result.index, dtype="object"),
        "period_id": pd.Series(pd.NA, index=result.index, dtype="object"),
        "is_halftime_shift": pd.Series(pd.NA, index=result.index, dtype="object"),
    }
    for column, values in required_defaults.items():
        if column not in result.columns:
            result[column] = values

    if result.empty:
        result["is_halftime_shift"] = result["is_halftime_shift"].fillna(False).astype(bool)
        return result

    # Identify the exact formation-change row using game/team/time.
    timeline = formation_timeline.copy()
    if {"game_id", "team_id", "event_seconds", "event_id"}.issubset(timeline.columns):
        timeline["_change_time_key"] = pd.to_numeric(
            timeline["event_seconds"], errors="coerce"
        ).round(6)
        timeline = timeline.drop_duplicates(
            ["game_id", "team_id", "_change_time_key"], keep="first"
        )

        result["_change_time_key"] = pd.to_numeric(
            result["change_time"], errors="coerce"
        ).round(6)

        lookup = timeline[
            ["game_id", "team_id", "_change_time_key", "event_id"]
        ].rename(columns={"event_id": "_derived_change_event_id"})

        result = result.merge(
            lookup,
            on=["game_id", "team_id", "_change_time_key"],
            how="left",
            validate="many_to_one",
        )

        result["change_event_id"] = result["change_event_id"].fillna(
            result["_derived_change_event_id"]
        )
        result.drop(columns=["_derived_change_event_id"], inplace=True)

    # Recover period and halftime flag from canonical event metadata.
    if {"game_id", "event_id", "period_id", "is_halftime_shift"}.issubset(canonical.columns):
        event_meta = canonical[
            ["game_id", "event_id", "period_id", "is_halftime_shift"]
        ].drop_duplicates(["game_id", "event_id"], keep="first")

        event_meta = event_meta.rename(
            columns={
                "event_id": "_meta_event_id",
                "period_id": "_meta_period_id",
                "is_halftime_shift": "_meta_halftime",
            }
        )

        result = result.merge(
            event_meta,
            left_on=["game_id", "change_event_id"],
            right_on=["game_id", "_meta_event_id"],
            how="left",
            validate="many_to_one",
        )

        result["period_id"] = result["period_id"].fillna(
            result["_meta_period_id"]
        )
        result["is_halftime_shift"] = result["is_halftime_shift"].fillna(
            result["_meta_halftime"]
        )

        result.drop(
            columns=["_meta_event_id", "_meta_period_id", "_meta_halftime"],
            inplace=True,
        )

    result["is_halftime_shift"] = (
        result["is_halftime_shift"].fillna(False).astype(bool)
    )
    result.drop(columns=["_change_time_key"], errors="ignore", inplace=True)
    return result


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_canonical_events = (
        SAVE_CANONICAL_EVENTS and not args.no_canonical_events
    )
    events_output_dir = output_dir / CANONICAL_EVENTS_DIRNAME

    if save_canonical_events:
        if events_output_dir.exists() and not args.keep_existing_events:
            shutil.rmtree(events_output_dir)
        events_output_dir.mkdir(parents=True, exist_ok=True)

    loader = StatsBombDataLoader()

    print()
    print("=" * 80)
    print("SPORTSM5 FULL PROCESSED-DATASET BUILD")
    print("=" * 80)

    inventory, competition_pair_count = _prepare_inventory(loader)

    inventory_path = output_dir / "dataset_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    games_to_process = (
        inventory[inventory["event_file_exists"]]
        .sort_values(
            ["competition_id", "season_id", "game_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero.")
        games_to_process = games_to_process.iloc[: args.limit].copy()

    # These are the dataset-builder outputs. Existing DiD/style/diagnostic
    # outputs are intentionally not deleted.
    summary_path = output_dir / "match_summary.csv"
    match_ppda_path = output_dir / "match_ppda.csv"
    ppda_15_path = output_dir / "ppda_15min.csv"
    formation_path = output_dir / "formation_timeline.csv"
    experiments_path = output_dir / "natural_experiments.csv"
    errors_path = output_dir / "processing_errors.csv"

    for path in (
        summary_path,
        match_ppda_path,
        ppda_15_path,
        formation_path,
        experiments_path,
        errors_path,
    ):
        if path.exists():
            path.unlink()

    summary_rows: list[dict] = []
    error_rows: list[dict] = []

    total_events = 0
    total_canonical_events = 0
    total_tactical_shifts = 0
    total_halftime_shifts = 0
    total_formation_changes = 0
    total_experiments = 0

    start_time = time.perf_counter()

    print(f"Competition/season pairs found: {competition_pair_count}")
    print(f"Games discovered in inventory:  {len(inventory)}")
    print(
        "Games with local event files:    "
        f"{int(inventory['event_file_exists'].sum())}"
    )
    print(f"Games selected for processing:  {len(games_to_process)}")
    print(f"Canonical event files saved:    {save_canonical_events}")
    print()

    for position, game in enumerate(
        games_to_process.to_dict(orient="records"),
        start=1,
    ):
        game_id = int(game["game_id"])
        competition_id = int(game["competition_id"])
        season_id = int(game["season_id"])

        print(
            f"[{position}/{len(games_to_process)}] "
            f"Processing game {game_id}..."
        )

        try:
            if "home_team_id" not in game:
                raise ValueError(
                    "Game record does not contain 'home_team_id'."
                )

            home_team_id = int(game["home_team_id"])

            events = loader.get_events(game_id)

            if events.empty:
                raise ValueError("Event file contains no events.")

            canonical = EventPreprocessor(
                events,
                home_team_id=home_team_id,
            ).transform()

            _validate_canonical(
                events,
                canonical,
                game_id,
            )

            formation_timeline = build_formation_timeline(canonical)
            match_ppda = compute_team_match_ppda(canonical)
            ppda_15 = compute_15_minute_ppda(canonical)
            experiments = extract_natural_experiments(
                canonical,
                formation_timeline,
            )

            match_ppda = _add_metadata(
                match_ppda,
                competition_id=competition_id,
                season_id=season_id,
                game_id=game_id,
            )
            ppda_15 = _add_metadata(
                ppda_15,
                competition_id=competition_id,
                season_id=season_id,
                game_id=game_id,
            )

            # Defensive: guarantee game_id exists on every per-match row,
            # even if an upstream change ever bypasses _add_metadata.
            match_ppda["game_id"] = game_id
            ppda_15["game_id"] = game_id
            
            formation_timeline = _add_metadata(
                formation_timeline,
                competition_id=competition_id,
                season_id=season_id,
                game_id=game_id,
            )
            experiments = _add_metadata(
                experiments,
                competition_id=competition_id,
                season_id=season_id,
                game_id=game_id,
            )

            experiments = _enrich_natural_experiments(
                experiments,
                canonical,
                formation_timeline,
            )

            # Save event-level canonical data. This is the reusable
            # event-level processed dataset for downstream modules.
            if save_canonical_events:
                canonical_path = events_output_dir / f"{game_id}.csv"
                canonical.to_csv(
                    canonical_path,
                    index=False,
                )
            _append_csv(match_ppda, match_ppda_path)
            _append_csv(ppda_15, ppda_15_path)
            _append_csv(formation_timeline, formation_path)
            _append_csv(experiments, experiments_path)

            tactical_shift_count = int(
                canonical["is_tactical_shift"].sum()
            )
            halftime_shift_count = int(
                canonical["is_halftime_shift"].sum()
            )
            formation_change_count = int(
                formation_timeline["is_formation_change"].sum()
            )

            summary_rows.append(
                {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "game_id": game_id,
                    "home_team_id": home_team_id,
                    "raw_events": len(events),
                    "canonical_events": len(canonical),
                    "lineup_file_exists": bool(
                        game["lineup_file_exists"]
                    ),
                    "tactical_shifts": tactical_shift_count,
                    "halftime_shifts": halftime_shift_count,
                    "formation_changes": formation_change_count,
                    "natural_experiment_rows": len(experiments),
                }
            )

            total_events += len(events)
            total_canonical_events += len(canonical)
            total_tactical_shifts += tactical_shift_count
            total_halftime_shifts += halftime_shift_count
            total_formation_changes += formation_change_count
            total_experiments += len(experiments)

        except Exception as exc:
            error_rows.append(
                {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "game_id": game_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(
                f"  ERROR: {type(exc).__name__}: {exc}"
            )

    summary = pd.DataFrame(summary_rows)
    errors = pd.DataFrame(
        error_rows,
        columns=[
            "competition_id",
            "season_id",
            "game_id",
            "error_type",
            "error_message",
        ],
    )

    summary.to_csv(summary_path, index=False)
    errors.to_csv(errors_path, index=False)

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 80)
    print("DATASET BUILD COMPLETE")
    print("=" * 80)
    print(f"Competition/season pairs:    {competition_pair_count}")
    print(f"Games discovered:             {len(inventory)}")
    print(f"Games selected:               {len(games_to_process)}")
    print(f"Games processed successfully: {len(summary)}")
    print(f"Games failed:                 {len(errors)}")
    print(f"Raw events processed:         {total_events}")
    print(f"Canonical events:             {total_canonical_events}")
    print(f"Tactical shifts:              {total_tactical_shifts}")
    print(f"Halftime shifts:              {total_halftime_shifts}")
    print(f"Formation changes:            {total_formation_changes}")
    print(f"Natural experiment rows:      {total_experiments}")
    print(f"Processing time:              {elapsed:.2f} seconds")
    print()
    print("PRIMARY PROCESSED DATA")
    print("-" * 80)
    print(f"Inventory:                    {inventory_path}")
    print(
        "Canonical events directory:   "
        f"{events_output_dir if save_canonical_events else 'disabled'}"
    )
    print(f"Match summary:                {summary_path}")
    print(f"Match PPDA:                   {match_ppda_path}")
    print(f"15-minute PPDA:               {ppda_15_path}")
    print(f"Formation timeline:           {formation_path}")
    print(f"Natural experiments:          {experiments_path}")
    print(f"Processing errors:            {errors_path}")
    print("=" * 80)

    # IMPORTANT:
    # `errors` is a pandas DataFrame, so `if errors:` raises:
    # ValueError: The truth value of a DataFrame is ambiguous.
    # Use `.empty` to test whether any processing errors were recorded.
    if not errors.empty:
        print()
        print(
            "BUILD STATUS: FAILED "
            f"({len(errors)} game(s) could not be processed)."
        )

        # Show a compact diagnostic summary so a large failed build
        # does not hide the actual root cause in thousands of lines.
        print()
        print("PROCESSING ERROR SUMMARY")
        print("-" * 80)

        error_counts = (
            errors.groupby(
                ["error_type", "error_message"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        for row in error_counts.head(10).itertuples(index=False):
            print(
                f"{row.count:>6}  "
                f"{row.error_type}: {row.error_message}"
            )

        print()
        print(f"Full error report: {errors_path}")
        return 1

    print()
    print("BUILD STATUS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
