from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "dataset_inventory.csv",
    "match_summary.csv",
    "match_ppda.csv",
    "ppda_15min.csv",
    "natural_experiments.csv",
    "processing_errors.csv",
)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def audit_processed_data(processed_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    processed_dir = Path(processed_dir).expanduser().resolve()

    data = {name: _read_csv(processed_dir / name) for name in REQUIRED_FILES[:-1]}
    errors_path = processed_dir / "processing_errors.csv"
    errors_text = errors_path.read_text(encoding="utf-8") if errors_path.exists() else ""

    inv = data["dataset_inventory.csv"]
    summary = data["match_summary.csv"]
    match_ppda = data["match_ppda.csv"]
    ppda15 = data["ppda_15min.csv"]
    experiments = data["natural_experiments.csv"]

    checks = []

    def add(name: str, passed: bool, value: object, expected: object = None, severity: str = "ERROR"):
        checks.append({
            "check": name,
            "status": "PASS" if passed else severity,
            "value": value,
            "expected": expected,
        })

    # Inventory integrity.
    add(
        "inventory_game_ids_unique",
        inv["game_id"].is_unique,
        int(inv["game_id"].duplicated().sum()),
        0,
    )
    add(
        "inventory_event_files_complete",
        bool(inv["event_file_exists"].all()),
        int((~inv["event_file_exists"]).sum()),
        0,
    )
    add(
        "inventory_lineup_files_complete",
        bool(inv["lineup_file_exists"].all()),
        int((~inv["lineup_file_exists"]).sum()),
        0,
    )
    add(
        "competition_season_pairs",
        inv[["competition_id", "season_id"]].drop_duplicates().shape[0],
        inv[["competition_id", "season_id"]].drop_duplicates().shape[0],
        80,
        severity="INFO",
    )

    # Match summary integrity.
    add(
        "summary_game_count_matches_inventory",
        summary["game_id"].nunique() == inv["game_id"].nunique(),
        summary["game_id"].nunique(),
        inv["game_id"].nunique(),
    )
    add(
        "summary_game_ids_unique",
        summary["game_id"].is_unique,
        int(summary["game_id"].duplicated().sum()),
        0,
    )
    add(
        "canonical_event_count_preserved_all_games",
        bool((summary["raw_events"] == summary["canonical_events"]).all()),
        int((summary["raw_events"] != summary["canonical_events"]).sum()),
        0,
    )
    add(
        "processing_errors_empty",
        errors_text.strip() == "",
        "empty" if errors_text.strip() == "" else errors_text[:500],
        "empty",
    )

    # PPDA output shape.
    expected_match_ppda = len(inv) * 2
    add(
        "match_ppda_row_count",
        len(match_ppda) == expected_match_ppda,
        len(match_ppda),
        expected_match_ppda,
    )
    match_counts = (
        match_ppda.groupby(["competition_id", "season_id"]).size()
        .rename("rows")
        .reset_index()
    )
    inventory_pair_counts = (
        inv.groupby(["competition_id", "season_id"]).size()
        .mul(2)
        .rename("expected_rows")
        .reset_index()
    )
    pair_check = match_counts.merge(
        inventory_pair_counts,
        on=["competition_id", "season_id"],
        how="outer",
    ).fillna(0)
    add(
        "match_ppda_two_rows_per_match_by_pair",
        bool((pair_check["rows"] == pair_check["expected_rows"]).all()),
        int((pair_check["rows"] != pair_check["expected_rows"]).sum()),
        0,
    )

    expected_15 = len(inv) * 2 * 6
    add(
        "ppda_15min_row_count",
        len(ppda15) == expected_15,
        len(ppda15),
        expected_15,
    )
    add(
        "ppda_15min_keys_unique",
        ppda15[["game_id", "team_id", "window_start_minute"]].duplicated().sum() == 0,
        int(ppda15[["game_id", "team_id", "window_start_minute"]].duplicated().sum()),
        0,
    )

    # Missing PPDA is not a processing failure: it means zero defensive actions.
    match_ppda_missing = int(match_ppda["ppda"].isna().sum())
    p15_missing = int(ppda15["ppda"].isna().sum())
    add(
        "match_ppda_missing_values",
        match_ppda_missing == 0,
        match_ppda_missing,
        0,
        severity="WARN",
    )
    add(
        "ppda_15min_missing_values",
        p15_missing == 0,
        p15_missing,
        0,
        severity="WARN",
    )

    # Natural experiment integrity.
    add(
        "natural_experiment_count_matches_summary",
        len(experiments) == int(summary["natural_experiment_rows"].sum()),
        len(experiments),
        int(summary["natural_experiment_rows"].sum()),
    )
    add(
        "natural_experiment_ids_unique",
        experiments["change_event_id"].is_unique,
        int(experiments["change_event_id"].duplicated().sum()),
        0,
        severity="WARN",
    )
    add(
        "natural_experiment_eligible_count",
        int(summary["eligible_natural_experiments"].sum()) == len(experiments),
        int(summary["eligible_natural_experiments"].sum()),
        len(experiments),
    )

    complete_ne = experiments[
        experiments["before_ppda"].notna() & experiments["after_ppda"].notna()
    ].copy()
    add(
        "natural_experiments_with_complete_ppda",
        len(complete_ne) == len(experiments),
        len(complete_ne),
        len(experiments),
        severity="WARN",
    )

    # Halftime and period-boundary diagnostics.
    halftime_count = int(experiments["is_halftime_shift"].sum())
    add(
        "natural_experiment_halftime_shifts_flagged",
        "is_halftime_shift" in experiments.columns,
        halftime_count,
        "column present",
    )

    experiments["change_time"] = pd.to_numeric(
        experiments["change_time"], errors="coerce"
    )
    experiments["period_id"] = pd.to_numeric(
        experiments["period_id"], errors="coerce"
    )
    cross_halftime = (
        ((experiments["change_time"] > 2400) & (experiments["change_time"] < 3000))
        | experiments["is_halftime_shift"].astype(bool)
    )
    extra_time = experiments["period_id"].isin([3, 4])
    add(
        "natural_experiments_near_halftime_or_crossing_boundary",
        int(cross_halftime.sum()) == 0,
        int(cross_halftime.sum()),
        0,
        severity="WARN",
    )
    add(
        "natural_experiments_in_extra_time",
        int(extra_time.sum()) == 0,
        int(extra_time.sum()),
        0,
        severity="WARN",
    )

    # Window sanity.
    if {"before_start", "before_end", "after_start", "after_end"}.issubset(experiments.columns):
        window_ok = (
            np.isclose(
                experiments["before_end"].to_numpy(dtype=float),
                experiments["after_start"].to_numpy(dtype=float),
                atol=1e-6,
                equal_nan=False,
            )
            & np.isclose(
                (
                    experiments["before_end"] - experiments["before_start"]
                ).to_numpy(dtype=float),
                300.0,
                atol=1e-6,
                equal_nan=False,
            )
            & np.isclose(
                (
                    experiments["after_end"] - experiments["after_start"]
                ).to_numpy(dtype=float),
                300.0,
                atol=1e-6,
                equal_nan=False,
            )
        )
        add(
            "natural_experiment_five_minute_windows",
            bool(window_ok.all()),
            int((~window_ok).sum()),
            0,
        )

    # Summary totals versus expected relationships.
    add(
        "eligible_not_greater_than_actual_changes",
        bool((summary["eligible_natural_experiments"] <= summary["formation_changes"]).all()),
        int((summary["eligible_natural_experiments"] > summary["formation_changes"]).sum()),
        0,
    )
    add(
        "halftime_shifts_not_greater_than_tactical_shifts",
        bool((summary["halftime_shifts"] <= summary["tactical_shifts"]).all()),
        int((summary["halftime_shifts"] > summary["tactical_shifts"]).sum()),
        0,
    )

    report = pd.DataFrame(checks)

    counts = {
        "games": int(len(inv)),
        "competition_season_pairs": int(inv[["competition_id", "season_id"]].drop_duplicates().shape[0]),
        "raw_events": int(summary["raw_events"].sum()),
        "canonical_events": int(summary["canonical_events"].sum()),
        "tactical_shifts": int(summary["tactical_shifts"].sum()),
        "halftime_shifts": int(summary["halftime_shifts"].sum()),
        "formation_changes": int(summary["formation_changes"].sum()),
        "eligible_natural_experiments": int(summary["eligible_natural_experiments"].sum()),
        "natural_experiment_rows": int(len(experiments)),
        "natural_experiments_complete_ppda": int(len(complete_ne)),
        "natural_experiments_missing_ppda": int(len(experiments) - len(complete_ne)),
        "natural_experiments_halftime": halftime_count,
        "natural_experiments_extra_time": int(extra_time.sum()),
        "match_ppda_rows": int(len(match_ppda)),
        "ppda_15min_rows": int(len(ppda15)),
        "ppda_15min_missing": p15_missing,
        "processing_errors_bytes": len(errors_text.encode("utf-8")),
    }

    summary_json = {
        "status": (
            "PASS"
            if not report["status"].eq("ERROR").any()
            else "FAIL"
        ),
        "counts": counts,
        "checks": report.to_dict(orient="records"),
    }

    return report, summary_json


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit the processed SportsM5 dataset."
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
    args = parser.parse_args()

    processed_dir = args.processed_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else processed_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report, summary = audit_processed_data(processed_dir)

    report_path = output_dir / "audit_checks.csv"
    summary_path = output_dir / "audit_summary.json"
    report.to_csv(report_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("SPORTSM5 PROCESSED-DATA AUDIT")
    print("=" * 80)
    print(f"Status:                         {summary['status']}")
    for key, value in summary["counts"].items():
        print(f"{key + ':':32s} {value}")
    print()
    print(report.to_string(index=False))
    print()
    print(f"Audit checks: {report_path}")
    print(f"Audit summary: {summary_path}")

    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
