from __future__ import annotations

from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "processed-data"
)

INPUT_PATH = (
    PROCESSED_DATA_DIR
    / "natural_experiments.csv"
)


# =============================================================================
# EXPECTED CORE COLUMNS
# =============================================================================

EXPECTED_COLUMNS = [
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
]


# =============================================================================
# LOAD
# =============================================================================

def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Natural-experiment file not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        raise ValueError(
            "natural_experiments.csv is empty."
        )

    return df


# =============================================================================
# COLUMN CHECK
# =============================================================================

def check_columns(df: pd.DataFrame) -> None:
    actual = set(df.columns)

    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual
    ]

    print("=" * 80)
    print("SPORTSM5 DiD DATA CHECK")
    print("=" * 80)

    print(
        f"Input rows:             {len(df)}"
    )

    print(
        f"Input columns:          {len(df.columns)}"
    )

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}")

    if missing:
        raise ValueError(
            "\nMissing expected columns:\n"
            + "\n".join(
                f"  {column}"
                for column in missing
            )
        )

    print(
        "\nRequired natural-experiment "
        "columns: PASS"
    )


# =============================================================================
# IDENTIFIER CHECKS
# =============================================================================

def check_identifiers(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("IDENTIFIER CHECKS")
    print("=" * 80)

    print(
        f"Unique games:           "
        f"{df['game_id'].nunique()}"
    )

    print(
        f"Unique treated teams:    "
        f"{df['team_id'].nunique()}"
    )

    print(
        f"Unique change events:    "
        f"{df['change_event_id'].nunique()}"
    )

    duplicate_changes = (
        df["change_event_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate change IDs:    "
        f"{duplicate_changes}"
    )

    if duplicate_changes:
        print(
            "WARNING: change_event_id is not "
            "unique."
        )
    else:
        print(
            "Change-event uniqueness: PASS"
        )


# =============================================================================
# WINDOW CHECKS
# =============================================================================

def check_windows(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("WINDOW CHECKS")
    print("=" * 80)

    numeric_columns = [
        "change_time",
        "before_start",
        "before_end",
        "after_start",
        "after_end",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(
        "Expected before-window length: "
        "300 seconds"
    )

    print(
        "Expected after-window length:  "
        "300 seconds"
    )

    before_lengths = (
        df["before_end"]
        - df["before_start"]
    )

    after_lengths = (
        df["after_end"]
        - df["after_start"]
    )

    print(
        f"\nBefore-window mean: "
        f"{before_lengths.mean():.3f}"
    )

    print(
        f"After-window mean:  "
        f"{after_lengths.mean():.3f}"
    )

    before_bad = (
        ~before_lengths.between(
            299.999,
            300.001,
        )
    ).sum()

    after_bad = (
        ~after_lengths.between(
            299.999,
            300.001,
        )
    ).sum()

    print(
        f"Invalid before windows: "
        f"{before_bad}"
    )

    print(
        f"Invalid after windows:  "
        f"{after_bad}"
    )


# =============================================================================
# PPDA CHECKS
# =============================================================================

def check_ppda(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("PPDA CHECKS")
    print("=" * 80)

    ppda_columns = [
        "before_ppda",
        "after_ppda",
    ]

    for column in ppda_columns:
        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        print(
            f"\n{column}:"
        )

        print(
            f"  non-null: "
            f"{numeric.notna().sum()}"
        )

        print(
            f"  missing:  "
            f"{numeric.isna().sum()}"
        )

        if numeric.notna().any():
            print(
                f"  mean:     "
                f"{numeric.mean():.6f}"
            )

            print(
                f"  median:   "
                f"{numeric.median():.6f}"
            )

            print(
                f"  min:      "
                f"{numeric.min():.6f}"
            )

            print(
                f"  max:      "
                f"{numeric.max():.6f}"
            )


# =============================================================================
# FORMATION / HALFTIME CHECKS
# =============================================================================

def check_treatment_structure(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("TREATMENT STRUCTURE")
    print("=" * 80)

    print(
        "Formation changes:"
    )

    print(
        df[
            [
                "old_formation",
                "new_formation",
            ]
        ]
        .value_counts()
        .head(20)
        .to_string()
    )

    print(
        "\nPeriod distribution:"
    )

    print(
        df["period_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nHalftime-shift distribution:"
    )

    print(
        df["is_halftime_shift"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\nOpponent response:"
    )

    print(
        df[
            "opponent_response_is_shift"
        ]
        .value_counts(dropna=False)
        .to_string()
    )


# =============================================================================
# SCORE-STATE CHECK
# =============================================================================

def check_score_state(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("SCORE-STATE CHECK")
    print("=" * 80)

    print(
        "Score difference distribution:"
    )

    print(
        df["score_difference"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nScore-state summary:"
    )

    print(
        f"Winning before change: "
        f"{(df['score_difference'] > 0).sum()}"
    )

    print(
        f"Drawing before change: "
        f"{(df['score_difference'] == 0).sum()}"
    )

    print(
        f"Losing before change:  "
        f"{(df['score_difference'] < 0).sum()}"
    )


# =============================================================================
# MISSINGNESS
# =============================================================================

def check_missingness(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("MISSINGNESS")
    print("=" * 80)

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:
        print(
            "No missing values: PASS"
        )
        return

    print(
        missing.to_string()
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    df = load_data()

    check_columns(df)

    check_identifiers(df)

    check_windows(df)

    check_ppda(df)

    check_treatment_structure(df)

    check_score_state(df)

    check_missingness(df)

    print("\n" + "=" * 80)
    print("DiD DATA CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()