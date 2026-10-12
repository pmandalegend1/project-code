"""
Discrimination check for the confound columns emitted by
extract_natural_experiments.

Runs on the full processed-data/natural_experiments.csv rather than on a
single match.  Verifies that the boolean and count flags actually vary
across the corpus -- if they do not, either every treatment is confounded
or the flag is silently zeroed, and either way the robustness grid in
did.py is meaningless.
"""

from pathlib import Path
import pandas as pd


def main() -> int:
    path = Path("processed-data") / "natural_experiments.csv"
    if not path.is_file():
        print(f"missing {path}; run build_dataset.py first")
        return 1

    df = pd.read_csv(path)
    n = len(df)
    if n == 0:
        print("natural_experiments.csv is empty")
        return 1

    print("=" * 72)
    print("CONFOUND PROVENANCE DISCRIMINATION CHECK")
    print("=" * 72)
    print(f"rows: {n}")

    failures = []

    # Boolean flags must be present, well-typed, and vary.
    for col in (
        "is_first_change_for_team_match",
        "other_formation_change_within_isolation_window",
    ):
        if col not in df.columns:
            failures.append(f"missing column: {col}")
            continue
        s = df[col].fillna(False).astype(bool)
        n_true = int(s.sum())
        n_false = n - n_true
        print(f"  {col:<52}  true={n_true:>5}  false={n_false:>5}")
        if n_true == 0:
            failures.append(f"{col} is all False")
        if n_false == 0:
            failures.append(f"{col} is all True")

    # Count flags must vary and be non-negative.
    for col in (
        "substitutions_within_did_window",
        "goals_within_did_window",
    ):
        if col not in df.columns:
            failures.append(f"missing column: {col}")
            continue
        v = pd.to_numeric(df[col], errors="coerce").fillna(0)
        n_nonzero = int((v > 0).sum())
        print(
            f"  {col:<52}  >0={n_nonzero:>5}  max={int(v.max()):>3}"
        )
        if (v < 0).any():
            failures.append(f"{col} went negative")
        if n_nonzero == 0:
            failures.append(f"{col} is always zero")

    # Distance to nearest other change should be finite for a meaningful
    # subset -- if every row is NaN, isolation filtering is a no-op.
    if "seconds_to_nearest_other_change" in df.columns:
        dist = pd.to_numeric(
            df["seconds_to_nearest_other_change"], errors="coerce"
        )
        n_finite = int(dist.notna().sum())
        print(
            f"  {'seconds_to_nearest_other_change':<52}  "
            f"finite={n_finite:>5}"
        )
        if n_finite == 0:
            failures.append("seconds_to_nearest_other_change is all NaN")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS -- all confound flags are present and discriminating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())