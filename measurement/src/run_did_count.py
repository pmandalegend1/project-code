"""
DiD estimated as a count model with an exposure offset.

    python src/run_did_count.py --processed-data processed-data \
                                --output-dir processed-data

CPU only. No GPU, no PyTorch — pandas + statsmodels. Runs in minutes.

Why this exists
---------------
The ratio-based DiD throws away every five-minute window in which the
opponent recorded zero defensive actions: PPDA = passes / actions is
undefined there. That was 1,075 of 4,712 windows (22.81%), and it forced
930 of 1,617 matched pairs out of the primary regression — with treatment
borderline-predictive of that missingness (p = 0.089).

Those windows are not missing information. Zero defensive actions against
forty opponent passes is the strongest available evidence of NOT pressing.
Modelling the counts directly, with opponent passes as the exposure, lets
every one of them back in as a legitimate y = 0.

Nothing is imputed. PPDA's NaN is never filled with a number — the outcome
variable is simply a different, better-behaved quantity that measures the
same thing.

Outputs
-------
did_count_results.csv      negative binomial + Poisson, primary specification
did_count_robustness.csv   the same robustness grid as the ratio analysis
did_count_vs_ratio.csv     side-by-side comparison for the paper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.did import (  # noqa: E402
    DiDConfig,
    build_did_dataset,
    run_did_count_regression,
    run_did_regression,
)


SPECIFICATIONS: dict[str, DiDConfig] = {
    "primary": DiDConfig(),
    "window_3_minutes": DiDConfig(window_minutes=3),
    "minimum_10_minute_gap": DiDConfig(
        first_change_per_team_match=False,
        minimum_change_gap_minutes=10,
    ),
    "exclude_opponent_responses": DiDConfig(exclude_opponent_responses=True),
    "halftime_only": DiDConfig(
        exclude_halftime=False,
        only_halftime=True,
    ),
}


def _report(name: str, row: pd.Series) -> None:
    if row.get("status") != "estimated":
        print(f"  {name:<28} {row.get('status')}")
        return
    rr = row["rate_ratio"]
    print(
        f"  {name:<28} log-RR {row['estimate_beta3_log_rr']:+.4f}  "
        f"SE {row['std_error']:.4f}  p {row['p_value']:.4f}  "
        f"rate ratio {rr:.3f} "
        f"[{row['rate_ratio_low_95']:.3f}, {row['rate_ratio_high_95']:.3f}]  "
        f"n {int(row['n_rows_used'])}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-data", default="processed-data")
    ap.add_argument("--output-dir", default="processed-data")
    ap.add_argument(
        "--skip-robustness",
        action="store_true",
        help="primary specification only",
    )
    args = ap.parse_args()

    processed = Path(args.processed_data).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DiD — count model with exposure offset")
    print("=" * 78)

    # ---------------- primary ----------------
    pairs, long, construction, missingness, long_all = build_did_dataset(
        processed_dir=processed,
        config=SPECIFICATIONS["primary"],
    )

    print(f"\nmatched pairs                    {len(pairs)}")
    print(f"complete-case long rows (ratio)  {len(long)}")
    print(f"all long rows (count model)      {len(long_all)}")
    gained = len(long_all) - len(long)
    print(f"rows recovered by the count model {gained}  "
          f"({gained / max(len(long_all), 1):.1%} of the sample)")

    print("\nprimary specification:")
    primary = run_did_count_regression(long_all)
    for _, row in primary.iterrows():
        _report(row.get("family", "?"), row)
    primary.insert(0, "analysis", "primary")
    primary.to_csv(out / "did_count_results.csv", index=False)
    print(f"\n  wrote {out / 'did_count_results.csv'}")

    # ------------- ratio vs count, side by side -------------
    _, ratio_result = run_did_regression(long, require_style_reference=False)
    nb = primary[primary["family"] == "negative_binomial"]
    comparison = pd.DataFrame([
        {
            "estimator": "ratio DiD (OLS on PPDA)",
            "effect": float(ratio_result["estimate_beta3"].iloc[0]),
            "effect_scale": "PPDA units",
            "std_error": float(ratio_result["std_error"].iloc[0]),
            "p_value": float(ratio_result["p_value"].iloc[0]),
            "n_rows": int(ratio_result["n_long_rows"].iloc[0]),
            "windows_discarded_as_undefined": gained,
        },
        *(
            [{
                "estimator": "count DiD (NB, offset=log passes)",
                "effect": float(nb["estimate_beta3_log_rr"].iloc[0]),
                "effect_scale": "log rate ratio, actions per pass",
                "std_error": float(nb["std_error"].iloc[0]),
                "p_value": float(nb["p_value"].iloc[0]),
                "n_rows": int(nb["n_rows_used"].iloc[0]),
                "windows_discarded_as_undefined": 0,
            }]
            if not nb.empty and nb["status"].iloc[0] == "estimated"
            else []
        ),
    ])
    comparison.to_csv(out / "did_count_vs_ratio.csv", index=False)
    print(f"  wrote {out / 'did_count_vs_ratio.csv'}")

    if args.skip_robustness:
        return 0

    # ---------------- robustness ----------------
    print("\nrobustness specifications:")
    rows: list[pd.DataFrame] = []
    for name, config in SPECIFICATIONS.items():
        if name == "primary":
            continue
        try:
            *_, spec_long_all = build_did_dataset(
                processed_dir=processed,
                config=config,
            )
            result = run_did_count_regression(spec_long_all)
        except Exception as error:  # noqa: BLE001 - reported, not hidden
            print(f"  {name:<28} failed: {type(error).__name__}")
            rows.append(pd.DataFrame([{
                "analysis": name,
                "status": f"failed: {type(error).__name__}",
                "error": str(error),
            }]))
            continue
        result.insert(0, "analysis", name)
        nb_row = result[result["family"] == "negative_binomial"]
        if not nb_row.empty:
            _report(name, nb_row.iloc[0])
        rows.append(result)

    robustness = pd.concat(rows, ignore_index=True)
    robustness.to_csv(out / "did_count_robustness.csv", index=False)
    print(f"\n  wrote {out / 'did_count_robustness.csv'}")

    # ---------------- how to read it ----------------
    print("\n" + "=" * 78)
    print("READING THE RESULT")
    print("=" * 78)
    print("""
beta3 is a LOG RATE RATIO of the opponent's defensive actions per opponent
pass. exp(beta3) is the multiplicative change after a formation change,
relative to matched controls.

  rate ratio > 1  ->  MORE defensive actions per pass  ->  opponent pressed
                      HARDER  ->  PPDA fell
  rate ratio < 1  ->  opponent pressed LESS  ->  PPDA rose

State the direction explicitly in the paper -- "PPDA increased" alone
will be misread, because a HIGHER PPDA means LESS pressing.

Compare this estimate against the ratio DiD in did_count_vs_ratio.csv,
which is written from the run that just happened.  (This paragraph used to
quote a beta3 of +0.1602 from an earlier run, hard-coded into the text; it
was left behind when the measurement fixes changed the estimate, and it
then told the reader the opposite of what the run had just produced.  Never
restate a result here -- point at the file that carries the live one.)

If the result is still not significant, report it with the power statement,
not as an absence of effect: the minimum detectable effect at 80% power is
2.8016 x SE (column mde_80_power_log_rr). Say "underpowered for effects of
this size", never "there is no effect".
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
