"""
Validation of the shipped style clustering.

Reports the four things a reviewer will ask about:

  1. Does a silhouette sweep over the actual feature space support the
     chosen K, or would another K have been better?
  2. Do the clusters cut across formations, or does each formation map to
     exactly one cluster (which would mean style is a re-spelling of
     formation)?
  3. Is the between-cluster spread in PPDA larger than the within-cluster
     spread, i.e. do the clusters actually separate on the metric they
     claim to summarise?
  4. How many teams carry an extrapolated label (fewer than 10 matches)?

The previous version hard-coded K = 4, clustered on features that
included formation counts, and silently produced a partition that no
current pipeline output corresponds to.  It also compared its own
refit against team_style_clusters.csv by raw label agreement, which is
meaningless for KMeans since cluster IDs are permutation-invariant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


PPDA_FEATURES = ["ppda_mean", "ppda_std", "ppda_slope"]
BEHAVIOUR_FEATURES = [
    "pass_length_mean",
    "possession_share",
    "width",
    "passes_per_possession",
    "def_line_height",
]
SWEEP_K = range(2, 8)


def _resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in PPDA_FEATURES if c in df.columns]
    for c in BEHAVIOUR_FEATURES:
        if c in df.columns:
            cols.append(c)
    if len(cols) < 2:
        raise ValueError(
            "At least two PPDA feature columns must be present; "
            f"found {cols}"
        )
    return cols


def _load(processed_dir: Path):
    features_path = processed_dir / "team_style_features.csv"
    clusters_path = processed_dir / "team_style_clusters.csv"
    if not features_path.is_file():
        raise FileNotFoundError(
            f"Missing {features_path}. Run src/run_style_clustering.py first."
        )
    if not clusters_path.is_file():
        raise FileNotFoundError(
            f"Missing {clusters_path}. Run src/run_style_clustering.py first."
        )
    return pd.read_csv(features_path), pd.read_csv(clusters_path)


def _prepare(features: pd.DataFrame, clusters: pd.DataFrame):
    feature_columns = _resolve_feature_columns(features)
    keep_features = ["team_id"] + feature_columns
    if "modal_formation_code" in features.columns:
        keep_features.append("modal_formation_code")

    keep_clusters = ["team_id", "style_cluster"]
    for c in (
        "style_fit_team",
        "style_label_is_extrapolated",
        "ppda_intensity_rank",
        "pressing_regime",
    ):
        if c in clusters.columns:
            keep_clusters.append(c)

    merged = features[keep_features].merge(
        clusters[keep_clusters],
        on="team_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=feature_columns).reset_index(drop=True)

    if "style_fit_team" in merged.columns:
        fit_mask = merged["style_fit_team"].fillna(False).astype(bool)
    else:
        print(
            "[validation] style_fit_team missing; treating every team as fit."
        )
        fit_mask = pd.Series(True, index=merged.index)

    return merged, merged[fit_mask].reset_index(drop=True), feature_columns


def _silhouette_sweep(
    X_scaled: np.ndarray,
    n_max: int,
) -> pd.DataFrame:
    rows = []
    n = X_scaled.shape[0]
    for k in SWEEP_K:
        if k >= n:
            break
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            continue
        rows.append({
            "k": k,
            "silhouette_score": silhouette_score(X_scaled, labels),
            "calinski_harabasz_score": calinski_harabasz_score(
                X_scaled, labels
            ),
            "davies_bouldin_score": davies_bouldin_score(
                X_scaled, labels
            ),
            "inertia": float(model.inertia_),
        })
    return pd.DataFrame(rows)


def _cluster_summary(
    merged: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for cluster_id, group in merged.groupby("style_cluster", sort=True):
        row = {
            "style_cluster": int(cluster_id),
            "team_count": int(len(group)),
        }
        for c in feature_columns:
            row[f"{c}_mean"] = float(group[c].mean())
            row[f"{c}_std"] = float(group[c].std(ddof=1))
        if "style_label_is_extrapolated" in group.columns:
            row["extrapolated_teams"] = int(
                group["style_label_is_extrapolated"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _formation_crosstab(merged: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "modal_formation_code" not in merged.columns:
        return pd.DataFrame(), 0
    ct = pd.crosstab(
        merged["style_cluster"],
        merged["modal_formation_code"],
    )
    # number of formations that appear in more than one cluster
    spanning = int((ct.gt(0).sum(axis=0) > 1).sum())
    return ct, spanning


def _between_within_ppda(merged: pd.DataFrame) -> tuple[float, float]:
    g = merged.groupby("style_cluster")["ppda_mean"]
    between = float(g.mean().max() - g.mean().min())
    within = float(g.std(ddof=1).median())
    return between, within


def _report(
    merged: pd.DataFrame,
    fit: pd.DataFrame,
    feature_columns: list[str],
    sweep: pd.DataFrame,
    summary: pd.DataFrame,
    crosstab: pd.DataFrame,
    spanning: int,
    between: float,
    within: float,
    chosen_k: int,
) -> None:

    print("=" * 80)
    print("SPORTSM5 STYLE-CLUSTER VALIDATION")
    print("=" * 80)
    print(f"Teams in shipped clustering:      {len(merged)}")
    print(f"Teams used to fit the clusterer:  {len(fit)}")
    print(f"Chosen K (from cluster file):     {chosen_k}")
    print(f"Feature columns:                  {feature_columns}")
    if "style_label_is_extrapolated" in merged.columns:
        rate = float(
            merged["style_label_is_extrapolated"]
            .fillna(False)
            .astype(bool)
            .mean()
        )
        print(f"Extrapolated team fraction:       {rate:.1%}")

    print()
    print("SILHOUETTE SWEEP (fit teams only)")
    print("-" * 80)
    if sweep.empty:
        print("(no valid sweep values)")
    else:
        print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        best_k = int(sweep.loc[sweep["silhouette_score"].idxmax(), "k"])
        print()
        print(f"  Best silhouette K:              {best_k}")
        print(f"  Shipped K:                      {chosen_k}")
        if best_k != chosen_k:
            print(
                f"  NOTE: shipped K does not match the best silhouette K.  "
                f"Check that style_clustering.py ran after the last change "
                f"to the feature set."
            )

    print()
    print("CLUSTER SUMMARY")
    print("-" * 80)
    display = ["style_cluster", "team_count"] + [
        f"{c}_mean" for c in feature_columns
    ]
    if "extrapolated_teams" in summary.columns:
        display.append("extrapolated_teams")
    print(
        summary[display].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    print()
    print("CLUSTER x MODAL FORMATION CROSSTAB")
    print("-" * 80)
    if crosstab.empty:
        print("(modal_formation_code not present in features file)")
    else:
        print(crosstab.to_string())
        print()
        if spanning == 0:
            print(
                "  FAIL  no formation appears in more than one cluster.  "
                "The clustering is a re-spelling of formation, not style."
            )
        else:
            print(
                f"  PASS  {spanning} formation(s) appear in more than one "
                f"cluster."
            )

    print()
    print("BETWEEN vs WITHIN PPDA SPREAD")
    print("-" * 80)
    print(f"  Between-cluster PPDA spread:    {between:.4f}")
    print(f"  Within-cluster median PPDA std: {within:.4f}")
    if between > within:
        print("  PASS  clusters separate on PPDA.")
    else:
        print(
            "  FAIL  between-cluster spread does not exceed within-cluster "
            "std."
        )
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the shipped style clustering."
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

    features, clusters = _load(processed_dir)
    merged, fit, feature_columns = _prepare(features, clusters)

    if fit.empty or len(fit) < 4:
        raise ValueError(
            "Too few fit teams to validate the clustering."
        )

    chosen_k = int(merged["style_cluster"].nunique())

    X = fit[feature_columns].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    sweep = _silhouette_sweep(X_scaled, n_max=len(fit))
    summary = _cluster_summary(merged, feature_columns)
    crosstab, spanning = _formation_crosstab(merged)
    between, within = _between_within_ppda(merged)

    sweep.to_csv(
        output_dir / "style_cluster_validation_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "style_cluster_validation_summary.csv",
        index=False,
    )
    if not crosstab.empty:
        crosstab.to_csv(
            output_dir / "style_cluster_validation_crosstab.csv",
        )

    _report(
        merged=merged,
        fit=fit,
        feature_columns=feature_columns,
        sweep=sweep,
        summary=summary,
        crosstab=crosstab,
        spanning=spanning,
        between=between,
        within=within,
        chosen_k=chosen_k,
    )

    print()
    print(f"Wrote {output_dir / 'style_cluster_validation_metrics.csv'}")
    print(f"Wrote {output_dir / 'style_cluster_validation_summary.csv'}")
    if not crosstab.empty:
        print(
            f"Wrote {output_dir / 'style_cluster_validation_crosstab.csv'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())