"""
Style-cluster stability, computed on the clustering that actually shipped.

The previous version hard-coded K = 4 and clustered on features that
included formation counts.  That clustering no longer exists: the
production pipeline (style_clustering.py) sets FORMATION_WEIGHT = 0,
sweeps k = 2..6 by silhouette, fits the scaler and KMeans on teams with
>= 10 matches, and flags everyone else as extrapolated.  Re-running the
old script reported ARI = 1.0 against a partition the pipeline does not
produce, which is a false validation.

This version:

  * reads team_style_clusters.csv to learn K (the shipped cluster count)
  * reads team_style_features.csv to get the same feature columns
  * fits only on teams where style_fit_team is True (the same teams the
    shipped model was fit on)
  * reports seed-stability and 80% subsample stability against the
    shipped labels
  * fails loudly if K = 1 (a degenerate clustering) or if no team carries
    style_fit_team
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler


# Feature selection mirrors style_clustering.py.  PPDA features are always
# present; the five behavioural features are optional depending on whether
# src/run_behaviour_features.py has been run.
PPDA_FEATURES = ["ppda_mean", "ppda_std", "ppda_slope"]
BEHAVIOUR_FEATURES = [
    "pass_length_mean",
    "possession_share",
    "width",
    "passes_per_possession",
    "def_line_height",
]

N_REPEATS = 100
SUBSAMPLE_FRACTION = 0.80
BASE_SEED = 42


def _resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in PPDA_FEATURES if c in df.columns]
    for c in BEHAVIOUR_FEATURES:
        if c in df.columns:
            cols.append(c)
    if len(cols) < 2:
        raise ValueError(
            "At least two of the PPDA feature columns must be present; "
            f"found {cols}"
        )
    return cols


def _load_merged(processed_dir: Path) -> tuple[pd.DataFrame, list[str], int]:
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

    features = pd.read_csv(features_path)
    clusters = pd.read_csv(clusters_path)

    needed = {"team_id", "style_cluster"}
    missing = needed - set(clusters.columns)
    if missing:
        raise ValueError(
            "team_style_clusters.csv is missing: "
            + ", ".join(sorted(missing))
        )

    feature_columns = _resolve_feature_columns(features)

    keep_cols = ["team_id"] + feature_columns
    if "style_fit_team" in clusters.columns:
        keep_cols_clusters = ["team_id", "style_cluster", "style_fit_team"]
    else:
        keep_cols_clusters = ["team_id", "style_cluster"]

    merged = features[keep_cols].merge(
        clusters[keep_cols_clusters],
        on="team_id",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.dropna(subset=feature_columns).reset_index(drop=True)

    K = int(merged["style_cluster"].nunique())
    if K < 2:
        raise ValueError(
            f"Shipped clustering has K = {K}; stability cannot be assessed. "
            "Re-run src/run_style_clustering.py and check the silhouette sweep."
        )

    return merged, feature_columns, K


def _fit_teams(merged: pd.DataFrame) -> pd.DataFrame:
    if "style_fit_team" in merged.columns:
        mask = merged["style_fit_team"].fillna(False).astype(bool)
    else:
        # Older cluster files did not record which teams were fit on.
        # Falling back to all teams is the only defensible choice, but the
        # user should re-run style_clustering.py to get the flag.
        print(
            "[stability] style_fit_team column not present; using all teams. "
            "Re-run style_clustering.py to record the fit subset."
        )
        mask = pd.Series(True, index=merged.index)
    return merged[mask].reset_index(drop=True)


def _repeated_kmeans(
    X_scaled: np.ndarray,
    shipped_labels: np.ndarray,
    K: int,
) -> pd.DataFrame:

    rows = []
    for repeat in range(N_REPEATS):
        seed = BASE_SEED + repeat + 1
        model = KMeans(n_clusters=K, random_state=seed, n_init=50)
        labels = model.fit_predict(X_scaled)
        rows.append({
            "test_type": "repeated_kmeans",
            "repeat": repeat + 1,
            "random_state": seed,
            "sample_size": len(X_scaled),
            "sample_fraction": 1.0,
            "adjusted_rand_index": adjusted_rand_score(
                shipped_labels, labels
            ),
            "normalized_mutual_information": normalized_mutual_info_score(
                shipped_labels, labels
            ),
        })
    return pd.DataFrame(rows)


def _subsample(
    X: np.ndarray,
    shipped_labels: np.ndarray,
    K: int,
) -> pd.DataFrame:

    rng = np.random.default_rng(BASE_SEED)
    n = len(X)
    size = int(round(n * SUBSAMPLE_FRACTION))
    if size <= K:
        raise ValueError(
            f"80% subsample ({size}) must exceed K ({K}) for a fit."
        )

    rows = []
    for repeat in range(N_REPEATS):
        idx = np.sort(rng.choice(n, size=size, replace=False))
        X_sub = X[idx]
        y_sub_shipped = shipped_labels[idx]

        scaler = StandardScaler().fit(X_sub)
        X_sub_scaled = scaler.transform(X_sub)

        seed = BASE_SEED + 1000 + repeat
        model = KMeans(n_clusters=K, random_state=seed, n_init=50)
        labels = model.fit_predict(X_sub_scaled)

        rows.append({
            "test_type": "subsample_kmeans",
            "repeat": repeat + 1,
            "random_state": seed,
            "sample_size": size,
            "sample_fraction": size / n,
            "adjusted_rand_index": adjusted_rand_score(
                y_sub_shipped, labels
            ),
            "normalized_mutual_information": normalized_mutual_info_score(
                y_sub_shipped, labels
            ),
        })
    return pd.DataFrame(rows)


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_type, group in results.groupby("test_type", sort=False):
        ari = group["adjusted_rand_index"]
        nmi = group["normalized_mutual_information"]
        rows.append({
            "test_type": test_type,
            "repeats": len(group),
            "ari_mean": ari.mean(),
            "ari_std": ari.std(ddof=1),
            "ari_min": ari.min(),
            "ari_median": ari.median(),
            "ari_max": ari.max(),
            "nmi_mean": nmi.mean(),
            "nmi_std": nmi.std(ddof=1),
            "nmi_min": nmi.min(),
            "nmi_median": nmi.median(),
            "nmi_max": nmi.max(),
        })
    return pd.DataFrame(rows)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Style-cluster stability against the shipped clustering "
            "produced by src/run_style_clustering.py."
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
    args = parser.parse_args()

    processed_dir = args.processed_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else processed_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SPORTSM5 STYLE-CLUSTER STABILITY")
    print("=" * 80)

    merged, feature_columns, K = _load_merged(processed_dir)
    print(f"Teams (merged):          {len(merged)}")
    print(f"Shipped K:               {K}")
    print(f"Feature columns used:    {feature_columns}")

    fit = _fit_teams(merged)
    print(f"Fit teams (style_fit_team=True): {len(fit)}")

    if len(fit) < K + 1:
        raise ValueError(
            f"Only {len(fit)} fit teams for K = {K}; cannot assess stability."
        )

    X = fit[feature_columns].to_numpy(dtype=float)
    y_shipped = fit["style_cluster"].to_numpy(dtype=int)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print(f"Repeats per test:        {N_REPEATS}")
    print(f"Subsample fraction:      {SUBSAMPLE_FRACTION:.0%}")
    print()

    repeated = _repeated_kmeans(X_scaled, y_shipped, K)
    subsample = _subsample(X, y_shipped, K)

    results = pd.concat([repeated, subsample], ignore_index=True)
    summary = _summarize(results)

    results.to_csv(
        output_dir / "style_cluster_stability.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "style_cluster_stability_summary.csv",
        index=False,
    )

    print("STABILITY RESULTS")
    print("-" * 80)
    display = [
        "test_type", "repeats",
        "ari_mean", "ari_std", "ari_min", "ari_median", "ari_max",
        "nmi_mean", "nmi_std", "nmi_min", "nmi_median", "nmi_max",
    ]
    print(
        summary[display].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )

    # --- interpretation guide, adjusted for continuous features --------
    print()
    print("INTERPRETATION")
    print("-" * 80)
    print(
        "ARI = 1.0 is no longer the target.  On continuous PPDA and\n"
        "behavioural features, KMeans restarts are expected to give ARI\n"
        "well below 1.0, because the clusters are not separable by a\n"
        "discrete driver.  Read the values as:\n"
        "  repeated  ARI > 0.85   stable under random initialisation\n"
        "  repeated  ARI 0.6-0.85 moderate, some restart sensitivity\n"
        "  repeated  ARI < 0.6    unstable; the clustering does not survive\n"
        "                          an arbitrary seed change and should not\n"
        "                          be reported as a fixed label.\n"
        "  subsample ARI < repeated ARI by a wide margin indicates the\n"
        "                          cluster boundaries depend on individual\n"
        "                          teams.\n"
    )
    print(f"Wrote {output_dir / 'style_cluster_stability.csv'}")
    print(f"Wrote {output_dir / 'style_cluster_stability_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())