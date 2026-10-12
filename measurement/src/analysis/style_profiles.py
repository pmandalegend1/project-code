from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WINDOW_STARTS = [0, 15, 30, 45, 60, 75]
WINDOW_LABELS = [
    "0-15",
    "15-30",
    "30-45",
    "45-60",
    "60-75",
    "75-90",
]


def load_inputs(
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the already-generated 15-minute PPDA data and
    team-level style-cluster assignments.
    """

    processed_dir = Path(processed_dir).expanduser().resolve()

    ppda_path = processed_dir / "ppda_15min.csv"
    clusters_path = processed_dir / "team_style_clusters.csv"

    if not ppda_path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {ppda_path}"
        )

    if not clusters_path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {clusters_path}"
        )

    ppda = pd.read_csv(ppda_path)
    clusters = pd.read_csv(clusters_path)

    return ppda, clusters


def validate_inputs(
    ppda: pd.DataFrame,
    clusters: pd.DataFrame,
) -> None:
    """
    Validate the minimum schema required for temporal
    style-profile analysis.
    """

    required_ppda = {
        "game_id",
        "team_id",
        "window_start_minute",
        "window_end_minute",
        "ppda",
    }

    required_clusters = {
        "team_id",
        "style_cluster",
    }

    missing_ppda = required_ppda - set(ppda.columns)

    if missing_ppda:
        raise ValueError(
            "ppda_15min.csv is missing required columns: "
            + ", ".join(sorted(missing_ppda))
        )

    missing_clusters = required_clusters - set(clusters.columns)

    if missing_clusters:
        raise ValueError(
            "team_style_clusters.csv is missing required columns: "
            + ", ".join(sorted(missing_clusters))
        )

    if ppda.empty:
        raise ValueError(
            "ppda_15min.csv is empty."
        )

    if clusters.empty:
        raise ValueError(
            "team_style_clusters.csv is empty."
        )

    if clusters["team_id"].duplicated().any():
        duplicated = (
            clusters.loc[
                clusters["team_id"].duplicated(keep=False),
                "team_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "team_style_clusters.csv contains duplicate team IDs: "
            f"{duplicated[:10]}"
        )

    invalid_starts = set(
        pd.to_numeric(
            ppda["window_start_minute"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    ) - set(WINDOW_STARTS)

    if invalid_starts:
        raise ValueError(
            "Unexpected PPDA window_start_minute values: "
            f"{sorted(invalid_starts)}"
        )


def build_temporal_profiles(
    ppda: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build cluster-level temporal PPDA profiles.

    Each row represents one style cluster and one 15-minute
    match window.

    The mean is calculated over all available team-match
    PPDA observations belonging to that style cluster and
    window.

    Missing PPDA values are excluded from the mean rather
    than converted to zero.
    """

    data = ppda[
        [
            "game_id",
            "team_id",
            "window_start_minute",
            "window_end_minute",
            "ppda",
        ]
    ].copy()

    style = clusters[
        ["team_id", "style_cluster"]
    ].copy()

    data["team_id"] = pd.to_numeric(
        data["team_id"],
        errors="coerce",
    )

    style["team_id"] = pd.to_numeric(
        style["team_id"],
        errors="coerce",
    )

    data["window_start_minute"] = pd.to_numeric(
        data["window_start_minute"],
        errors="coerce",
    )

    data["window_end_minute"] = pd.to_numeric(
        data["window_end_minute"],
        errors="coerce",
    )

    data["ppda"] = pd.to_numeric(
        data["ppda"],
        errors="coerce",
    )

    style["style_cluster"] = pd.to_numeric(
        style["style_cluster"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "team_id",
            "window_start_minute",
        ]
    )

    style = style.dropna(
        subset=[
            "team_id",
            "style_cluster",
        ]
    )

    data["team_id"] = data["team_id"].astype(int)
    style["team_id"] = style["team_id"].astype(int)
    data["window_start_minute"] = (
        data["window_start_minute"].astype(int)
    )

    merged = data.merge(
        style,
        on="team_id",
        how="inner",
        validate="many_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No PPDA observations matched any style-cluster "
            "team IDs."
        )

    profiles = (
        merged.groupby(
            [
                "style_cluster",
                "window_start_minute",
            ],
            dropna=False,
        )["ppda"]
        .agg(
            mean_ppda="mean",
            std_ppda="std",
            n_observations="count",
        )
        .reset_index()
    )

    profiles["window_end_minute"] = (
        profiles["window_start_minute"] + 15
    )

    profiles["window_label"] = (
        profiles["window_start_minute"]
        .map(
            dict(
                zip(
                    WINDOW_STARTS,
                    WINDOW_LABELS,
                )
            )
        )
    )

    profiles["style_cluster"] = (
        profiles["style_cluster"].astype(int)
    )

    # Ensure every cluster has an explicit row for each of
    # the six expected windows. Missing windows remain NaN
    # rather than being incorrectly treated as PPDA = 0.
    clusters_present = sorted(
        profiles["style_cluster"].unique()
    )

    complete_index = pd.MultiIndex.from_product(
        [
            clusters_present,
            WINDOW_STARTS,
        ],
        names=[
            "style_cluster",
            "window_start_minute",
        ],
    )

    profiles = (
        profiles.set_index(
            [
                "style_cluster",
                "window_start_minute",
            ]
        )
        .reindex(complete_index)
        .reset_index()
    )

    profiles["window_end_minute"] = (
        profiles["window_start_minute"] + 15
    )

    profiles["window_label"] = (
        profiles["window_start_minute"]
        .map(
            dict(
                zip(
                    WINDOW_STARTS,
                    WINDOW_LABELS,
                )
            )
        )
    )

    profiles = profiles[
        [
            "style_cluster",
            "window_start_minute",
            "window_end_minute",
            "window_label",
            "mean_ppda",
            "std_ppda",
            "n_observations",
        ]
    ]

    profiles = profiles.sort_values(
        [
            "style_cluster",
            "window_start_minute",
        ]
    ).reset_index(drop=True)

    return profiles


def add_cluster_metadata(
    profiles: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cluster size and PPDA intensity rank when available.
    """

    metadata_columns = [
        "style_cluster",
    ]

    optional_columns = [
        "ppda_intensity_rank",
    ]

    available = [
        column
        for column in optional_columns
        if column in clusters.columns
    ]

    if available:
        metadata = (
            clusters[
                metadata_columns + available
            ]
            .drop_duplicates("style_cluster")
        )

        cluster_sizes = (
            clusters.groupby("style_cluster")
            .size()
            .rename("team_count")
            .reset_index()
        )

        metadata = metadata.merge(
            cluster_sizes,
            on="style_cluster",
            how="left",
            validate="one_to_one",
        )

        profiles = profiles.merge(
            metadata,
            on="style_cluster",
            how="left",
            validate="many_to_one",
        )

    else:
        cluster_sizes = (
            clusters.groupby("style_cluster")
            .size()
            .rename("team_count")
            .reset_index()
        )

        profiles = profiles.merge(
            cluster_sizes,
            on="style_cluster",
            how="left",
            validate="many_to_one",
        )

    return profiles


def save_profile_csv(
    profiles: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Save the long-format temporal profile table.
    """

    output_path = (
        output_dir
        / "style_cluster_temporal_profiles.csv"
    )

    profiles.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_profile_plot(
    profiles: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Save the temporal PPDA profile figure.

    Cluster numbers are intentionally kept as numerical
    labels because K-means cluster IDs have no intrinsic
    semantic meaning.
    """

    output_path = (
        output_dir
        / "style_cluster_temporal_profiles.png"
    )

    plt.figure(figsize=(10, 6))

    for cluster_id in sorted(
        profiles["style_cluster"].dropna().unique()
    ):
        group = profiles[
            profiles["style_cluster"] == cluster_id
        ].sort_values("window_start_minute")

        plt.plot(
            group["window_start_minute"],
            group["mean_ppda"],
            marker="o",
            label=f"Style cluster {int(cluster_id)}",
        )

    plt.xticks(
        WINDOW_STARTS,
        WINDOW_LABELS,
    )

    plt.xlabel("Match window (minutes)")
    plt.ylabel("Mean PPDA")
    plt.title(
        "Temporal PPDA Profiles by Tactical Style Cluster"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def print_summary(
    profiles: pd.DataFrame,
) -> None:
    """
    Print a concise validation summary.
    """

    print("=" * 80)
    print("SPORTSM5 TEMPORAL STYLE PROFILES")
    print("=" * 80)

    cluster_ids = sorted(
        profiles["style_cluster"].unique()
    )

    print(
        f"Style clusters:        {len(cluster_ids)}"
    )

    print(
        f"Expected windows:      {len(WINDOW_STARTS)}"
    )

    print()

    print("Clusters:")
    print(
        profiles[
            [
                "style_cluster",
                "window_start_minute",
                "window_end_minute",
                "mean_ppda",
                "n_observations",
            ]
        ].to_string(index=False)
    )

    print()

    missing = profiles[
        profiles["n_observations"].isna()
        | (profiles["n_observations"] == 0)
    ]

    if missing.empty:
        print(
            "Window coverage:       PASS"
        )
    else:
        print(
            "Window coverage:       WARN"
        )

        print(
            "Missing cluster-window combinations:"
        )

        print(
            missing[
                [
                    "style_cluster",
                    "window_start_minute",
                    "window_end_minute",
                ]
            ].to_string(index=False)
        )

    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build temporal PPDA profiles for "
            "SportsM5 tactical style clusters."
        )
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed-data"),
        help="Directory containing processed SportsM5 CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to --processed-dir."
        ),
    )

    args = parser.parse_args()

    processed_dir = (
        args.processed_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else processed_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ppda, clusters = load_inputs(
        processed_dir
    )

    validate_inputs(
        ppda,
        clusters,
    )

    profiles = build_temporal_profiles(
        ppda,
        clusters,
    )

    profiles = add_cluster_metadata(
        profiles,
        clusters,
    )

    csv_path = save_profile_csv(
        profiles,
        output_dir,
    )

    plot_path = save_profile_plot(
        profiles,
        output_dir,
    )

    print_summary(
        profiles,
    )

    print()
    print(
        f"Profile CSV: {csv_path}"
    )
    print(
        f"Profile plot: {plot_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())