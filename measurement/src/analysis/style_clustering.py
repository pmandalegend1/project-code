from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


MAIN_MATCH_MINUTES = 90
# Two clusters, not four.
#
# The clustering features are ppda_mean, ppda_std and ppda_slope -- three
# summaries of one behaviour, pressing.  ppda_mean and ppda_std correlate at
# r = +0.69 on the 1200-match corpus, so the feature space is effectively
# about one and a half dimensions, and a silhouette sweep over the 46 teams
# with >= 10 matches picks k = 2 (0.3906), with k = 4 the WORST value in the
# range 2..7 (0.2943):
#
#     k        2       3       4       5       6       7
#     sil   .3906   .3792   .2943   .3031   .2870   .2899
#
# Four labels were therefore splitting noise.  The honest construct these
# features support is a binary PRESSING REGIME -- high press vs low block --
# and the paper should name it that rather than "playing style", which would
# imply directness, possession, width and tempo that are not measured here.
#
# To recover a genuine multi-dimensional style, add non-PPDA behavioural
# features (pass length, possession share, lateral spread, passes per
# possession, defensive line height) and re-run the sweep before raising k.
N_CLUSTERS = 2


def _formation_from_event(event: dict) -> int | None:
    tactics = event.get("tactics")
    if not isinstance(tactics, dict):
        return None
    formation = tactics.get("formation")
    if formation is None:
        return None
    try:
        return int(formation)
    except (TypeError, ValueError):
        return None


def _formation_observations(
    events_dir: Path,
    game_ids: set[int],
) -> pd.DataFrame:
    rows: list[dict] = []

    for game_id in sorted(game_ids):
        path = events_dir / f"{game_id}.json"
        if not path.is_file():
            continue

        with path.open("r", encoding="utf-8") as file:
            events = json.load(file)

        first_by_team: dict[int, int] = {}

        # Prefer the Starting XI formation: exactly one match-start formation
        # per team, rather than overweighting teams that make many tactical shifts.
        for event in events:
            if event.get("type", {}).get("name") != "Starting XI":
                continue
            team = event.get("team")
            if not isinstance(team, dict):
                continue
            team_id = team.get("id")
            formation = _formation_from_event(event)
            if team_id is None or formation is None:
                continue
            first_by_team.setdefault(int(team_id), formation)

        # Fallback to the first available tactical formation if Starting XI
        # is unavailable.
        if len(first_by_team) < 2:
            for event in events:
                if event.get("type", {}).get("name") != "Tactical Shift":
                    continue
                team = event.get("team")
                if not isinstance(team, dict):
                    continue
                team_id = team.get("id")
                formation = _formation_from_event(event)
                if team_id is None or formation is None:
                    continue
                first_by_team.setdefault(int(team_id), formation)

        for team_id, formation in first_by_team.items():
            rows.append(
                {
                    "game_id": int(game_id),
                    "team_id": int(team_id),
                    "formation_code": int(formation),
                }
            )

    return pd.DataFrame(rows)


def _formation_vector(code: int | float | None) -> list[float]:
    if code is None or pd.isna(code):
        return [np.nan] * 5

    digits = [int(x) for x in str(int(code))]
    if len(digits) < 3:
        return [np.nan] * 5

    # Fixed representation for K-means:
    # defenders, first midfield line, second midfield line,
    # third midfield/attacking line, forwards.
    defenders = digits[0]
    forwards = digits[-1]
    middle = digits[1:-1][:3]
    middle += [0] * (3 - len(middle))
    return [
        float(defenders),
        float(middle[0]),
        float(middle[1]),
        float(middle[2]),
        float(forwards),
    ]


def build_style_features(
    processed_dir: str | Path = "processed-data",
    open_data_root: str | Path = "open-data/data",
) -> pd.DataFrame:
    processed_dir = Path(processed_dir).expanduser().resolve()
    data_root = Path(open_data_root).expanduser().resolve()

    match_ppda = pd.read_csv(processed_dir / "match_ppda.csv")
    ppda15 = pd.read_csv(processed_dir / "ppda_15min.csv")
    inventory = pd.read_csv(processed_dir / "dataset_inventory.csv")

    # Match-level PPDA statistics.
    base = (
        match_ppda.groupby("pressing_team_id")["ppda"]
        .agg(
            ppda_mean="mean",
            ppda_std="std",
            match_count="count",
        )
        .reset_index()
        .rename(columns={"pressing_team_id": "team_id"})
    )

    # Within-match 15-minute slope for each team-match.
    slopes = []
    for (game_id, team_id), group in ppda15.groupby(
        ["game_id", "team_id"],
        sort=False,
    ):
        group = group.dropna(subset=["ppda"]).copy()
        if len(group) < 3:
            continue

        x = (
            group["window_start_minute"].to_numpy(dtype=float)
            + 7.5
        )
        y = group["ppda"].to_numpy(dtype=float)

        slope = float(np.polyfit(x, y, 1)[0])
        slopes.append(
            {
                "game_id": int(game_id),
                "team_id": int(team_id),
                "ppda_slope": slope,
            }
        )

    slope_df = pd.DataFrame(slopes)

    if slope_df.empty:
        raise RuntimeError(
            "Could not compute any within-match PPDA slopes."
        )

    slope_team = (
        slope_df.groupby("team_id")["ppda_slope"]
        .agg(ppda_slope="mean", slope_match_count="count")
        .reset_index()
    )

    features = base.merge(
        slope_team,
        on="team_id",
        how="left",
    )

    # Modal formation across one formation observation per team-match.
    game_ids = set(inventory["game_id"].astype(int))
    formation_obs = _formation_observations(
        data_root / "events",
        game_ids,
    )

    if formation_obs.empty:
        raise RuntimeError(
            "No formation observations could be extracted from the "
            "StatsBomb event files."
        )

    modal = (
        formation_obs.groupby("team_id")["formation_code"]
        .agg(
            modal_formation_code=lambda s: int(s.mode().iloc[0]),
            formation_match_count="count",
        )
        .reset_index()
    )

    features = features.merge(
        modal,
        on="team_id",
        how="left",
    )

    vector_df = features["modal_formation_code"].apply(_formation_vector)
    vector_df = pd.DataFrame(
        vector_df.tolist(),
        columns=[
            "formation_defenders",
            "formation_mid_1",
            "formation_mid_2",
            "formation_mid_3",
            "formation_forwards",
        ],
        index=features.index,
    )
    features = pd.concat([features, vector_df], axis=1)

    features = features.dropna(
        subset=[
            "ppda_mean",
            "ppda_std",
            "ppda_slope",
            "modal_formation_code",
        ]
    ).reset_index(drop=True)

    # --- behavioural features, if the extractor has been run ---------------
    # PPDA answers "how hard do they press". These answer what kind of side
    # they are: how directly they play, how much ball they take, how wide
    # they spread, how quickly they move it, and how high they defend.
    # Without them "style" is one behaviour measured three ways.
    behaviour_path = (
        Path(processed_dir)
        / "team_behaviour_features.csv"
    )

    if behaviour_path.exists():
        behaviour = pd.read_csv(behaviour_path)
        behaviour["team_id"] = pd.to_numeric(
            behaviour["team_id"],
            errors="coerce",
        )
        features = features.merge(
            behaviour.drop(columns=["behaviour_match_count"],
                           errors="ignore"),
            on="team_id",
            how="left",
        )
        print(
            f"[style] merged {len(behaviour)} teams of behavioural features"
        )
    else:
        print(
            f"[style] {behaviour_path} not found -- clustering on PPDA "
            f"summaries alone. Run src/run_behaviour_features.py first if "
            f"you want style to mean more than pressing intensity."
        )

    return features


def cluster_styles(
    features: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    continuous = [
        "ppda_mean",
        "ppda_std",
        "ppda_slope",
    ]

    BEHAVIOUR = [
        "pass_length_mean",
        "possession_share",
        "width",
        "passes_per_possession",
        "def_line_height",
    ]
    present = [c for c in BEHAVIOUR if c in features.columns]
    if present:
        # Drop teams the extractor could not describe rather than imputing a
        # style for them: a made-up style label is worse than no label.
        before = len(features)
        features = features.dropna(subset=present).reset_index(drop=True)
        if len(features) < before:
            print(f"[style] dropped {before - len(features)} teams with no "
                  f"behavioural features")
        continuous = continuous + present
        print(f"[style] clustering on {len(continuous)} features: "
              f"{continuous}")
    formation = [
        "formation_defenders",
        "formation_mid_1",
        "formation_mid_2",
        "formation_mid_3",
        "formation_forwards",
    ]

    # FIX: formation counts are excluded from the clustering features.
    #
    # With them included, K-means split purely on formation: every formation
    # code mapped to exactly one cluster (zero overlap), cluster mean PPDA was
    # 2.396 / 2.405 / 2.373 / 2.164 — a 0.24 spread against a 0.33-0.54 spread
    # WITHIN clusters — and stability came out at ari_mean = 1.0000 exactly
    # over 100 repeats, which continuous features never produce.
    #
    # The label was therefore a re-spelling of modal formation, and the
    # opponent model already receives the opponent's formation separately,
    # so style conditioning could not add information by construction.
    #
    # Set FORMATION_WEIGHT > 0 to reintroduce shape as a *weighted* signal
    # rather than at parity; report the weight in the paper if you do.
    FORMATION_WEIGHT = 0.0

    # FIX: fit the clusters only on teams with enough matches behind their
    # style vector.
    #
    # The open-data corpus contains whole league seasons alongside partial
    # seasons where a single club's opponents appear once or twice.  A team
    # mean PPDA estimated from 2 matches carries roughly sqrt(19) times the
    # standard error of one estimated from 38 (the median), so those teams
    # land at extreme values and K-means spends clusters on them: on the
    # 1200-match corpus the raw fit returned sizes 37 / 1 / 1 / 28, i.e. two
    # clusters holding one team each, both with match_count = 2.  A style
    # label learned from a single team is noise, and the opponent model
    # would be conditioning on it.
    #
    # So: fit the centroids on reliable teams, then assign every team --
    # including the sparse ones -- to its nearest centroid, and flag the
    # extrapolated ones so downstream code can drop or down-weight them.
    MIN_MATCHES_FOR_FIT = 10

    if "match_count" not in features.columns:
        raise ValueError(
            "style features must carry 'match_count' so that unreliable "
            "team vectors can be excluded from the K-means fit"
        )

    match_count = features["match_count"].to_numpy(dtype=float)
    reliable = match_count >= MIN_MATCHES_FOR_FIT

    matrix = features[continuous].to_numpy(dtype=float)

    if int(reliable.sum()) <= n_clusters:
        raise ValueError(
            f"Need more than {n_clusters} teams with >= "
            f"{MIN_MATCHES_FOR_FIT} matches for K-means; "
            f"got {int(reliable.sum())}."
        )

    # Scaler is fitted on reliable teams only, so outliers cannot stretch
    # the standardisation either.
    scaler = StandardScaler()
    scaler.fit(matrix[reliable])
    X = scaler.transform(matrix)

    if FORMATION_WEIGHT > 0:
        formation_scaler = StandardScaler()
        X = np.hstack([
            X,
            FORMATION_WEIGHT * formation_scaler.fit_transform(
                features[formation].to_numpy(dtype=float)
            ),
        ])

    # Choose k the same way it was chosen before: a silhouette sweep on the
    # reliable teams. On the three PPDA summaries that rule picked k = 2, and
    # N_CLUSTERS was set to 2 on its say-so. With the behavioural features
    # added the feature space is different, so the same rule has to be asked
    # again rather than inheriting its old answer. Reporting the whole sweep
    # keeps the choice auditable.
    sweep = {}
    for k in range(2, 7):
        if k >= int(reliable.sum()):
            break
        trial = KMeans(n_clusters=k, random_state=random_state,
                       n_init=50).fit(X[reliable])
        sweep[k] = float(silhouette_score(X[reliable], trial.labels_))
    if sweep:
        best = max(sweep, key=sweep.get)
        print("[style] silhouette sweep: "
              + "  ".join(f"k={k}:{v:.4f}" for k, v in sweep.items()))
        if best != n_clusters:
            print(f"[style] k = {best} selected (was {n_clusters}); "
                  f"set ramfp.vocab.N_STYLES = {best} before rebuilding "
                  f"the triples, or the opponent model will index an "
                  f"embedding that does not exist")
        n_clusters = best

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=50,
    )
    model.fit(X[reliable])
    labels = model.predict(X)

    result = features.copy()
    result["style_cluster"] = labels.astype(int)
    result["style_fit_team"] = reliable
    result["style_label_is_extrapolated"] = ~reliable
    result["style_min_matches_for_fit"] = MIN_MATCHES_FOR_FIT

    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_[:, : len(continuous)]),
        columns=continuous,
    )
    centers["style_cluster"] = range(n_clusters)

    # Cluster numbering is arbitrary. Add a descriptive rank by mean PPDA.
    ppda_order = (
        centers.sort_values("ppda_mean")["style_cluster"]
        .tolist()
    )
    rank_map = {
        cluster: rank
        for rank, cluster in enumerate(ppda_order, start=1)
    }
    result["ppda_intensity_rank"] = result["style_cluster"].map(rank_map)
    centers["ppda_intensity_rank"] = centers["style_cluster"].map(rank_map)

    # Human-readable name for the thing actually being measured.  The
    # numeric column keeps the name "style_cluster" so that did.py and the
    # opponent model do not have to change, but the paper should report
    # this label: the construct is pressing intensity, not playing style.
    if n_clusters == 2:
        regime = {1: "high_press", 2: "low_block"}
    else:
        regime = {
            r: f"press_regime_{r}"
            for r in range(1, n_clusters + 1)
        }
    result["pressing_regime"] = (
        result["ppda_intensity_rank"].map(regime)
    )
    centers["pressing_regime"] = (
        centers["ppda_intensity_rank"].map(regime)
    )

    # Silhouette on the fitted subset: scoring extrapolated teams would
    # report the quality of a fit that never saw them.
    silhouette = float(
        silhouette_score(X[reliable], labels[reliable])
    )

    return result, centers.sort_values("ppda_intensity_rank"), silhouette


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and cluster SportsM5 tactical style features."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed-data"),
    )
    parser.add_argument(
        "--open-data-root",
        type=Path,
        default=Path("open-data/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.processed_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_style_features(
        processed_dir=args.processed_dir,
        open_data_root=args.open_data_root,
    )
    clustered, centers, silhouette = cluster_styles(features)

    features.to_csv(
        output_dir / "team_style_features.csv",
        index=False,
    )
    clustered.to_csv(
        output_dir / "team_style_clusters.csv",
        index=False,
    )
    centers.to_csv(
        output_dir / "style_cluster_centers.csv",
        index=False,
    )

    print("=" * 80)
    print("SPORTSM5 PRESSING REGIME CLUSTERING")
    print("=" * 80)
    print(f"Teams clustered:       {len(clustered)}")
    print(f"Clusters:              {clustered['style_cluster'].nunique()}")
    print(f"Silhouette score:      {silhouette:.4f}")
    print()
    print("Cluster sizes:")
    print(clustered["style_cluster"].value_counts().sort_index().to_string())
    print()
    print("Cluster centers:")
    print(centers.to_string(index=False))
    print()
    print(f"Features: {output_dir / 'team_style_features.csv'}")
    print(f"Labels:   {output_dir / 'team_style_clusters.csv'}")
    print(f"Centers:  {output_dir / 'style_cluster_centers.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
