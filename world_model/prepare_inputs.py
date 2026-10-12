#!/usr/bin/env python3
"""
prepare_inputs.py -- turn the measurement pipeline's outputs into the two
inputs the world model expects that nothing else produces.

The measurement pipeline writes CSVs under processed-data/ with
its own column names.  train_world_model.py reads parquet files under
data/processed/ with different names, and a formation_vocab.json that no
script in the repository ever wrote.  This is the adapter between them, kept
as one explicit file rather than as edits scattered through either side, so
the contract between the two halves of the pipeline is written down in one
place and can be stated in the paper in a sentence.

Writes
------
data/processed/team_style_clusters.parquet   team_id, style_cluster (+ context)
data/processed/formation_vocab.json          {"vocab": [codes...]}

Run it AFTER extract_stable_windows.py, so the vocabulary is derived from the
windows the model will actually train on rather than assumed.

    python prepare_inputs.py --measurement ../measurements/processed-data
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# The opponent model's vocabulary, so both halves of the pipeline index the
# same formations. Kept literal rather than imported: the two packages are
# not installed together, and a silent divergence here would be invisible.
RAMFP_FORMATIONS = (
    442, 4231, 433, 4141, 4411, 4321, 4222, 451,
    4132, 352, 3412, 3421, 343, 532, 541,
    41212, 3511,
)


def build_style_clusters(measurement_dir, out_path):
    src = os.path.join(measurement_dir, "team_style_clusters.csv")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run the measurement pipeline first")
    df = pd.read_csv(src)

    if "style_cluster" not in df.columns or "team_id" not in df.columns:
        raise SystemExit(f"{src} has no team_id/style_cluster columns: {list(df.columns)}")

    keep = [c for c in (
        "team_id", "style_cluster", "pressing_regime", "ppda_mean", "ppda_std",
        "ppda_slope", "match_count", "style_label_is_extrapolated",
    ) if c in df.columns]
    out = df[keep].copy()
    out["team_id"] = out["team_id"].astype(int)
    out["style_cluster"] = out["style_cluster"].astype(int)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)

    counts = out["style_cluster"].value_counts().sort_index().to_dict()
    print(f"[style ] {len(out)} teams -> {out_path}")
    print(f"         clusters: {counts}")
    if "style_label_is_extrapolated" in out.columns:
        n_ext = int(out["style_label_is_extrapolated"].sum())
        print(f"         {n_ext} of {len(out)} labels are extrapolated "
              f"(team had too few matches to fit on)")
    if out["style_cluster"].nunique() < 2:
        print("         WARNING: a single style cluster -- the style-conditioning "
              "ablation cannot say anything on this data")
    return out


def build_formation_vocab(windows_path, out_path):
    if not os.path.exists(windows_path):
        raise SystemExit(
            f"missing {windows_path} -- run extract_stable_windows.py first, so the "
            f"vocabulary comes from the data instead of being assumed"
        )
    w = pd.read_parquet(windows_path, columns=["home_formation", "away_formation"])
    observed = pd.concat([w["home_formation"], w["away_formation"]]).dropna().astype(int)
    counts = observed.value_counts()

    vocab = sorted(set(counts.index.tolist()) | set(RAMFP_FORMATIONS))
    covered = int(counts[[c for c in counts.index if c in set(vocab)]].sum())
    print(f"[vocab ] {len(counts)} formations observed in {len(w):,} windows; "
          f"vocabulary of {len(vocab)} covers {100.0 * covered / max(counts.sum(), 1):.2f}% of appearances")

    missing_from_data = sorted(set(RAMFP_FORMATIONS) - set(counts.index))
    if missing_from_data:
        print(f"         in the opponent-model vocabulary but unseen here: {missing_from_data}")
    print(f"         top 8: {counts.head(8).to_dict()}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "vocab": [int(c) for c in vocab],
            "derived_from": os.path.basename(windows_path),
            "note": "union of formations observed in the stable windows and the "
                    "opponent model's vocabulary, so both halves index the same set",
        }, f, indent=2)
    print(f"[vocab ] wrote {out_path}")
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement", default="../measurements/processed-data",
                    help="directory holding team_style_clusters.csv")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--windows", default=None,
                    help="stable_windows.parquet (default: <data-dir>/stable_windows.parquet)")
    args = ap.parse_args()

    windows = args.windows or os.path.join(args.data_dir, "stable_windows.parquet")

    build_style_clusters(args.measurement,
                         os.path.join(args.data_dir, "team_style_clusters.parquet"))
    build_formation_vocab(windows,
                          os.path.join(args.data_dir, "formation_vocab.json"))
    print("\nready: python train_world_model.py --config configs/world_model.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
