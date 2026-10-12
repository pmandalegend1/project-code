#!/usr/bin/env python3
"""
make_crossfit_folds.py -- write the match lists for cross-fitting.

Uses exactly the split train_opponent.py uses (GroupKFold by match on the
validated triples, same order), so fold k's opponent model, fold k's world
model and fold k's decisions all agree on which matches are held out.

    python make_crossfit_folds.py --out-dir ~/user/crossfit
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import GroupKFold

from ramfp.config import Config
from ramfp.data.schema import validate_triples

ap = argparse.ArgumentParser()
ap.add_argument("--triples", default="data/processed/triples.parquet")
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()

cfg = Config.load(a.config)
df = validate_triples(pd.read_parquet(a.triples))
out = os.path.expanduser(a.out_dir)
os.makedirs(out, exist_ok=True)
summary = {}
for k, (_, te) in enumerate(GroupKFold(n_splits=cfg.train.n_folds).split(df, groups=df["match_id"].to_numpy())):
    m = sorted(int(x) for x in pd.unique(df.iloc[te]["match_id"]))
    with open(os.path.join(out, f"fold{k}_matches.txt"), "w") as f:
        f.write(f"# fold {k}: {len(m)} matches, {len(te)} decisions (GroupKFold by match, "
                f"n_splits={cfg.train.n_folds})\n")
        f.write("\n".join(map(str, m)) + "\n")
    summary[k] = {"matches": len(m), "decisions": int(len(te))}
    print(f"fold {k}: {len(m)} matches, {len(te)} decisions")
json.dump(summary, open(os.path.join(out, "folds.json"), "w"), indent=2)
print(f"wrote {out}/fold*_matches.txt")
