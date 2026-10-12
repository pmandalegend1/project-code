"""
E1 baselines 1-3 from section 10 of the draft.

  1. Random          uniform over the formation vocabulary
  2. Majority class   always the most frequent response in training
  3. Persistence      the opponent keeps its current shape  <- the hard one
  4. Logistic reg.    game state + style features only, no transition input

Persistence was absent from the draft's table and is by far the strongest
of the four: the opponent's formation moves in only 11% of decisions, so
"nothing changes" is right about 89% of the time.  Reporting Top-1 against
random and majority alone would have flattered the model by roughly 60
points.

Baseline 5 (transformer without style conditioning) is an architectural
ablation and lives in train_opponent.py behind --ablate-style, so that it
shares every other hyperparameter with the full model.

    python -m ramfp.train.baselines --triples data/processed/triples.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..data.datasets import tabular_features
from ..data.schema import validate_triples
from ..eval.metrics import evaluate as compute_metrics
from ..vocab import N_FORMATIONS, formation_id
from .common import save_results, set_seed


def _labels(df: pd.DataFrame) -> np.ndarray:
    return df["opp_response"].map(formation_id).to_numpy()


def run_random(_tr: pd.DataFrame, te: pd.DataFrame, rng: np.random.Generator) -> Dict:
    probs = rng.dirichlet(np.ones(N_FORMATIONS) * 50, size=len(te))
    return compute_metrics(probs, _labels(te)).as_row()


def run_majority(tr: pd.DataFrame, te: pd.DataFrame, _rng) -> Dict:
    counts = np.bincount(_labels(tr), minlength=N_FORMATIONS).astype(np.float64)
    p = (counts + 1e-6) / (counts + 1e-6).sum()
    return compute_metrics(np.tile(p, (len(te), 1)), _labels(te)).as_row()


def run_logistic(tr: pd.DataFrame, te: pd.DataFrame, _rng) -> Dict:
    scaler = StandardScaler().fit(tabular_features(tr))
    clf = LogisticRegression(max_iter=2000, C=1.0)   # multinomial is the default
    clf.fit(scaler.transform(tabular_features(tr)), _labels(tr))
    # LogisticRegression only knows classes present in train; re-expand to |V|
    probs = np.full((len(te), N_FORMATIONS), 1e-9)
    probs[:, clf.classes_] = clf.predict_proba(scaler.transform(tabular_features(te)))
    probs /= probs.sum(axis=1, keepdims=True)
    return compute_metrics(probs, _labels(te)).as_row()


def run_persistence(_tr: pd.DataFrame, te: pd.DataFrame, _rng) -> Dict:
    """Predict that the opponent keeps the shape it is already in.

    The baseline that actually matters, and the one the draft was missing.
    On 2651 matches the opponent's formation moves in 235 of 2138 decisions,
    so this one-line rule scores about 89% -- far above "majority class",
    which only ever names the single most common formation in training and
    scores around 27%.  Any headline accuracy for the model has to be read
    against THIS number; a reviewer will ask for it on the first pass.
    """
    idx = te["opp_formation"].map(formation_id).to_numpy()
    probs = np.full((len(te), N_FORMATIONS), 1e-9)
    valid = (idx >= 0) & (idx < N_FORMATIONS)
    probs[np.arange(len(te))[valid], idx[valid]] = 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    return compute_metrics(probs, _labels(te)).as_row()


def run_transition_matrix(tr: pd.DataFrame, te: pd.DataFrame, _rng) -> Dict:
    """P(opponent's next shape | the shape they are in), counted on training rows.

    Persistence is the special case of this that puts all its mass on the
    diagonal. The full matrix is the stronger and more standard baseline: it
    also knows which shapes a side tends to move TO from where it is, which is
    real tactical structure and costs nothing to estimate. A model that reads
    the game state has to beat this, not just persistence, or it has not shown
    that the state is doing any work.

    Laplace smoothing keeps unseen (from, to) pairs from being impossible,
    which would make the log-loss infinite on the first unseen transition.
    """
    counts = np.full((N_FORMATIONS, N_FORMATIONS), 0.5)   # Laplace prior
    for cur, nxt in zip(tr["opp_formation"].map(formation_id),
                        tr["opp_response"].map(formation_id)):
        if 0 <= cur < N_FORMATIONS and 0 <= nxt < N_FORMATIONS:
            counts[cur, nxt] += 1.0
    counts /= counts.sum(axis=1, keepdims=True)

    idx = te["opp_formation"].map(formation_id).to_numpy()
    probs = np.full((len(te), N_FORMATIONS), 1.0 / N_FORMATIONS)
    valid = (idx >= 0) & (idx < N_FORMATIONS)
    probs[valid] = counts[idx[valid]]
    return compute_metrics(probs, _labels(te)).as_row()


def run_gradient_boosting(tr: pd.DataFrame, te: pd.DataFrame, _rng) -> Dict:
    """Gradient boosting on the same tabular features as the logistic model.

    Included because it beats the proposed architecture. On the binary
    "does the opponent respond" question it reaches AUROC 0.567 against the
    transformer's 0.530, with 0.474 on shuffled labels -- which is what an
    over-parameterised model looks like: 1.4M parameters fitted to 235
    positive examples. A result that loses to a gradient-boosted baseline
    has to report that baseline.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3, random_state=0)
    clf.fit(tabular_features(tr), _labels(tr))
    probs = np.full((len(te), N_FORMATIONS), 1e-9)
    probs[:, clf.classes_] = clf.predict_proba(tabular_features(te))
    probs /= probs.sum(axis=1, keepdims=True)
    return compute_metrics(probs, _labels(te)).as_row()


BASELINES = {
    "random": run_random,
    "majority": run_majority,
    "persistence (no change)": run_persistence,
    "transition matrix P(next|current)": run_transition_matrix,
    "logistic (state + style)": run_logistic,
    "gradient boosting (state + style)": run_gradient_boosting,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--triples", default=None)
    ap.add_argument("--n-synthetic", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.paths.ensure()

    if args.triples:
        df = validate_triples(pd.read_parquet(args.triples))
    else:
        from ..data.synthetic import generate_triples
        df, stats = generate_triples(args.n_synthetic, seed=cfg.seed,
                                     temperature=args.temperature,
                                     seq_len=cfg.encoder.seq_len)
        print(stats, "\n")

    rows: List[Dict] = []
    for seed in cfg.train.seeds:
        set_seed(int(seed))
        rng = np.random.default_rng(int(seed))
        gkf = GroupKFold(n_splits=cfg.train.n_folds)
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=df["match_id"].to_numpy())):
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            for name, fn in BASELINES.items():
                rows.append({"variant": name, "seed": int(seed), "fold": fold, **fn(tr, te, rng)})

    res = pd.DataFrame(rows)
    print("=== E1 baselines (mean +- std) ===")
    for name, g in res.groupby("variant", sort=False):
        print(f"{name:<26} Top-1 {g.top1.mean():.4f}+-{g.top1.std():.4f}   "
              f"Top-3 {g.top3.mean():.4f}+-{g.top3.std():.4f}   "
              f"ECE {g.ece.mean():.4f}")
    save_results(rows, Path(cfg.paths.results) / "table3_e1_baselines.csv")


if __name__ == "__main__":
    main()
