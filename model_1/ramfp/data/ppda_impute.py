"""
ppda_impute.py -- leak-free imputation of `opp_ppda_pre`.

The problem (team-lead audit, "PPDA global-median imputation")
---------------------------------------------------------------
`opp_ppda_pre` is the opponent's PPDA in the 5-minute window before our
formation change. When the opponent made no defensive action in that window
the ratio is undefined; that happens on 380 of 2,133 decisions.

build_triples.py used to fill those rows from two sources:

  1. the opponent's FULL-MATCH PPDA from match_ppda.csv. That number is
     measured over the whole match, including the minutes AFTER the change we
     are trying to predict the response to -- post-treatment information. And
     because match_ppda.csv has no game_id, the lookup fell back to the median
     over every match that team played, including later ones and matches in
     the evaluation fold.
  2. otherwise, a median over every experiment in the corpus, again including
     the evaluation fold.

Both leak information the model should not have at decision time.

The fix
-------
build_triples.py now leaves the value missing (NaN) and keeps the
`opp_ppda_pre_imputed` flag. Imputation happens at use time, fitted on
training rows only, from OBSERVED pre-change windows of other decisions:

  1. median over observed training rows for the same opponent team
  2. else median over observed training rows with the same opponent style
  3. else median over all observed training rows

Every value used is a pre-change 5-minute window, so nothing post-treatment,
and nothing from the rows being evaluated.

Triples files built before this fix carry already-filled values; `as_missing`
turns every flagged row back into NaN, so old files are handled the same way
without a rebuild.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

FLAG = "opp_ppda_pre_imputed"
COL = "opp_ppda_pre"
SOURCE = "opp_ppda_pre_source"


def as_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with every imputed (or undefined) opp_ppda_pre set to NaN."""
    out = df.copy()
    out[COL] = pd.to_numeric(out[COL], errors="coerce").astype(float)
    if FLAG in out.columns:
        out.loc[out[FLAG].astype(bool).to_numpy(), COL] = np.nan
    return out


class PPDAImputer:
    """Fit on training rows only; fill missing opp_ppda_pre elsewhere."""

    def __init__(self, team_col: str = "opp_team_id", style_col: str = "opp_style"):
        self.team_col, self.style_col = team_col, style_col
        self.by_team: dict = {}
        self.by_style: dict = {}
        self.global_: Optional[float] = None

    def fit(self, train_df: pd.DataFrame) -> "PPDAImputer":
        obs = as_missing(train_df)
        obs = obs[obs[COL].notna()]
        if len(obs) == 0:
            raise ValueError("no observed opp_ppda_pre in the training rows")
        if self.team_col in obs:
            self.by_team = obs.groupby(self.team_col)[COL].median().to_dict()
        if self.style_col in obs:
            self.by_style = obs.groupby(self.style_col)[COL].median().to_dict()
        self.global_ = float(obs[COL].median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_ is None:
            raise RuntimeError("PPDAImputer.transform called before fit")
        out = as_missing(df)
        src = np.where(out[COL].notna(), "observed", "").astype(object)
        vals = out[COL].to_numpy(dtype=float).copy()
        for i in np.where(np.isnan(vals))[0]:
            row = out.iloc[i]
            t = row.get(self.team_col)
            s = row.get(self.style_col)
            if t in self.by_team:
                vals[i], src[i] = self.by_team[t], "train_team_median"
            elif s in self.by_style:
                vals[i], src[i] = self.by_style[s], "train_style_median"
            else:
                vals[i], src[i] = self.global_, "train_global_median"
        out[COL] = vals
        out[SOURCE] = src
        return out


def impute_out_of_fold(df: pd.DataFrame, n_folds: int, group_col: str = "match_id") -> pd.DataFrame:
    """Fill every row using statistics from the OTHER folds only.

    Uses the same GroupKFold(n_folds) by match as train_opponent.py, so a row in
    fold k gets exactly the value fold k's model saw for it in testing. Row
    order and index are preserved.
    """
    out = as_missing(df)
    filled = out.copy()
    filled[SOURCE] = ""
    for tr_idx, te_idx in GroupKFold(n_splits=n_folds).split(out, groups=out[group_col].to_numpy()):
        part = PPDAImputer().fit(out.iloc[tr_idx]).transform(out.iloc[te_idx])
        filled.iloc[te_idx, filled.columns.get_loc(COL)] = part[COL].to_numpy()
        filled.iloc[te_idx, filled.columns.get_loc(SOURCE)] = part[SOURCE].to_numpy()
    return filled
