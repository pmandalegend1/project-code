"""
Datasets and collation.

Every categorical id stored in a tensor is offset by +1 so that 0 can mean
PAD everywhere. `ramfp.models.state_encoder.PAD_ID` is that 0. Offsetting in
one place — here — is what stops the classic "embedding index off by one"
bug that silently costs a few accuracy points and is invisible in the loss.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..vocab import (
    INTENT_TO_ID, N_LATERALS, PHASE_TO_ID, formation_id, formation_layout,
    intra_adjacency, mirror_zone, ppda_bucket, score_bucket,
)

OFFSET = 1  # every id shifts by this so that 0 is PAD


def _mirror_role(rid: int) -> int:
    band, lat = divmod(rid, N_LATERALS)
    return band * N_LATERALS + (N_LATERALS - 1 - lat)


def _pad(seq: Sequence[int], length: int) -> np.ndarray:
    """Right-pad (or right-truncate to the most recent `length` events)."""
    arr = np.asarray(seq, dtype=np.int64)[-length:] + OFFSET
    if arr.size < length:
        arr = np.concatenate([arr, np.zeros(length - arr.size, dtype=np.int64)])
    return arr


class SequenceDataset(Dataset):
    """Unlabelled event sequences for encoder pretraining."""

    FIELDS = ("event_type", "event_zone", "event_team", "event_minute", "event_outcome")

    def __init__(self, df: pd.DataFrame, seq_len: int = 50, mirror: bool = False):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.mirror = mirror

    def __len__(self) -> int:
        return len(self.df)

    def _sequence(self, row: pd.Series, flip: bool) -> Dict[str, torch.Tensor]:
        zones = list(row["event_zone"])
        if flip:
            zones = [mirror_zone(z) for z in zones]
        raw = {
            "event_type": row["event_type"], "event_zone": zones,
            "event_team": row["event_team"], "event_minute": row["event_minute"],
            "event_outcome": row["event_outcome"],
        }
        out = {k: torch.from_numpy(_pad(v, self.seq_len)) for k, v in raw.items()}
        out["length"] = torch.tensor(
            min(len(row["event_type"]), self.seq_len), dtype=torch.long)
        return out

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[i]
        flip = self.mirror and bool(np.random.rand() < 0.5)
        return self._sequence(row, flip)


class TripleDataset(SequenceDataset):
    """Natural-experiment triples for the opponent co-behavior model."""

    def __init__(self, df: pd.DataFrame, seq_len: int = 50, mirror: bool = False,
                 knn: int = 3):
        super().__init__(df, seq_len=seq_len, mirror=mirror)
        self.knn = knn

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[i]
        flip = self.mirror and bool(np.random.rand() < 0.5)
        item = self._sequence(row, flip)

        # --- current shape of the team that is about to change -------------
        f_src = int(row["f_src"])
        _, roles = formation_layout(f_src)
        adj = intra_adjacency(f_src, self.knn)
        roles_arr = np.asarray(
            [_mirror_role(r) for r in roles] if flip else roles, dtype=np.int64)

        item["roles"] = torch.from_numpy(roles_arr)
        item["adj"] = torch.from_numpy(adj.copy())
        item["ppda_bucket"] = torch.tensor(
            ppda_bucket(float(row["opp_ppda_pre"])), dtype=torch.long)
        item["score_bucket"] = torch.tensor(
            score_bucket(int(row["goal_diff"])), dtype=torch.long)
        item["phase"] = torch.tensor(PHASE_TO_ID[row["phase"]], dtype=torch.long)

        # --- the proposed transition ---------------------------------------
        item["f_src"] = torch.tensor(formation_id(f_src), dtype=torch.long)
        item["f_tgt"] = torch.tensor(formation_id(int(row["f_tgt"])), dtype=torch.long)
        item["intent"] = torch.tensor(INTENT_TO_ID[row["intent"]], dtype=torch.long)
        item["style"] = torch.tensor(int(row["opp_style"]), dtype=torch.long)

        # --- the opponent's CURRENT shape -----------------------------------
        # Needed for two things the single-head model never asked for: the
        # persistence baseline ("they keep this shape"), and the mixture in
        # predict_proba, which parks the no-change probability mass here.
        opp_formation_id = formation_id(int(row["opp_formation"]))
        item["opp_formation_id"] = torch.tensor(
            opp_formation_id, dtype=torch.long)

        # --- labels ----------------------------------------------------------
        target = formation_id(int(row["opp_response"]))
        item["target"] = torch.tensor(target, dtype=torch.long)

        # Did the shape actually move?  Note this is NOT the same as the
        # pipeline's opponent_response_is_shift: on 1200 matches 125 decisions
        # carried a Tactical Shift but only 39 changed the formation -- the
        # other 86 were personnel swaps inside the same shape.  Training on
        # the shift flag would label 86 non-events as responses.
        item["changed"] = torch.tensor(
            float(target != opp_formation_id), dtype=torch.float32)

        # kept for grouped splits and error analysis, never fed to the model
        item["match_id"] = torch.tensor(int(row["match_id"]), dtype=torch.long)
        return item


def collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def tabular_features(df: pd.DataFrame) -> np.ndarray:
    """
    Flat feature matrix for the logistic-regression baseline (E1 baseline 3):
    game state and style only, deliberately no transition input.
    """
    from ..vocab import N_STYLES

    style = np.eye(N_STYLES)[df["opp_style"].to_numpy()]
    return np.column_stack([
        style,
        df["goal_diff"].to_numpy(),
        df["minute"].to_numpy() / 90.0,
        df["is_home"].to_numpy(),
        df["opp_ppda_pre"].to_numpy(),
        df["phase"].map(PHASE_TO_ID).to_numpy(),
    ]).astype(np.float32)
