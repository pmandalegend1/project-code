"""
The data contract between this module and the data/statistics teammates.

Nothing in ramfp/ reads raw StatsBomb JSON. It reads exactly two parquet
files, whose columns are specified here. Hand this file to whoever owns
01_build_event_store.py .. 05_triples.py — if their output validates against
`validate_triples` and `validate_sequences`, integration is a no-op.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

# --------------------------------------------------------------------------
# triples.parquet  — one row per natural-experiment triple (step 05)
# --------------------------------------------------------------------------
TRIPLE_COLUMNS: dict[str, str] = {
    # identifiers -----------------------------------------------------------
    "triple_id":      "int64   unique row id",
    "match_id":       "int64   StatsBomb match id (used for grouped CV splits)",
    "competition_id": "int64   used to stratify the test split",
    "season_id":      "int64   used with team_id to key the style cluster",
    "team_id":        "int64   team that executed the formation change (team a)",
    "opp_team_id":    "int64   the responding team (team b)",
    # the intervention ------------------------------------------------------
    "f_src":          "int64   formation code before the change, e.g. 442",
    "f_tgt":          "int64   formation code after the change, e.g. 4231",
    "intent":         "str     one of ramfp.vocab.INTENTS",
    # game state at the moment of change -----------------------------------
    "minute":         "int64   0..120",
    "period":         "int64   1 or 2 (extra time excluded upstream)",
    "goal_diff":      "int64   team a's goals minus team b's, at time t",
    "is_home":        "int64   1 if team a is at home",
    "opp_formation":  "int64   team b's formation at time t",
    "opp_style":      "int64   team b's style cluster, 0..3 (step 04)",
    "opp_ppda_pre":   "float64 team b's PPDA in the 5-minute before-window",
    "phase":          "str     attacking | defending | transition, from last 10 events",
    # the label -------------------------------------------------------------
    "opp_response":   "int64   team b's formation in the after-window (the target)",
    # event context ---------------------------------------------------------
    "event_type":     "object  list[int]  last N event-type ids, oldest first",
    "event_zone":     "object  list[int]  matching zone ids",
    "event_team":     "object  list[int]  0 = team a acted, 1 = team b acted",
    "event_minute":   "object  list[int]  matching minute buckets",
    "event_outcome":  "object  list[int]  matching outcome ids",
}

REQUIRED_TRIPLE_COLUMNS: List[str] = list(TRIPLE_COLUMNS)

SEQUENCE_COLUMNS: dict[str, str] = {
    "match_id":      "int64   source match",
    "event_type":    "object  list[int] event-type ids, oldest first",
    "event_zone":    "object  list[int]",
    "event_team":    "object  list[int]",
    "event_minute":  "object  list[int]",
    "event_outcome": "object  list[int]",
}
REQUIRED_SEQUENCE_COLUMNS: List[str] = list(SEQUENCE_COLUMNS)

_LIST_COLUMNS = ("event_type", "event_zone", "event_team", "event_minute", "event_outcome")


class SchemaError(ValueError):
    pass


def _check_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} is missing columns: {missing}")


def _check_list_lengths(df: pd.DataFrame, name: str) -> None:
    lens = {c: df[c].map(len) for c in _LIST_COLUMNS if c in df.columns}
    if not lens:
        return
    ref_name, ref = next(iter(lens.items()))
    for c, s in lens.items():
        if not (s.to_numpy() == ref.to_numpy()).all():
            raise SchemaError(f"{name}: '{c}' and '{ref_name}' have different lengths on some rows")


def validate_triples(df: pd.DataFrame) -> pd.DataFrame:
    """Raise SchemaError on anything that would silently corrupt training."""
    from ..vocab import INTENT_TO_ID, N_STYLES, PHASE_TO_ID

    _check_columns(df, REQUIRED_TRIPLE_COLUMNS, "triples")
    _check_list_lengths(df, "triples")

    if df["triple_id"].duplicated().any():
        raise SchemaError("triples: triple_id is not unique")
    bad_intent = set(df["intent"].unique()) - set(INTENT_TO_ID)
    if bad_intent:
        raise SchemaError(f"triples: unknown intent labels {sorted(bad_intent)}")
    bad_phase = set(df["phase"].unique()) - set(PHASE_TO_ID)
    if bad_phase:
        raise SchemaError(f"triples: unknown phase labels {sorted(bad_phase)}")
    if not df["opp_style"].between(0, N_STYLES - 1).all():
        raise SchemaError(f"triples: opp_style outside 0..{N_STYLES - 1}")

    # The label must be a real class. formation_id maps an unknown code to
    # UNK_FORMATION_ID = |V|, which is one past the last output unit, so an
    # out-of-vocab opp_response surfaces much later as an IndexError from
    # cross_entropy or from the metrics -- far from the file that caused it.
    from ..vocab import FORMATIONS
    vocab = set(int(c) for c in FORMATIONS)
    bad = sorted(set(int(c) for c in df["opp_response"]) - vocab)
    if bad:
        raise SchemaError(
            f"triples: opp_response contains codes outside the formation "
            f"vocabulary: {bad}. Either add them to ramfp.vocab.FORMATIONS "
            f"or drop those rows when building the triples."
        )
    if (df["f_src"] == df["f_tgt"]).any():
        n = int((df["f_src"] == df["f_tgt"]).sum())
        raise SchemaError(
            f"triples: {n} rows have f_src == f_tgt. These are personnel changes "
            "logged as Tactical Shift, not shape changes — filter them in step 02."
        )
    return df


def validate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, REQUIRED_SEQUENCE_COLUMNS, "sequences")
    _check_list_lengths(df, "sequences")
    return df


def describe() -> str:
    lines = ["triples.parquet", "-" * 60]
    lines += [f"  {k:<16} {v}" for k, v in TRIPLE_COLUMNS.items()]
    lines += ["", "sequences.parquet", "-" * 60]
    lines += [f"  {k:<16} {v}" for k, v in SEQUENCE_COLUMNS.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
