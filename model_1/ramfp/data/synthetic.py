"""
Synthetic StatsBomb-shaped data with a KNOWN ground-truth response rule.

Why this exists
---------------
The real triples come from steps 02-05, which other people on the team own.
Waiting for them costs a week you do not have. More importantly, when the real
data arrives and your model scores 34% Top-1, you will have no idea whether
that is a bug or the ceiling of a genuinely noisy problem.

So this module generates data from an *explicit* conditional distribution

    P(opp_response | f_tgt, opp_style, goal_diff, intent)

and reports the Bayes-optimal Top-1 and Top-3 accuracy — the score a perfect
model would get. A correct implementation should land close to that ceiling on
synthetic data. If it does not, the bug is in your model, not in the football.

The generated frames validate against ramfp.data.schema, so swapping in the
real triples.parquet is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..vocab import (
    EVENT_TYPE_TO_ID, FORMATIONS, INTENTS, N_MINUTE_BUCKETS, N_OUTCOMES,
    N_STYLES, N_ZONES, NO_ZONE_ID, PHASES, decode_formation, minute_bucket,
)

# --------------------------------------------------------------------------
# The ground-truth response rule
# --------------------------------------------------------------------------
# A team's response is modelled as a preference over formations driven by how
# *defensive* they want to be, which depends on their tactical style, the
# scoreline from their point of view, and what the opponent just did.
#
# defensiveness  ~  style_base + intent_effect + scoreline_effect + shape_effect
#
# We then score every formation by how well its backline and forward count
# match that target defensiveness, and softmax with a temperature. Temperature
# controls how much irreducible noise the problem has, i.e. the Bayes ceiling.

STYLE_BASE = {0: -0.9, 1: 0.0, 2: +1.1, 3: +0.4}   # high-press .. counter

INTENT_EFFECT = {
    "offensive":    +0.7,   # they saw an attack coming, so they sit deeper
    "defensive":    -0.5,   # opponent went safe, so they can push up
    "wide":         +0.2,
    "narrow":       -0.1,
    "pressing":     +0.5,
    "conservative": -0.6,
}


INTENT_WIDTH = {
    "offensive": 0.2, "defensive": -0.1, "wide": +1.0,
    "narrow": -1.0, "pressing": +0.3, "conservative": -0.3,
}


def _shape_stats(code: int) -> Tuple[int, int, int, int]:
    """(defenders, forwards, n_lines, widest_line) for a formation code."""
    lines = decode_formation(code)
    return lines[0], lines[-1], len(lines), max(lines)


def _traits(code: int) -> Tuple[float, float]:
    """
    Two latent traits of a shape: how defensive it is, and how wide it is.
    The number of lines is included so that 4-4-2 and 4-2-3-1 — which share a
    backline and a forward count — are not degenerate duplicates.
    """
    d, f, n_lines, widest = _shape_stats(code)
    defensiveness = 0.9 * (d - 4) - 0.7 * (f - 2) + 0.35 * (n_lines - 3)
    width = 0.8 * (widest - 4) - 0.3 * (n_lines - 3)
    return defensiveness, width


def response_logits(f_tgt: int, style: int, goal_diff_opp: int, intent: str,
                    temperature: float = 0.5) -> np.ndarray:
    """
    Ground-truth (unnormalised) log-probabilities over FORMATIONS for the
    opponent's response. `goal_diff_opp` is from the RESPONDING team's view.

    `temperature` sets how much irreducible noise the problem has, and hence
    the Bayes ceiling. Sweep it to check your model tracks the ceiling rather
    than a single operating point: 0.3 gives an easy problem (~60% Top-1),
    0.85 a hard one (~20%), and real football is probably nearer the hard end.
    """
    d_att, f_att, _, w_att = _shape_stats(f_tgt)
    target_def = (
        STYLE_BASE[int(style)]
        + INTENT_EFFECT[intent]
        - 0.45 * float(goal_diff_opp)     # losing -> more attacking
        + 0.30 * (f_att - 2)              # opponent added forwards -> sit deeper
        - 0.20 * (d_att - 4)
    )
    target_width = INTENT_WIDTH[intent] + 0.5 * (w_att - 4)

    scores = np.empty(len(FORMATIONS), dtype=np.float64)
    for i, code in enumerate(FORMATIONS):
        defensiveness, width = _traits(code)
        scores[i] = -((defensiveness - target_def) ** 2) - 0.6 * ((width - target_width) ** 2)
    return scores / temperature


def response_distribution(f_tgt: int, style: int, goal_diff_opp: int,
                          intent: str, temperature: float = 0.5) -> np.ndarray:
    z = response_logits(f_tgt, style, goal_diff_opp, intent, temperature)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


# --------------------------------------------------------------------------
# Event sequences
# --------------------------------------------------------------------------
# Pressing teams generate more Pressure/Duel/Interception events. The signal is
# deliberately weak and noisy so that explicitly conditioning on the style label
# still helps — otherwise the "no style conditioning" ablation would be
# uninformative, which would make the synthetic benchmark useless for E1.

_COMMON = ["Pass", "Ball Receipt*", "Carry", "Pressure", "Duel", "Interception",
           "Ball Recovery", "Dribble", "Clearance", "Block", "Shot", "Miscontrol"]
_COMMON_IDS = np.array([EVENT_TYPE_TO_ID[t] for t in _COMMON])
_PRESS_MASK = np.array([t in ("Pressure", "Duel", "Interception", "Block")
                        for t in _COMMON], dtype=bool)


def _event_type_probs(style: int, rng: np.random.Generator) -> np.ndarray:
    base = np.array([0.30, 0.26, 0.14, 0.09, 0.04, 0.03,
                     0.04, 0.03, 0.03, 0.02, 0.01, 0.01])
    press_lift = {0: 1.9, 1: 1.3, 2: 0.8, 3: 1.0}[int(style)]
    p = base.copy()
    p[_PRESS_MASK] *= press_lift
    p *= rng.uniform(0.85, 1.15, size=p.shape)     # match-to-match noise
    return p / p.sum()


def _sample_sequence(rng: np.random.Generator, style: int, minute: int,
                     length: int) -> Dict[str, List[int]]:
    probs = _event_type_probs(style, rng)
    types = rng.choice(_COMMON_IDS, size=length, p=probs)

    # possession runs rather than i.i.d. team flags
    team, teams = 0, []
    for _ in range(length):
        if rng.random() < 0.18:
            team = 1 - team
        teams.append(team)

    zones = rng.integers(0, N_ZONES - 1, size=length)
    zones[rng.random(length) < 0.05] = NO_ZONE_ID
    minutes = np.clip(
        minute_bucket(minute) - rng.integers(0, 2, size=length), 0, N_MINUTE_BUCKETS - 1)
    outcomes = rng.integers(0, N_OUTCOMES, size=length)

    return {
        "event_type": types.astype(np.int64).tolist(),
        "event_zone": zones.astype(np.int64).tolist(),
        "event_team": np.asarray(teams, dtype=np.int64).tolist(),
        "event_minute": minutes.astype(np.int64).tolist(),
        "event_outcome": outcomes.astype(np.int64).tolist(),
    }


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------
@dataclass
class SyntheticStats:
    n_triples: int
    bayes_top1: float
    bayes_top3: float
    majority_top1: float
    entropy: float

    def __str__(self) -> str:
        return (
            f"synthetic triples : {self.n_triples}\n"
            f"  majority-class Top-1 : {self.majority_top1:.4f}   <- beat this\n"
            f"  Bayes-optimal  Top-1 : {self.bayes_top1:.4f}   <- aim for this\n"
            f"  Bayes-optimal  Top-3 : {self.bayes_top3:.4f}\n"
            f"  mean label entropy   : {self.entropy:.3f} nats"
        )


def generate_triples(n: int = 1500, seed: int = 0, temperature: float = 0.5,
                     seq_len: int = 50) -> Tuple[pd.DataFrame, SyntheticStats]:
    """Generate a triples frame matching ramfp.data.schema.TRIPLE_COLUMNS."""
    rng = np.random.default_rng(seed)
    formations = np.asarray(FORMATIONS)

    n_matches = max(1, n // 4)
    match_ids = rng.integers(3_000_000, 3_999_999, size=n_matches)
    comp_ids = rng.choice([11, 43, 49, 2, 16], size=n_matches)
    team_pool = rng.integers(100, 999, size=max(20, n // 20))
    team_style = {int(t): int(rng.integers(0, N_STYLES)) for t in team_pool}

    rows, dists = [], []
    for i in range(n):
        m = int(rng.integers(0, n_matches))
        team = int(rng.choice(team_pool))
        opp = int(rng.choice(team_pool))
        while opp == team:
            opp = int(rng.choice(team_pool))

        f_src = int(rng.choice(formations))
        f_tgt = int(rng.choice(formations))
        while f_tgt == f_src:                      # step 02 filters these out
            f_tgt = int(rng.choice(formations))

        intent = str(rng.choice(INTENTS))
        minute = int(rng.integers(3, 91))
        goal_diff = int(np.clip(rng.normal(0, 1.2), -3, 3).round())
        style = team_style[opp]

        p = response_distribution(f_tgt, style, -goal_diff, intent, temperature)
        response = int(formations[rng.choice(len(formations), p=p)])
        dists.append(p)

        seq = _sample_sequence(rng, style, minute, seq_len)
        rows.append({
            "triple_id": i,
            "match_id": int(match_ids[m]),
            "competition_id": int(comp_ids[m]),
            "season_id": 90,
            "team_id": team,
            "opp_team_id": opp,
            "f_src": f_src,
            "f_tgt": f_tgt,
            "intent": intent,
            "minute": minute,
            "period": 1 if minute <= 45 else 2,
            "goal_diff": goal_diff,
            "is_home": int(rng.integers(0, 2)),
            "opp_formation": int(rng.choice(formations)),
            "opp_style": style,
            "opp_ppda_pre": float(np.clip(rng.normal(12 - 2.5 * (1 - style), 3), 3, 30)),
            "phase": str(rng.choice(PHASES)),
            "opp_response": response,
            **seq,
        })

    df = pd.DataFrame(rows)
    P = np.asarray(dists)
    counts = df["opp_response"].value_counts()
    stats = SyntheticStats(
        n_triples=len(df),
        bayes_top1=float(P.max(axis=1).mean()),
        bayes_top3=float(np.sort(P, axis=1)[:, -3:].sum(axis=1).mean()),
        majority_top1=float(counts.iloc[0] / len(df)),
        entropy=float((-P * np.log(P + 1e-12)).sum(axis=1).mean()),
    )
    return df, stats


def generate_sequences(n: int = 20_000, seed: int = 0, seq_len: int = 50) -> pd.DataFrame:
    """Unlabelled event sequences for encoder pretraining (step 07)."""
    rng = np.random.default_rng(seed + 7919)
    rows = []
    for i in range(n):
        style = int(rng.integers(0, N_STYLES))
        minute = int(rng.integers(0, 95))
        rows.append({"match_id": int(rng.integers(3_000_000, 3_999_999)),
                     **_sample_sequence(rng, style, minute, seq_len)})
    return pd.DataFrame(rows)
