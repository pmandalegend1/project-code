"""
Vocabularies and formation geometry for RAMFP.

Everything the models need to turn StatsBomb-style categorical fields into
integer ids lives here, plus the one non-obvious piece of engineering in the
state encoder: turning a formation *code* (an int like 4231) into an
11-node graph with role labels and an adjacency matrix, automatically.

Design note (defend this in the paper)
--------------------------------------
The draft assumes a hand-built adjacency table per formation. That is 15
hand-written tables which nobody will maintain and which cannot generalise
to a formation the taxonomy has not seen. Instead we:

  1. Decode the formation code into lines, e.g. 4231 -> (4, 2, 3, 1).
  2. Place each of the 11 players at a canonical (x, y) on a 120x80 pitch:
     a line's depth comes from its index, a player's width from its position
     within the line.
  3. Assign each player a ROLE id from a fixed 6 x 5 grid
     (vertical band x lateral channel). This is the structural prior:
     two formations that share a back four produce the *same* four DEF-band
     role ids, so role embeddings are shared across formations without any
     hand-written taxonomy.
  4. Build intra-team adjacency by k-nearest-neighbour over those positions.

This gives contribution N5 (structural similarity inductive prior) for free
and is fully deterministic, so it can be unit-tested.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Pitch
# --------------------------------------------------------------------------
PITCH_X = 120.0
PITCH_Y = 80.0

# --------------------------------------------------------------------------
# Formation vocabulary
# --------------------------------------------------------------------------
# The 15 most frequent formation codes in StatsBomb open data. Replace this
# list with the real top-15 once 02_formation_timeline.py has run; the code
# below does not care what the entries are, only that they are valid codes.
# The vocabulary is derived from the corpus, not assumed.
#
# It held 15 codes and covered 90.8% of the 2138 decisions built from 2651
# StatsBomb matches.  The gap was two ordinary shapes that were simply
# absent: 4-1-2-1-2 (the midfield diamond, 151 occurrences) and 3-5-1-1
# (40).  Every one of those rows mapped to UNK_FORMATION_ID = |V|, which is
# one past the last output class, so cross_entropy raised "Target 15 is out
# of bounds" -- training could not start at all.
#
# Adding them takes coverage to 99.8%.  The five remaining stragglers
# (41221, 3142, 42211 -- five rows in total) stay UNK, which is correct:
# they are genuinely rare and an UNK input embedding is a fair description
# of them.  UNK is still never a prediction TARGET.
#
# Paper consequence, to be carried into the text: |V| = 17, so the full
# transition space is |V| x |V| x |I| = 17 x 17 x 6 = 1734 and the
# per-decision ceiling (f_src fixed by the game state) is |V| x |I| = 102.
FORMATIONS: Tuple[int, ...] = (
    442, 4231, 433, 4141, 4411, 4321, 4222, 451,
    4132, 352, 3412, 3421, 343, 532, 541,
    41212, 3511,
)
N_FORMATIONS = len(FORMATIONS)
FORMATION_TO_ID = {f: i for i, f in enumerate(FORMATIONS)}
UNK_FORMATION_ID = N_FORMATIONS  # id reserved for a code outside the vocab


def formation_id(code: int) -> int:
    """Map a StatsBomb formation code to a vocabulary id (UNK if unseen)."""
    return FORMATION_TO_ID.get(int(code), UNK_FORMATION_ID)


def decode_formation(code: int) -> Tuple[int, ...]:
    """4231 -> (4, 2, 3, 1). Raises if the outfield players do not sum to 10."""
    lines = tuple(int(d) for d in str(int(code)))
    if sum(lines) != 10:
        raise ValueError(f"formation {code} has {sum(lines)} outfield players, expected 10")
    return lines


# --------------------------------------------------------------------------
# Tactical intent
# --------------------------------------------------------------------------
INTENTS: Tuple[str, ...] = (
    "offensive", "defensive", "wide", "narrow", "pressing", "conservative",
)
N_INTENTS = len(INTENTS)
INTENT_TO_ID = {v: i for i, v in enumerate(INTENTS)}

# --------------------------------------------------------------------------
# Role vocabulary: vertical band x lateral channel
# --------------------------------------------------------------------------
BANDS: Tuple[str, ...] = ("GK", "DEF", "DM", "MID", "AM", "FWD")
LATERALS: Tuple[str, ...] = ("L", "LC", "C", "RC", "R")
N_BANDS, N_LATERALS = len(BANDS), len(LATERALS)
N_ROLES = N_BANDS * N_LATERALS  # 30


def role_id(band: int, lateral: int) -> int:
    return band * N_LATERALS + lateral


def role_name(rid: int) -> str:
    return f"{BANDS[rid // N_LATERALS]}-{LATERALS[rid % N_LATERALS]}"


def _band_for_line(line_idx: int, n_lines: int) -> int:
    """
    Map the i-th outfield line of an n-line formation onto one of the five
    outfield bands (DEF=1 .. FWD=5).

    The first line is always the defence and the last always the attack, so a
    4-4-2 and a 4-2-3-1 put their backlines in the same band and therefore
    share role embeddings. Intermediate lines spread monotonically over the
    midfield bands: one middle line becomes MID, two become DM and AM.
    """
    if line_idx == 0:
        return 1                       # DEF
    if line_idx == n_lines - 1:
        return N_BANDS - 1             # FWD
    n_mid = n_lines - 2
    m = line_idx - 1
    if n_mid <= 1:
        return 3                       # MID
    return int(2 + round(m / (n_mid - 1) * 2))   # DM .. AM


# Football-sensible lateral patterns. Line sizes in real formation codes are
# always 1..5, so these are exhaustive; anything larger falls back to a spread.
_LATERAL_PATTERNS = {
    1: (2,),
    2: (1, 3),
    3: (0, 2, 4),
    4: (0, 1, 3, 4),
    5: (0, 1, 2, 3, 4),
}


def _lateral_for_slot(slot: int, line_size: int) -> int:
    """Position within a line -> one of five lateral channels."""
    pattern = _LATERAL_PATTERNS.get(line_size)
    if pattern is not None:
        return pattern[slot]
    frac = slot / max(line_size - 1, 1)
    return int(round(frac * (N_LATERALS - 1)))


# --------------------------------------------------------------------------
# Formation geometry
# --------------------------------------------------------------------------
@lru_cache(maxsize=256)
def formation_layout(code: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Return (positions, roles) for a formation code.

    positions : float array [11, 2] of canonical (x, y) on a 120x80 pitch,
                attacking towards x = 120. Node 0 is always the keeper.
    roles     : tuple of 11 role ids.
    """
    lines = decode_formation(code)
    n_lines = len(lines)

    pos: List[Tuple[float, float]] = [(6.0, PITCH_Y / 2)]      # keeper
    roles: List[int] = [role_id(0, 2)]                          # GK-C

    for li, size in enumerate(lines):
        # depth: backline sits ~22m up, front line ~100m up
        x = 22.0 + (li + 1) / (n_lines + 1) * 78.0
        band = _band_for_line(li, n_lines)
        for slot in range(size):
            lat = _lateral_for_slot(slot, size)
            # lateral channel -> y, leaving a 10m margin either side
            y = 10.0 + lat / (N_LATERALS - 1) * (PITCH_Y - 20.0)
            pos.append((x, y))
            roles.append(role_id(band, lat))

    return np.asarray(pos, dtype=np.float32), tuple(roles)


@lru_cache(maxsize=256)
def intra_adjacency(code: int, k: int = 3) -> np.ndarray:
    """
    Symmetric [11, 11] bool adjacency: each node linked to its k nearest
    team-mates by canonical position, plus self-loops. Deterministic, and it
    works for any valid formation code including ones never seen in training.
    """
    pos, _ = formation_layout(code)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = np.zeros((11, 11), dtype=bool)
    for i in range(11):
        for j in np.argsort(d[i])[:k]:
            adj[i, j] = adj[j, i] = True
    np.fill_diagonal(adj, True)
    return adj


@lru_cache(maxsize=1024)
def inter_adjacency(code_a: int, code_b: int, k: int = 2) -> np.ndarray:
    """
    [11, 11] bool matchup matrix between team a (attacking x+) and team b.

    Team b is mirrored into a's frame (x -> 120 - x, y -> 80 - y) and every
    a-node is connected to its k nearest b-nodes. This is what the world-model
    teammate needs for E_inter; it lives here so both models share one
    definition of "who marks whom".
    """
    pa, _ = formation_layout(code_a)
    pb, _ = formation_layout(code_b)
    pb_mirror = np.stack([PITCH_X - pb[:, 0], PITCH_Y - pb[:, 1]], axis=-1)
    d = np.linalg.norm(pa[:, None, :] - pb_mirror[None, :, :], axis=-1)
    adj = np.zeros((11, 11), dtype=bool)
    for i in range(11):
        for j in np.argsort(d[i])[:k]:
            adj[i, j] = True
    return adj


def structural_distance(code_a: int, code_b: int) -> float:
    """
    L2 between the two formations' canonical shapes after optimal-ish
    role-wise pairing. Used by the stability critic (Delta_struct) and by the
    transition generator's grammar rules. Cheap and symmetric.
    """
    pa, _ = formation_layout(code_a)
    pb, _ = formation_layout(code_b)
    return float(np.linalg.norm(np.sort(pa, axis=0) - np.sort(pb, axis=0)))


def formation_family(code: int) -> str:
    """Coarse family label, e.g. 'back4-1line' — useful for stratified splits."""
    lines = decode_formation(code)
    return f"back{lines[0]}-{len(lines)}line"


def mirror_formation(code: int) -> int:
    """
    Lateral mirror. Formation codes describe line sizes, which are invariant
    under a left-right flip, so the code is unchanged — but the *zones* of the
    associated events are not. Kept as a function so the augmentation reads
    explicitly and so an asymmetric vocabulary could override it later.
    """
    return int(code)


# --------------------------------------------------------------------------
# Event vocabulary
# --------------------------------------------------------------------------
EVENT_TYPES: Tuple[str, ...] = (
    "Pass", "Ball Receipt*", "Carry", "Pressure", "Duel", "Interception",
    "Block", "Clearance", "Ball Recovery", "Dribble", "Dribbled Past",
    "Foul Committed", "Foul Won", "Shot", "Goal Keeper", "Miscontrol",
    "Dispossessed", "Substitution", "Tactical Shift", "Starting XI",
    "Half Start", "Half End", "Injury Stoppage", "Offside", "Own Goal Against",
    "Own Goal For", "Player On", "Player Off", "Referee Ball-Drop",
    "Shield", "50/50", "Error", "Bad Behaviour", "Camera On", "Camera off",
)
N_EVENT_TYPES = len(EVENT_TYPES)
EVENT_TYPE_TO_ID = {v: i for i, v in enumerate(EVENT_TYPES)}
UNK_EVENT_TYPE_ID = N_EVENT_TYPES
EVENT_TYPE_VOCAB = N_EVENT_TYPES + 1

# Defensive actions used in the PPDA numerator/denominator. Kept here so the
# data teammate and the encoder never disagree about the definition.
# NOTE: "Tackle" is not a StatsBomb event type; it is Duel with
# duel.type.name == "Tackle". Handle that in 03_ppda.py.
DEFENSIVE_ACTIONS = ("Pressure", "Duel", "Interception", "Block")

OUTCOMES: Tuple[str, ...] = (
    "none", "complete", "incomplete", "goal", "saved", "off_target",
    "won", "lost", "out",
)
N_OUTCOMES = len(OUTCOMES)
OUTCOME_TO_ID = {v: i for i, v in enumerate(OUTCOMES)}

# Pitch zones: a 6 x 4 grid over 120 x 80, plus one id for a missing location.
ZONE_NX, ZONE_NY = 6, 4
N_ZONES = ZONE_NX * ZONE_NY + 1
NO_ZONE_ID = ZONE_NX * ZONE_NY


def zone_id(x: float | None, y: float | None) -> int:
    """Map a StatsBomb location to a zone id. None-safe."""
    if x is None or y is None or (isinstance(x, float) and math.isnan(x)):
        return NO_ZONE_ID
    ix = min(int(x / PITCH_X * ZONE_NX), ZONE_NX - 1)
    iy = min(int(y / PITCH_Y * ZONE_NY), ZONE_NY - 1)
    return max(ix, 0) * ZONE_NY + max(iy, 0)


def mirror_zone(z: int) -> int:
    """Left-right flip of a zone id, for lateral mirror augmentation."""
    if z == NO_ZONE_ID:
        return z
    ix, iy = divmod(z, ZONE_NY)
    return ix * ZONE_NY + (ZONE_NY - 1 - iy)


# Minute buckets: six 15-minute windows plus one for stoppage beyond 90.
N_MINUTE_BUCKETS = 7


def minute_bucket(minute: int) -> int:
    return min(int(minute) // 15, N_MINUTE_BUCKETS - 1)


# Game phase, inferred from the last few events.
PHASES: Tuple[str, ...] = ("attacking", "defending", "transition")
N_PHASES = len(PHASES)
PHASE_TO_ID = {v: i for i, v in enumerate(PHASES)}

# Scoreline state, clipped to [-3, +3] -> 7 buckets.
N_SCORE_BUCKETS = 7


def score_bucket(goal_diff: int) -> int:
    return int(np.clip(goal_diff, -3, 3)) + 3


# PPDA buckets: quantile edges recomputed from real data in 03_ppda.py.
# These defaults are sane for StatsBomb-scale PPDA (roughly 5 to 25).
PPDA_EDGES = (7.0, 9.5, 12.0, 15.0, 20.0)
N_PPDA_BUCKETS = len(PPDA_EDGES) + 1


def ppda_bucket(ppda: float) -> int:
    return int(np.searchsorted(np.asarray(PPDA_EDGES), float(ppda)))


# Opponent pressing regime, produced by the style-clustering step.
#
# This was 4 with names ("high-press", "mid-block", "low-block", "counter"),
# which described a playing style the pipeline does not actually measure.
# The clustering runs on three PPDA summaries only (mean, std, slope);
# mean and std correlate at r = +0.69, and a silhouette sweep over the 46
# teams with >= 10 matches selects k = 2 (0.3906) with k = 4 the worst value
# in 2..7 (0.2943).  Four labels were splitting noise, so the construct is
# binary: high press vs low block.
#
# Everything downstream derives from this constant -- the style embedding,
# the one-hot encoder in data/datasets.py and the range check in
# data/schema.py -- so raising it again only requires changing it here,
# together with the clustering k.  Do that once the style features include
# non-PPDA behaviour (pass length, possession share, lateral spread,
# passes per possession, defensive line height).
N_STYLES = 2
STYLE_NAMES: Tuple[str, ...] = ("high-press", "low-block")


def summarise() -> str:
    return (
        f"formations={N_FORMATIONS}(+1 unk) intents={N_INTENTS} roles={N_ROLES} "
        f"event_types={EVENT_TYPE_VOCAB} zones={N_ZONES} outcomes={N_OUTCOMES} "
        f"styles={N_STYLES} phases={N_PHASES}"
    )


if __name__ == "__main__":  # quick self-check
    print(summarise())
    for c in FORMATIONS:
        pos, roles = formation_layout(c)
        assert pos.shape == (11, 2) and len(roles) == 11
        assert intra_adjacency(c).shape == (11, 11)
        print(f"{c:>5}  {decode_formation(c)!s:<16} roles={[role_name(r) for r in roles]}")
