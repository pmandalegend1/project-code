"""
formation_fallback.py -- deterministic graph spec for formations that are NOT
in the hand-specified taxonomy of canonical_roles.json.

Why this file exists
--------------------
canonical_roles.json hand-specifies 15 formations.  StatsBomb records more than
that, and the opponent-response vocabulary used downstream contains 17.  Three
codes that occur in the data (3412, 41212, 3511) have no entry, and
build_bipartite_graph used to raise ValueError on them.

We did NOT invent new hand entries.  We first tested whether the existing 15
follow any derivable rule, so that the same rule could be extended: they do
not.  Formations that share the same first two lines (e.g. 4141 / 4132,
442 / 4411, 433 / 4312 / 4321) carry DIFFERENT intra_edges lists, so the 15
are hand-authored per-formation judgements, not the output of a generator.
Extending them by rule would therefore have been guesswork dressed up as a
derivation.

Instead this module implements the fallback that canonical_roles.json already
documents but that graph_builder never wired up:

    "canonical_adjacency": {
        "description": "Default adjacency for any formation, by canonical role
                        idx. Used as fallback in graph_builder."}

Behaviour
---------
* A formation present in formation_mappings is untouched -- it keeps its exact
  hand-authored node_assignments and intra_edges.  No existing result changes.
* A formation absent from formation_mappings gets
    - node_assignments from the line-role templates below, every one of which
      is attested verbatim in at least one hand-specified formation, and
    - intra_edges by projecting canonical_adjacency (which is expressed in
      canonical-role indices) onto position indices.
* Every fallback use is recorded in FALLBACK_USED so the paper can report
  exactly which formations and how many windows took this path.

Reporting line for the paper: "N of the M formations observed were covered by
the hand-specified taxonomy; the remainder used the documented canonical
fallback adjacency."
"""

from __future__ import annotations

from collections import Counter

# canonical role indices, from canonical_roles.json
GK, RB, RCB, CB, LCB, LB, CDM, CM, CAM, W, ST = range(11)

NUM_POSITIONS = 11

# every template below appears verbatim in a hand-specified formation
_BACK = {
    3: [RCB, CB, LCB],                # 343, 352, 3421
    4: [RB, RCB, LCB, LB],            # 442, 433, 4231, ...
    5: [RCB, RB, CB, LB, LCB],        # 532, 541
}
_ATTACK = {
    1: [ST],                          # 451, 4141, 4231, ...
    2: [ST, ST],                      # 442, 532, 4222, ...
    3: [W, W, ST],                    # 343, 433
}

FALLBACK_USED: Counter = Counter()


def _midfield_line(size, is_first_mid, is_last_mid, back_size):
    """Roles for one midfield line.

    Flank slots of the FIRST midfield line are wing-backs when the side plays a
    back three (343, 352, 3421 all do this) and wide midfielders otherwise
    (442, 451, 4141, 4411, 541).
    Lines of one or two players are holding when they sit directly in front of
    the defence (4141, 4132, 4231, 4222) and attacking when they sit directly
    behind the forwards (4411, 4312, 4321, 3421, 4222).
    Lines of three or more are central-midfield lines in every hand-specified
    formation; the middle slot of an odd first midfield line is holding
    (451, 352, 532, 4321, 4312).
    """
    if size <= 2:
        if is_first_mid and not is_last_mid:
            return [CDM] * size            # 4141, 4132, 4231, 4222
        if is_last_mid:
            return [CAM] * size            # 4411, 4312, 4321, 3421
        return [CM] * size                 # interior line, e.g. 41212

    flank = RB if (is_first_mid and back_size == 3) else W
    other_flank = LB if (is_first_mid and back_size == 3) else W

    line = [CM] * size
    if is_first_mid and size % 2 == 1:
        line[size // 2] = CDM              # 451, 352, 532, 4321, 4312
    if size >= 4:
        line[0] = flank                    # 442, 4141, 4411, 541, 343, 352
        line[-1] = other_flank
    elif is_first_mid and back_size == 3:
        line[0], line[-1] = flank, other_flank
    return line


def derive_node_assignments(code):
    """Canonical role index for each of the 11 positions of `code`."""
    digits = [int(d) for d in str(int(code))]
    if sum(digits) != 10:
        raise ValueError(f"formation {code} does not describe 10 outfield players")
    back, attack, mids = digits[0], digits[-1], digits[1:-1]
    if back not in _BACK or attack not in _ATTACK:
        raise ValueError(f"formation {code}: unsupported back ({back}) or attack ({attack}) line")

    roles = [GK] + list(_BACK[back])
    for i, size in enumerate(mids):
        roles += _midfield_line(size, i == 0, i == len(mids) - 1, back)
    roles += _ATTACK[attack]

    assert len(roles) == NUM_POSITIONS, f"{code} -> {len(roles)} nodes"
    return roles


def derive_intra_edges(roles, canonical_adjacency):
    """Project role-level canonical_adjacency onto position indices.

    canonical_adjacency is expressed in canonical ROLE indices; intra_edges are
    expressed in position indices 0-10.  An edge between roles a and b becomes
    an edge between every position holding a and every position holding b.
    Positions that share a role (the two centre-forwards of a 442, the two
    centre-midfielders of a 433) are joined as well, which the hand-specified
    lists also do.
    """
    role_pos = {}
    for i, r in enumerate(roles):
        role_pos.setdefault(r, []).append(i)

    edges = set()
    for a, b in canonical_adjacency:
        for i in role_pos.get(a, []):
            for j in role_pos.get(b, []):
                if i != j:
                    edges.add((min(i, j), max(i, j)))
    for positions in role_pos.values():           # same-role pairs
        for i in positions:
            for j in positions:
                if i < j:
                    edges.add((i, j))

    _ensure_connected(edges, roles)
    return [list(e) for e in sorted(edges)]


def _ensure_connected(edges, roles):
    """Join any component that the projection left detached.

    A formation can omit a role that canonical_adjacency routes through (a side
    with no CDM loses the 1-6 / 5-6 / 6-7 edges), which would leave an isolated
    block.  Rather than silently emitting a disconnected graph -- which a GNN
    would read as 'these players never interact' -- we link the detached
    component to the nearest position index in the main component.  This is
    reported, not hidden: it only ever ADDS edges.
    """
    parent = list(range(NUM_POSITIONS))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in edges:
        union(i, j)

    while True:
        comps = {}
        for i in range(NUM_POSITIONS):
            comps.setdefault(find(i), []).append(i)
        if len(comps) == 1:
            return
        main = max(comps.values(), key=len)
        other = next(c for c in comps.values() if c is not main)
        i = min(other, key=lambda p: min(abs(p - m) for m in main))
        j = min(main, key=lambda m: abs(m - i))
        edges.add((min(i, j), max(i, j)))
        union(i, j)


def fallback_mapping(code, canon):
    """A formation_mappings-shaped dict for a formation with no hand entry."""
    roles = derive_node_assignments(code)
    adjacency = [tuple(e) for e in canon["canonical_adjacency"]["edges"]]
    FALLBACK_USED[str(int(code))] += 1
    return {
        "node_assignments": roles,
        "intra_edges": derive_intra_edges(roles, adjacency),
        "source": "canonical_fallback",
    }


def fallback_report():
    """One line per formation that used the fallback, for the paper/logs."""
    if not FALLBACK_USED:
        return "all formations covered by the hand-specified taxonomy"
    total = sum(FALLBACK_USED.values())
    detail = ", ".join(f"{k} x{v}" for k, v in sorted(FALLBACK_USED.items()))
    return f"canonical fallback used for {total} graphs across {len(FALLBACK_USED)} formations: {detail}"
