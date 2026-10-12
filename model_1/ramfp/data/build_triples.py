"""
Build the RAMFP training triples from the data pipeline's output.

    python -m ramfp.data.build_triples \
        --open-data   ~/Downloads/preprocessing/open-data \
        --experiments ~/Downloads/preprocessing/processed-data/natural_experiments.csv \
        --clusters    ~/Downloads/preprocessing/processed-data/team_style_clusters.csv \
        --match-ppda  ~/Downloads/preprocessing/processed-data/match_ppda.csv \
        --out         data/processed/triples.parquet

What this bridges
-----------------
natural_experiments.csv has ONE LINE per formation change: what changed, when,
and what the opponent did.  The state encoder needs something that line does
not carry -- the run of play immediately before the change.  This script goes
back to the raw StatsBomb events, lifts the last N events before each change,
maps them through ramfp.vocab, and joins the pressing-regime label and the
opponent's own formation, producing the schema in ramfp/data/schema.py.

Alignment with the pipeline
---------------------------
Event times are reconstructed exactly as src/data/preprocessing.py does them:
the StatsBomb timestamp restarts every period, so

    event_seconds = timestamp_within_period + PERIOD_OFFSETS[period]

If that ever drifts from the pipeline, every sequence would be cut at the
wrong moment while still looking perfectly well-formed, so the offsets are
duplicated here deliberately rather than imported -- and --self-check
verifies the reconstruction against the change times in the experiments file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..vocab import (
        EVENT_TYPE_TO_ID, INTENTS, N_STYLES, NO_ZONE_ID, OUTCOME_TO_ID,
        PHASES, UNK_EVENT_TYPE_ID, decode_formation, formation_layout,
        minute_bucket, zone_id,
    )
except ImportError:                                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ramfp.vocab import (                          # noqa: E402
        EVENT_TYPE_TO_ID, INTENTS, N_STYLES, NO_ZONE_ID, OUTCOME_TO_ID,
        PHASES, UNK_EVENT_TYPE_ID, decode_formation, formation_layout,
        minute_bucket, zone_id,
    )

# Must match src/data/preprocessing.py.  Period 5 is the shootout and is not
# continuous match-play time, so it is absent and those events are dropped.
PERIOD_OFFSETS = {1: 0.0, 2: 2700.0, 3: 5400.0, 4: 6300.0}

try:
    from ..vocab import FORMATIONS as _F
except ImportError:
    from ramfp.vocab import FORMATIONS as _F
_VOCAB = frozenset(int(c) for c in _F)


# --------------------------------------------------------------------- time
def event_seconds(ev: dict) -> float | None:
    period = ev.get("period")
    if period not in PERIOD_OFFSETS:
        return None
    ts = ev.get("timestamp")
    if isinstance(ts, str):
        try:
            h, m, s = ts.split(":")
            within = int(h) * 3600 + int(m) * 60 + float(s)
        except ValueError:
            within = None
    else:
        within = None
    if within is None:                                 # same fallback order
        minute, second = ev.get("minute"), ev.get("second")
        if minute is None or second is None:
            return None
        within = float(minute) * 60.0 + float(second)
    return within + PERIOD_OFFSETS[period]


# ------------------------------------------------------------------ outcome
_OUTCOME_MAP = {
    "goal": "goal",
    "saved": "saved", "saved off target": "saved", "saved to post": "saved",
    "off t": "off_target", "wayward": "off_target", "post": "off_target",
    "blocked": "off_target",
    "complete": "complete", "success": "won", "success in play": "won",
    "success out": "won", "won": "won",
    "incomplete": "incomplete", "lost": "lost", "lost in play": "lost",
    "aerial lost": "lost", "pass offside": "incomplete",
    "out": "out", "lost out": "out", "injury clearance": "out",
    "unknown": "none",
}

# Event types whose ABSENT outcome field means the action succeeded.  For
# everything else an absent outcome carries no information and maps to "none".
_COMPLETE_BY_DEFAULT = {"Pass", "Ball Receipt*", "Dribble", "Carry"}

_OUTCOME_KEYS = (
    "pass", "shot", "duel", "dribble", "ball_receipt", "interception",
    "50_50", "substitution", "goalkeeper",
)


def outcome_id(ev: dict) -> int:
    name = None
    for key in _OUTCOME_KEYS:
        block = ev.get(key)
        if isinstance(block, dict):
            outcome = block.get("outcome")
            if isinstance(outcome, dict) and outcome.get("name"):
                name = str(outcome["name"]).strip().lower()
                break
    if name is None:
        default = ("complete"
                   if ev["type"]["name"] in _COMPLETE_BY_DEFAULT else "none")
        return OUTCOME_TO_ID[default]
    return OUTCOME_TO_ID.get(_OUTCOME_MAP.get(name, "none"),
                             OUTCOME_TO_ID["none"])


# -------------------------------------------------------------------- shape
def _shape_axes(code: int) -> tuple[float, float, float]:
    """(forwardness, width, defensive line height) from the canonical layout.

    Read off ramfp.vocab.formation_layout, so the intent labels and the
    formation graph the encoder sees are derived from the same geometry.
    """
    pos, _ = formation_layout(code)
    outfield = pos[1:]                                  # drop the keeper
    forwardness = float(outfield[:, 0].mean())
    width = float(outfield[:, 1].std())
    n_def = decode_formation(code)[0]
    line = float(pos[1:1 + n_def, 0].mean())
    return forwardness, width, line


def derive_intent(f_src: int, f_tgt: int) -> str:
    """Label a transition on the axis it moves most.

    Intent is NOT observed in StatsBomb -- no event says why a coach changed
    shape.  It is derived here from three geometric axes, each giving an
    opposed pair of the six vocabulary intents:

        forwardness  up -> offensive     down -> conservative
        width        up -> wide          down -> narrow
        line height  up -> pressing      down -> defensive

    Each delta is scaled by the spread of that axis across the whole
    formation vocabulary, so the three are comparable, and the largest wins.
    Treat this column as a coarse proxy: it is reproducible and it is honest
    about its own basis, but it is not a record of the coach's reasoning.
    """
    a, b = _shape_axes(f_src), _shape_axes(f_tgt)
    deltas = np.array(b) - np.array(a)
    scale = derive_intent._scale
    scaled = np.divide(deltas, scale, out=np.zeros(3), where=scale > 0)
    axis = int(np.argmax(np.abs(scaled)))
    if abs(scaled[axis]) < 1e-9:
        return "conservative"
    pairs = (("offensive", "conservative"),
             ("wide", "narrow"),
             ("pressing", "defensive"))
    return pairs[axis][0 if scaled[axis] > 0 else 1]


def _init_intent_scale() -> None:
    from ..vocab import FORMATIONS  # noqa: PLC0415
    axes = np.array([_shape_axes(c) for c in FORMATIONS])
    derive_intent._scale = axes.std(axis=0)


try:
    _init_intent_scale()
except ImportError:
    from ramfp.vocab import FORMATIONS                  # noqa: E402
    derive_intent._scale = np.array(
        [_shape_axes(c) for c in FORMATIONS]
    ).std(axis=0)


# -------------------------------------------------------------------- phase
_ON_BALL = {"Pass", "Carry", "Dribble", "Shot", "Ball Receipt*"}


def derive_phase(window: list[dict], team_id: int) -> str:
    """attacking / defending / transition from who is on the ball.

    Settled possession one way or the other is attacking or defending; a
    window where neither side holds the ball is a transition.  The 0.65 / 0.35
    cut is a judgement call, recorded here rather than buried.
    """
    on_ball = [e for e in window if e["type"]["name"] in _ON_BALL]
    if not on_ball:
        return "transition"
    share = sum(1 for e in on_ball if e["team"]["id"] == team_id) / len(on_ball)
    if share >= 0.65:
        return "attacking"
    if share <= 0.35:
        return "defending"
    return "transition"


# ----------------------------------------------------------------- the build
def load_match(path: Path) -> list[dict] | None:
    try:
        events = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    out = []
    for ev in events:
        secs = event_seconds(ev)
        if secs is None or "team" not in ev or "type" not in ev:
            continue
        ev["_secs"] = secs
        out.append(ev)
    out.sort(key=lambda e: e["_secs"])
    return out


def formation_at(events: list[dict], team_id: int, t: float) -> int | None:
    """The team's formation as of time t, from Starting XI + Tactical Shift."""
    current = None
    for ev in events:
        if ev["_secs"] > t:
            break
        if ev["team"]["id"] != team_id:
            continue
        if ev["type"]["name"] not in ("Starting XI", "Tactical Shift"):
            continue
        tactics = ev.get("tactics") or {}
        if tactics.get("formation") is not None:
            current = int(tactics["formation"])
    return current


def build(args) -> int:
    exp = pd.read_csv(args.experiments)
    exp = exp[exp["period_id"].isin((1, 2))].copy()     # schema: period 1 or 2
    print(f"experiments (periods 1-2): {len(exp)}")

    clusters = pd.read_csv(args.clusters)
    style_of = dict(zip(clusters["team_id"].astype(int),
                        clusters["style_cluster"].astype(int)))
    extrapolated = dict(zip(
        clusters["team_id"].astype(int),
        clusters.get("style_label_is_extrapolated",
                     pd.Series(False, index=clusters.index)).astype(bool)))

    # match-level PPDA, used only to fill an undefined 5-minute window
    match_ppda = {}
    if args.match_ppda and Path(args.match_ppda).exists():
        mp = pd.read_csv(args.match_ppda)
        if "game_id" in mp.columns:
            key = zip(mp["game_id"].astype(int),
                      mp["pressing_team_id"].astype(int), mp["ppda"])
        else:                     # match_ppda.csv has no game_id column
            key = zip([-1] * len(mp), mp["pressing_team_id"].astype(int),
                      mp["ppda"])
        for g, t, v in key:
            if pd.notna(v):
                match_ppda.setdefault((g, t), []).append(float(v))
    ppda_fallback = float(
        pd.to_numeric(exp["before_ppda"], errors="coerce").median()
    )

    # home team per match
    home_of: dict[int, int] = {}
    for (cid, sid) in exp[["competition_id", "season_id"]].drop_duplicates().values:
        p = Path(args.open_data) / "data" / "matches" / str(int(cid)) / f"{int(sid)}.json"
        if not p.exists():
            continue
        for m in json.loads(p.read_text()):
            # StatsBomb names this "home_team_id" inside the home_team block,
            # not "id"; accept either so a schema tweak upstream cannot
            # silently turn every is_home into 0.
            block = m.get("home_team") or {}
            hid = block.get("home_team_id", block.get("id"))
            if hid is None:
                continue
            home_of[int(m["match_id"])] = int(hid)

    rows, cache, stats = [], {}, Counter()
    drift_max = 0.0

    for _, r in exp.sort_values("game_id").iterrows():
        gid = int(r["game_id"])
        if gid not in cache:
            cache.clear()                               # one match in memory
            cache[gid] = load_match(
                Path(args.open_data) / "data" / "events" / f"{gid}.json"
            )
        events = cache[gid]
        if not events:
            stats["no_events_file"] += 1
            continue

        t = float(r["change_time"])
        team_id = int(r["team_id"])
        opp_id = int(r["opponent_team_id"])

        # self-check: the change itself must exist at this time in the raw JSON
        near = [e for e in events
                if e["type"]["name"] == "Tactical Shift"
                and e["team"]["id"] == team_id
                and abs(e["_secs"] - t) < 2.0]
        if near:
            drift_max = max(drift_max, min(abs(e["_secs"] - t) for e in near))
        else:
            stats["change_not_found_in_raw"] += 1

        window = [e for e in events if e["_secs"] < t][-args.seq_len:]
        if len(window) < args.min_events:
            stats["too_few_prior_events"] += 1
            continue

        opp_formation = formation_at(events, opp_id, t)
        if opp_formation is None:
            stats["no_opponent_formation"] += 1
            continue

        style = style_of.get(opp_id)
        if style is None:
            stats["no_style_cluster"] += 1
            continue
        if not 0 <= style < N_STYLES:
            stats["style_out_of_range"] += 1
            continue

        ppda = pd.to_numeric(pd.Series([r["before_ppda"]]),
                             errors="coerce").iloc[0]
        imputed = False
        if pd.isna(ppda):
            vals = match_ppda.get((gid, opp_id)) or match_ppda.get((-1, opp_id))
            ppda = float(np.median(vals)) if vals else ppda_fallback
            imputed = True

        f_src, f_tgt = int(r["old_formation_code"]), int(r["new_formation_code"])

        # The TARGET has to be a class the output layer actually has.
        # f_src, f_tgt and opp_formation may fall outside the vocabulary --
        # they are inputs, and UNK_FORMATION_ID is an honest embedding for a
        # shape we have not catalogued.  opp_response cannot: formation_id
        # would return |V|, one past the last output class, and both
        # cross_entropy and the metrics would index out of bounds.  Five rows
        # in 2138 are lost this way (41221, 3142, 42211).
        opp_response = int(r["opponent_response_formation_code"])
        if opp_response not in _VOCAB:
            stats["response_outside_vocab"] += 1
            continue

        rows.append({
            "triple_id": len(rows),
            "match_id": gid,
            "competition_id": int(r["competition_id"]),
            "season_id": int(r["season_id"]),
            "team_id": team_id,
            "opp_team_id": opp_id,
            "f_src": f_src,
            "f_tgt": f_tgt,
            "intent": derive_intent(f_src, f_tgt),
            "minute": int(min(max(t / 60.0, 0), 120)),
            "period": int(r["period_id"]),
            "goal_diff": int(r["score_difference"]),
            "is_home": int(home_of.get(gid, -1) == team_id),
            "opp_formation": opp_formation,
            "opp_style": style,
            "opp_ppda_pre": float(ppda),
            "phase": derive_phase(window[-10:], team_id),
            "opp_response": opp_response,
            "event_type": [EVENT_TYPE_TO_ID.get(e["type"]["name"],
                                                UNK_EVENT_TYPE_ID)
                           for e in window],
            "event_zone": [zone_id(*(e.get("location") or (None, None))[:2])
                           if e.get("location") else NO_ZONE_ID
                           for e in window],
            "event_team": [0 if e["team"]["id"] == team_id else 1
                           for e in window],
            "event_minute": [minute_bucket(e.get("minute") or 0)
                             for e in window],
            "event_outcome": [outcome_id(e) for e in window],
            # provenance, not part of the required schema
            "opp_ppda_pre_imputed": imputed,
            "opp_style_is_extrapolated": bool(extrapolated.get(opp_id, False)),
            "opponent_response_is_shift": bool(
                str(r["opponent_response_is_shift"]).strip().lower() == "true"
            ),
            "response_window_complete": bool(
                str(r.get("response_window_complete", True)
                    ).strip().lower() == "true"
            ),
        })

    df = pd.DataFrame(rows)
    print(f"\ntriples built: {len(df)}")
    if stats:
        print("dropped:")
        for k, v in stats.most_common():
            print(f"   {k:<28} {v}")
    print(f"\nmax |raw - pipeline| change-time drift: {drift_max:.3f}s "
          f"(must be < 2s; a larger value means the two time "
          f"reconstructions disagree)")

    if df.empty:
        print("\nNOTHING BUILT -- see the drop reasons above.")
        return 1

    print(f"\nintent:   {dict(Counter(df['intent']))}")
    print(f"phase:    {dict(Counter(df['phase']))}")
    print(f"opp_style:{dict(Counter(df['opp_style']))}")
    n_shift = int(df["opponent_response_is_shift"].sum())
    same = int((df["opp_response"] == df["opp_formation"]).sum())
    n_changed = int((df["opp_response"] != df["opp_formation"]).sum())
    print(f"\nopponent made a tactical shift : {n_shift}/{len(df)} "
          f"= {100 * n_shift / len(df):.1f}%")
    print(f"opponent's FORMATION moved     : {n_changed}/{len(df)} "
          f"= {100 * n_changed / len(df):.1f}%   <- the training target; the "
          f"other shifts were personnel swaps inside the same shape")
    print(f"PERSISTENCE BASELINE   : {same}/{len(df)} "
          f"= {100 * same / len(df):.1f}%   "
          f"<- the number your model has to beat")
    print(f"imputed opp_ppda_pre   : {int(df['opp_ppda_pre_imputed'].sum())}")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
    except Exception as exc:                            # noqa: BLE001
        print(f"\nparquet write failed ({exc}).")
        print("The five event columns are lists, which CSV cannot round-trip,")
        print("so install a parquet engine:  pip install pyarrow")
        return 1
    print(f"\nwrote {out}")

    # Validate against the schema the model actually enforces, so a bad
    # build fails here rather than three epochs into training.
    try:
        from ramfp.data.schema import validate_triples
    except ImportError:
        print("(schema not importable; skipped validation)")
        return 0
    try:
        validate_triples(df)
    except Exception as exc:                            # noqa: BLE001
        print(f"\nSCHEMA CHECK FAILED: {exc}")
        return 1
    print("schema check passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open-data", required=True)
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--match-ppda", default=None)
    ap.add_argument("--out", default="data/processed/triples.parquet")
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--min-events", type=int, default=10,
                    help="skip a change with fewer prior events than this")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
