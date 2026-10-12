"""
Per-team behavioural features, for a style variable that is not just PPDA.

    python src/run_behaviour_features.py --open-data open-data \
        --out processed-data/team_behaviour_features.csv

Why
---
The style clusters were built from ppda_mean, ppda_std and ppda_slope --
three summaries of one behaviour, with mean and std correlating at r=+0.69.
That supports a binary pressing regime and nothing more, and the opponent
model's own ablation found no benefit from conditioning on it. That
ablation tested a single bit of pressing intensity, not playing style.

These five features measure things PPDA does not:

    pass_length_mean       directness  -- long or short build-up
    possession_share       how much of the ball the team takes
    width                  lateral spread of their own on-ball actions
    passes_per_possession  tempo -- patient circulation or quick release
    def_line_height        where they defend, as opposed to how hard

Height and intensity are deliberately separate: a team can press hard in a
low block, or hold a high line without pressing. PPDA cannot tell those
apart; def_line_height with ppda_mean can.

Read straight from the raw StatsBomb JSON and importing nothing from the
rest of src/, so this cannot inherit a bug from the pipeline it feeds.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

PITCH_X, PITCH_Y = 120.0, 80.0
ON_BALL = {"Pass", "Carry", "Dribble"}
DEF_TYPES = {"Interception", "Block", "Clearance", "Foul Committed"}


def is_tackle(e: dict) -> bool:
    return (e["type"]["name"] == "Duel"
            and (e.get("duel") or {}).get("type", {}).get("name") == "Tackle")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-data", default="open-data")
    ap.add_argument("--out", default="processed-data/team_behaviour_features.csv")
    ap.add_argument("--min-matches", type=int, default=1)
    args = ap.parse_args()

    files = sorted((Path(args.open_data) / "data" / "events").glob("*.json"))
    if not files:
        print(f"no event files under {args.open_data}/data/events")
        return 1
    print(f"matches: {len(files)}")

    pass_len = defaultdict(list)
    ys = defaultdict(list)
    def_x = defaultdict(list)
    n_passes = defaultdict(int)
    n_poss = defaultdict(int)
    poss_total = defaultdict(int)          # possessions in matches the team played
    matches = defaultdict(set)

    for i, fp in enumerate(files, 1):
        try:
            ev = json.load(open(fp))
        except Exception:                                   # noqa: BLE001
            continue
        mid = fp.stem
        teams = list(dict.fromkeys(e["team"]["id"] for e in ev if "team" in e))[:2]
        if len(teams) != 2:
            continue
        for t in teams:
            matches[t].add(mid)

        seen_poss = {}
        for e in ev:
            tname = e["type"]["name"]
            tm = e.get("team", {}).get("id")
            if tm is None:
                continue

            pid = e.get("possession")
            pteam = (e.get("possession_team") or {}).get("id")
            if pid is not None and pid not in seen_poss:
                seen_poss[pid] = pteam

            loc = e.get("location")
            if tname == "Pass":
                n_passes[tm] += 1
                length = (e.get("pass") or {}).get("length")
                if length is not None:
                    pass_len[tm].append(float(length))
            if tname in ON_BALL and loc:
                ys[tm].append(float(loc[1]))
            if (tname in DEF_TYPES or is_tackle(e)) and loc:
                def_x[tm].append(float(loc[0]))

        for pid, pteam in seen_poss.items():
            if pteam is not None:
                n_poss[pteam] += 1
        for t in teams:
            poss_total[t] += len(seen_poss)

        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    rows = []
    for t in sorted(matches):
        nm = len(matches[t])
        if nm < args.min_matches:
            continue
        if not pass_len[t] or not ys[t] or not def_x[t] or not poss_total[t]:
            continue
        rows.append({
            "team_id": int(t),
            "behaviour_match_count": nm,
            "pass_length_mean": round(st.mean(pass_len[t]), 4),
            "possession_share": round(n_poss[t] / poss_total[t], 4),
            "width": round(st.pstdev(ys[t]), 4),
            "passes_per_possession": round(n_passes[t] / max(n_poss[t], 1), 4),
            "def_line_height": round(st.median(def_x[t]), 4),
        })

    if not rows:
        print("no teams produced features")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nteams: {len(rows)}")
    for c in ("pass_length_mean", "possession_share", "width",
              "passes_per_possession", "def_line_height"):
        v = [r[c] for r in rows]
        print(f"  {c:<24} median {st.median(v):7.3f}   "
              f"min {min(v):7.3f}   max {max(v):7.3f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
