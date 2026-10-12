#!/usr/bin/env python3
"""
Acceptance gates for the PPDA / coordinate / style-clustering fixes.

Run this BEFORE re-running the DiD. If any gate fails, the DiD result is
not interpretable and must not go in the paper.

    python verify_fixes.py \
        --open-data   path/to/open-data \
        --ppda-15min  processed-data/ppda_15min.csv \
        --clusters    processed-data/team_style_clusters.csv \
        --windows     processed-data/ppda-diagnostics/ppda_window_diagnostics.csv

Every argument is optional; each one enables the gates it can check.
Exit code 0 = all runnable gates passed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"
results: list[tuple[str, str, str]] = []


def gate(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ",
            SKIP: " skip ", INFO: " info "}[status]
    print(f"[{mark}] {name}\n         {detail}")


# ----------------------------------------------------------------- raw JSON
def check_raw(open_data: str, n_matches: int = 6) -> None:
    files = sorted(glob.glob(os.path.join(open_data, "data", "events", "*.json")))
    if not files:
        gate("G1 tackles are counted", SKIP, "no --open-data events found")
        gate("G2 coordinates already normalised", SKIP, "no --open-data events found")
        gate("G3 PPDA lands in the published range", SKIP, "no --open-data events found")
        return
    files = files[:n_matches]

    def is_tackle(e):
        return (e["type"]["name"] == "Duel"
                and (e.get("duel") or {}).get("type", {}).get("name") == "Tackle")

    # ---- G1: tackles exist and are only reachable via Duel ----
    tackles = types = 0
    for f in files:
        ev = json.load(open(f, encoding='utf-8'))
        tackles += sum(1 for e in ev if is_tackle(e))
        types += sum(1 for e in ev if e["type"]["name"] == "Tackle")
    if tackles > 0 and types == 0:
        gate("G1 tackles are counted", PASS,
             f"{tackles} tackles found inside Duel across {len(files)} matches; "
             f'literal "Tackle" event type appears {types} times (as expected)')
    else:
        gate("G1 tackles are counted", FAIL,
             f"tackles-in-Duel={tackles}, literal-Tackle-type={types}")

    # ---- G2: coordinates are pre-normalised (no flip needed) ----
    shot_x, gk_x = [], []
    for f in files:
        ev = json.load(open(f, encoding='utf-8'))
        shot_x += [e["location"][0] for e in ev
                   if e["type"]["name"] == "Shot" and e.get("location")]
        gk_x += [e["location"][0] for e in ev
                 if e["type"]["name"] == "Goal Keeper" and e.get("location")]
    if shot_x and gk_x:
        ms, mg = st.median(shot_x), st.median(gk_x)
        if ms > 90 and mg < 25:
            gate("G2 coordinates already normalised", PASS,
                 f"shot x median {ms:.1f} (>90), goalkeeper x median {mg:.1f} (<25) "
                 "-> every team attacks towards 120; do NOT mirror by home/away")
        else:
            gate("G2 coordinates already normalised", FAIL,
                 f"shot x median {ms:.1f}, goalkeeper x median {mg:.1f} — "
                 "unexpected; inspect before trusting the zone filter")
    else:
        gate("G2 coordinates already normalised", SKIP, "no shot/keeper events")

    # ---- G3: PPDA magnitude ----
    def ppda(ev, presser, opp, include_pressure, t=40.0):
        DEF = {"Interception", "Foul Committed", "Block", "Clearance"}
        if include_pressure:
            DEF = DEF | {"Pressure"}
        p = sum(1 for e in ev if e["team"]["id"] == opp and e["type"]["name"] == "Pass"
                and e.get("location") and e["location"][0] < 120 - t)
        a = sum(1 for e in ev if e["team"]["id"] == presser
                and (e["type"]["name"] in DEF or is_tackle(e))
                and e.get("location") and e["location"][0] > t)
        return p / a if a else None

    for label, incl in (("without Pressure (default)", False),
                        ("with Pressure (draft spec)", True)):
        vals = []
        for f in files:
            ev = json.load(open(f, encoding='utf-8'))
            tm = list({e["team"]["id"] for e in ev if "team" in e})
            if len(tm) != 2:
                continue
            for a, b in ((tm[0], tm[1]), (tm[1], tm[0])):
                v = ppda(ev, a, b, incl)
                if v is not None:
                    vals.append(v)
        if not vals:
            continue
        med = st.median(vals)
        in_range = 8.0 <= med <= 15.0
        if incl:
            # The draft spec counts Pressure as a defensive action, which
            # inflates the denominator and drives PPDA far below the
            # published 8-15 band.  Demonstrating that is the POINT of this
            # variant, so being out of range is the expected result and the
            # evidence for excluding Pressure -- not a failure of the
            # pipeline.  (This gate previously asserted 8-15 for both
            # variants, which made the fix look broken every run.)
            gate(f"G3b PPDA with Pressure is out of range, as expected",
                 PASS if not in_range else FAIL,
                 f"median {med:.2f}, mean {st.mean(vals):.2f} — the draft "
                 f"spec is not comparable to published PPDA, which is why "
                 f"INCLUDE_PRESSURE is False"
                 + ("" if not in_range else
                    "  <-- unexpectedly inside 8-15; re-check the definition"))
        else:
            gate(f"G3 PPDA in published range — {label}",
                 PASS if in_range else FAIL,
                 f"median {med:.2f}, mean {st.mean(vals):.2f} (target 8-15)"
                 + ("" if in_range else
                    "  <-- not comparable to published PPDA"))


# ------------------------------------------------------------ processed CSV
def check_processed(ppda_csv, clusters_csv, windows_csv, experiments_csv) -> None:
    try:
        import pandas as pd
    except ImportError:
        gate("G4-G6 processed-data gates", SKIP, "pandas not available")
        return

    # ---- G4: undefined-PPDA rate ----
    # G4 was originally "undefined-PPDA rate below 5%".  That threshold
    # encoded an assumption the data falsified: that one definition could
    # give BOTH a PPDA in the published band AND few empty windows.  It
    # cannot.  On 1200 matches the Opta-standard definition (no Pressure)
    # yields PPDA 11.28 with 14.3% zero-action 5-minute windows, while the
    # draft spec (with Pressure) yields 1.4% empty windows and a PPDA of
    # 2.66 that no published work can be compared against.  The definition
    # was chosen for comparability, deliberately and on the record.
    #
    # So the old gate no longer asks "is the pipeline sound?" -- it asks
    # "did we pick the low-missingness definition?", which we did not.
    # The property that actually protects the analysis is that empty
    # windows are MODELLED rather than dropped: a zero-action window is an
    # observation (y = 0), not a missing value.  G4a reports the rate,
    # which the paper must state; G4b is the gate that can fail.
    if windows_csv and os.path.exists(windows_csv):
        d = pd.read_csv(windows_csv)
        if "ppda_missing" in d:
            rate = float(d["ppda_missing"].mean())
            tk = float(d["tackle_events"].mean()) if "tackle_events" in d else float("nan")
            gate("G4a undefined-PPDA rate (reported, not a pass/fail)", INFO,
                 f"{rate:.2%} of {len(d)} windows have zero defensive "
                 f"actions, so the PPDA ratio is undefined there; mean "
                 f"tackle_events = {tk:.2f}. Expected to be well above zero "
                 f"under the Opta definition -- this is exactly why the DiD "
                 f"uses the count model, not the ratio.")
    else:
        gate("G4a undefined-PPDA rate (reported, not a pass/fail)", SKIP,
             "no --windows file")

    # ---- G4b: no experiment is lost to an undefined ratio ----
    if experiments_csv and os.path.exists(experiments_csv):
        e = pd.read_csv(experiments_csv)
        n = len(e)
        count_cols = ["before_defensive_actions", "after_defensive_actions",
                      "before_passes", "after_passes"]
        missing_cols = [c for c in count_cols if c not in e.columns]
        if missing_cols:
            gate("G4b count model can use every experiment", FAIL,
                 f"natural_experiments.csv is missing {missing_cols}; without "
                 f"the raw counts the DiD can only use the ratio, and every "
                 f"zero-action window is dropped")
        else:
            usable = int(e[count_cols].notna().all(axis=1).sum())
            ratio_usable = int(
                e[["before_ppda", "after_ppda"]].notna().all(axis=1).sum()
            )
            ok = usable == n
            gate("G4b count model can use every experiment",
                 PASS if ok else FAIL,
                 f"counts present for {usable}/{n} experiments "
                 f"({100 * usable / n:.1f}%); the ratio outcome would keep "
                 f"only {ratio_usable}/{n} ({100 * ratio_usable / n:.1f}%), "
                 f"losing {n - ratio_usable} rows whose windows had zero "
                 f"defensive actions"
                 + ("" if ok else "  <-- some counts are missing entirely"))
    else:
        gate("G4b count model can use every experiment", SKIP,
             "no --experiments file")

    # ---- G5 / G6: style clustering ----
    if clusters_csv and os.path.exists(clusters_csv):
        c = pd.read_csv(clusters_csv)
        ct = pd.crosstab(c["style_cluster"], c["modal_formation_code"])
        shared = int((ct.gt(0).sum(axis=0) > 1).sum())
        ok5 = shared > 0
        gate("G5 clusters are not pure formation labels", PASS if ok5 else FAIL,
             f"{shared} formation(s) appear in more than one cluster "
             f"(was 0 — a perfect partition)")

        g = c.groupby("style_cluster")["ppda_mean"]
        between = float(g.mean().max() - g.mean().min())
        within = float(g.std().median())
        ok6 = between > within
        gate("G6 clusters separate on PPDA", PASS if ok6 else FAIL,
             f"between-cluster spread {between:.3f} vs median within-cluster "
             f"std {within:.3f} (was 0.241 vs 0.335 — i.e. failing)")
        print("\n  cluster x formation crosstab:")
        print("  " + ct.to_string().replace("\n", "\n  ") + "\n")
    else:
        gate("G5 clusters are not pure formation labels", SKIP, "no --clusters file")
        gate("G6 clusters separate on PPDA", SKIP, "no --clusters file")

    # ---- G7: match-level PPDA distribution ----
    if ppda_csv and os.path.exists(ppda_csv):
        d = pd.read_csv(ppda_csv)
        v = d["ppda"].dropna()
        if len(v):
            med = float(v.median())
            ok = 6.0 <= med <= 20.0
            gate("G7 windowed PPDA magnitude sane", PASS if ok else FAIL,
                 f"median {med:.2f} over {len(v)} windows (was ~2.4)")
    else:
        gate("G7 windowed PPDA magnitude sane", SKIP, "no --ppda-15min file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-data")
    ap.add_argument("--ppda-15min")
    ap.add_argument("--clusters")
    ap.add_argument("--windows")
    ap.add_argument("--experiments")
    a = ap.parse_args()

    print("=" * 72)
    print("RAMFP data-pipeline acceptance gates")
    print("=" * 72 + "\n")

    if a.open_data:
        check_raw(a.open_data)
    else:
        gate("G1-G3 raw-data gates", SKIP, "pass --open-data to enable")
    print()
    check_processed(a.ppda_15min, a.clusters, a.windows, a.experiments)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print("\n" + "=" * 72)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nDO NOT re-run the DiD until every gate above passes.")
        print("A null result from a broken measurement is not a finding.")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
