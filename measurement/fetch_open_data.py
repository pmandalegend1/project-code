#!/usr/bin/env python3
"""
fetch_open_data.py  --  download ONLY the StatsBomb matches this project needs.

Why this exists
---------------
`git clone` of statsbomb/open-data needs about 7 GB of free disk (1.6 GB of
git objects plus a ~5 GB checkout) and pulls in three-sixty tracking data we
never touch.  This script skips git entirely and fetches, straight over HTTPS,
only:

    data/competitions.json
    data/matches/<competition_id>/<season_id>.json
    data/events/<match_id>.json
    data/lineups/<match_id>.json

for a bounded number of matches.  400 matches is roughly 700 MB.

Usage
-----
    python fetch_open_data.py                 # 400 matches, default folder
    python fetch_open_data.py 150             # fewer matches
    python fetch_open_data.py 400 --dir /path/to/open-data

Re-running is safe: any file already on disk is skipped, so an interrupted
run just resumes.

After it finishes, each matches/<cid>/<sid>.json is rewritten to list ONLY
the matches whose events were actually downloaded.  That keeps the local
open-data folder internally consistent, so socceraction's StatsBombLoader
never asks for an events file that is not there.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Competitions we want first, best-for-this-study first.  Matching is a
# case-insensitive substring test on StatsBomb's competition_name.
#
# Ordering rationale: full domestic league seasons give the most matches per
# competition and the widest spread of teams, which is what the DiD design
# needs.  Tournaments come next; they have fewer matches but plenty of
# in-game tactical shifts.
PREFERRED = [
    "premier league",
    "la liga",
    "serie a",
    "bundesliga",
    "ligue 1",
    "indian super league",
    "uefa euro",
    "fifa world cup",
    "copa america",
    "champions league",
    "africa cup of nations",
]

BYTES_PER_MATCH = 2_000_000   # generous estimate for the free-space check
HEADROOM = 1_000_000_000      # leave 1 GB free on the volume


# ---------------------------------------------------------------- http
def fetch(url: str, tries: int = 4) -> bytes:
    """GET with exponential backoff.  Raises on final failure."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "sportsm5-fetch/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:          # genuinely absent, do not retry
                raise
            last = e
        except Exception as e:         # noqa: BLE001 - network is messy
            last = e
        time.sleep((2 ** attempt) + random.random())
    raise RuntimeError(f"failed after {tries} tries: {url} ({last})")


def fetch_json(url: str):
    return json.loads(fetch(url).decode("utf-8"))


def save(path: Path, data: bytes) -> None:
    """Write atomically, so an interrupted run never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


# ---------------------------------------------------------------- selection
def priority(name: str) -> int:
    low = (name or "").lower()
    for i, key in enumerate(PREFERRED):
        if key in low:
            return i
    return len(PREFERRED)


def pick_competitions(comps):
    """Men's competitions first, by the preference order above."""
    men = [c for c in comps if c.get("competition_gender") != "female"]
    pool = men or comps
    return sorted(
        pool,
        key=lambda c: (
            priority(c.get("competition_name", "")),
            -int(c.get("season_id", 0)),      # newest season first
        ),
    )


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=400,
                    help="number of matches to download (default 400)")
    ap.add_argument("--dir", default=None,
                    help="open-data folder (default: ./open-data next to this script)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.dir).expanduser() if args.dir \
        else Path(__file__).resolve().parent / "open-data"
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    # ---- free space guard -------------------------------------------------
    # Count only what is still missing.  This script is meant to be re-run to
    # extend an existing corpus, and charging the caller for matches already
    # on disk made a resumed run refuse to start on a nearly-full volume.
    already = len(list((data / "events").glob("*.json"))) if (data / "events").exists() else 0
    remaining = max(0, args.n - already)
    need = remaining * BYTES_PER_MATCH + HEADROOM
    free = shutil.disk_usage(data).free
    print(f"already on disk: {already} matches; fetching about {remaining} more")
    print(f"free disk    : {free/1e9:6.2f} GB")
    print(f"estimated use: {remaining * BYTES_PER_MATCH/1e9:6.2f} GB "
          f"for {remaining} more matches\n")
    if free < need:
        can = already + max(0, int((free - HEADROOM) // BYTES_PER_MATCH))
        print(f"NOT ENOUGH SPACE. With {free/1e9:.2f} GB free you can fetch "
              f"about {can} matches.")
        print(f"Either free some disk, or re-run:  python {Path(__file__).name} {can}")
        return 1

    # ---- competitions -----------------------------------------------------
    print("1/4  competitions.json")
    comps = fetch_json(f"{RAW}/competitions.json")
    save(data / "competitions.json", json.dumps(comps).encode("utf-8"))
    ordered = pick_competitions(comps)
    print(f"     {len(comps)} competition-seasons available\n")

    # ---- match lists, until we have enough --------------------------------
    print("2/4  match lists")
    wanted = []           # (cid, sid, match_id)
    used_seasons = []     # (cid, sid, full match list)
    # Fetched in parallel and only then walked in preference order.  Done
    # one at a time this took longer than the whole event download for a
    # 2700-match target, because it is ~60 sequential round trips before a
    # single event file is touched.
    def season_list(c):
        cid, sid = c["competition_id"], c["season_id"]
        try:
            return c, fetch_json(f"{RAW}/matches/{cid}/{sid}.json"), None
        except Exception as e:                   # noqa: BLE001
            return c, None, e

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        listed = list(pool.map(season_list, ordered))

    for c, matches, err in listed:               # preference order preserved
        if len(wanted) >= args.n:
            break
        cid, sid = c["competition_id"], c["season_id"]
        if err is not None:
            print(f"     skip {c.get('competition_name')} {c.get('season_name')}: {err}")
            continue
        if not matches:
            continue
        used_seasons.append((cid, sid, matches))
        for m in matches:
            wanted.append((cid, sid, m["match_id"]))
        print(f"     {str(c.get('competition_name','?')):<28} "
              f"{str(c.get('season_name','?')):<10} "
              f"+{len(matches):>4}  (total {len(wanted)})")

    wanted = wanted[:args.n]
    keep_ids = {mid for _, _, mid in wanted}
    print(f"\n     downloading {len(wanted)} matches\n")

    # ---- events + lineups -------------------------------------------------
    print("3/4  events + lineups")
    done = 0
    failed = []
    total_bytes = 0

    def one(mid):
        got = 0
        for kind in ("events", "lineups"):
            dest = data / kind / f"{mid}.json"
            if dest.exists() and dest.stat().st_size > 0:
                got += dest.stat().st_size
                continue
            try:
                blob = fetch(f"{RAW}/{kind}/{mid}.json")
            except urllib.error.HTTPError as e:
                if e.code == 404 and kind == "lineups":
                    continue                     # lineups are optional
                return mid, got, f"{kind}: {e}"
            except Exception as e:               # noqa: BLE001
                return mid, got, f"{kind}: {e}"
            save(dest, blob)
            got += len(blob)
        return mid, got, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, mid): mid for _, _, mid in wanted}
        for fut in as_completed(futures):
            mid, nbytes, err = fut.result()
            done += 1
            total_bytes += nbytes
            if err:
                failed.append(mid)
                keep_ids.discard(mid)
                print(f"\n     FAIL {mid}  {err}")
            if done % 25 == 0 or done == len(wanted):
                print(f"     {done:>4}/{len(wanted)}   "
                      f"{total_bytes/1e6:7.1f} MB", flush=True)

    # ---- trim the match lists to what we actually have ---------------------
    print("\n4/4  trimming match lists to downloaded matches")
    kept_total = 0
    kept_pairs = set()
    for cid, sid, matches in used_seasons:
        kept = [m for m in matches if m["match_id"] in keep_ids]
        if not kept:
            continue
        kept_total += len(kept)
        kept_pairs.add((cid, sid))
        save(data / "matches" / str(cid) / f"{sid}.json",
             json.dumps(kept).encode("utf-8"))
    print(f"     {kept_total} matches listed across {len(kept_pairs)} seasons")

    # competitions.json must list ONLY the seasons whose matches file exists,
    # or socceraction's loader walks every pair in it and dies on the first
    # one we never fetched.
    trimmed = [c for c in comps
               if (c["competition_id"], c["season_id"]) in kept_pairs]
    save(data / "competitions.json", json.dumps(trimmed).encode("utf-8"))
    print(f"     competitions.json: {len(comps)} -> {len(trimmed)} pairs")

    # ---- verdict ----------------------------------------------------------
    n_events = len(list((data / "events").glob("*.json")))
    print("\n" + "=" * 60)
    print(f"  events files on disk : {n_events}")
    print(f"  downloaded this run  : {total_bytes/1e6:.1f} MB")
    print(f"  failed matches       : {len(failed)}")
    print(f"  folder               : {root}")
    print("=" * 60)
    if failed:
        print("\nSome matches failed. Just re-run this script - it resumes.")
        return 1
    print("\nDone. Next:")
    print("  python src/process_dataset.py --limit 20 --output-dir processed-data-smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
