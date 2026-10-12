"""
Build the unsupervised pretraining corpus for the state encoder.

    python -m ramfp.data.build_sequences \
        --open-data ~/Downloads/preprocessing/open-data \
        --out data/processed/sequences.parquet

Why this exists
---------------
The opponent model has 235 positive examples.  A 1.4M-parameter transformer
cannot learn what a football sequence looks like from that, and the first
real training run showed exactly that: respond-head AUROC 0.52, i.e. chance.

Pretraining is the answer the architecture was designed around and the step
that had never been run, for the mundane reason that nothing produced the
file it reads.  Here the labels are free: the task is "predict the next
event type", every event in every match is a training signal, and 2651
matches hold about 9.4 million of them.  The encoder learns sequence
structure there, and the 235 labelled decisions are then only asked to
fine-tune a representation that already exists.

Conventions, kept identical to build_triples.py
-----------------------------------------------
Same vocabulary mappings, same period offsets, and in particular the same
meaning for event_team: 0 = the reference team, 1 = the other side.  A
window is therefore emitted TWICE, once from each team's point of view.
That doubles the corpus and, more importantly, keeps the team embedding
meaning the same thing here as it does in the triples -- if pretraining
used home/away instead, the fine-tuning step would inherit an embedding
trained on a different question.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from .build_triples import (EVENT_TYPE_TO_ID, NO_ZONE_ID,
                                UNK_EVENT_TYPE_ID, load_match, minute_bucket,
                                outcome_id, zone_id)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ramfp.data.build_triples import (  # noqa: E402
        EVENT_TYPE_TO_ID, NO_ZONE_ID, UNK_EVENT_TYPE_ID, load_match,
        minute_bucket, outcome_id, zone_id)


def encode(window, reference_team: int) -> dict:
    return {
        "event_type": [EVENT_TYPE_TO_ID.get(e["type"]["name"],
                                            UNK_EVENT_TYPE_ID) for e in window],
        "event_zone": [zone_id(*(e.get("location") or (None, None))[:2])
                       if e.get("location") else NO_ZONE_ID for e in window],
        "event_team": [0 if e["team"]["id"] == reference_team else 1
                       for e in window],
        "event_minute": [minute_bucket(e.get("minute") or 0) for e in window],
        "event_outcome": [outcome_id(e) for e in window],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open-data", required=True)
    ap.add_argument("--out", default="data/processed/sequences.parquet")
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--stride", type=int, default=50,
                    help="gap between window starts; = seq-len means no overlap")
    ap.add_argument("--max-sequences", type=int, default=200_000)
    args = ap.parse_args()

    events_dir = Path(args.open_data).expanduser() / "data" / "events"
    files = sorted(events_dir.glob("*.json"))
    # Shuffle before the cap bites.
    #
    # sorted() orders by filename, and StatsBomb match ids cluster by
    # competition and era, so stopping at --max-sequences took a contiguous
    # slice of the id space: 200k sequences came from the first 1,441 of
    # 2,651 matches. The triples the encoder is later fine-tuned on span all
    # of them, and there is no reason for the two corpora to be drawn from
    # different competitions. A fixed seed keeps the build reproducible.
    import random
    random.Random(0).shuffle(files)
    if not files:
        print(f"no event files under {events_dir}")
        return 1
    print(f"matches: {len(files)}")

    rows, n_events, skipped = [], 0, 0
    for i, fp in enumerate(files, 1):
        if len(rows) >= args.max_sequences:
            break
        events = load_match(fp)
        if not events or len(events) < args.seq_len:
            skipped += 1
            continue
        n_events += len(events)
        teams = list(dict.fromkeys(e["team"]["id"] for e in events))[:2]
        if len(teams) != 2:
            skipped += 1
            continue
        match_id = int(fp.stem)
        for start in range(0, len(events) - args.seq_len + 1, args.stride):
            window = events[start:start + args.seq_len]
            for ref in teams:
                rows.append({"match_id": match_id, **encode(window, ref)})
            if len(rows) >= args.max_sequences:
                break
        if i % 250 == 0:
            print(f"  {i}/{len(files)} matches  ->  {len(rows):,} sequences",
                  flush=True)

    df = pd.DataFrame(rows)
    print(f"\nsequences      : {len(df):,}")
    print(f"events read    : {n_events:,}")
    print(f"matches skipped: {skipped}")
    if df.empty:
        print("NOTHING BUILT")
        return 1

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
    except Exception as exc:                       # noqa: BLE001
        print(f"parquet write failed ({exc}); install pyarrow")
        return 1
    print(f"wrote {out}")

    try:
        from ramfp.data.schema import validate_sequences
    except ImportError:
        return 0
    try:
        validate_sequences(df)
    except Exception as exc:                       # noqa: BLE001
        print(f"\nSCHEMA CHECK FAILED: {exc}")
        return 1
    print("schema check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
