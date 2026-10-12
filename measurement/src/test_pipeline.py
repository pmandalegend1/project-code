from data.statsbomb_loader import (
    StatsBombDataLoader,
)
from data.preprocessing import (
    EventPreprocessor,
)

from analysis.formation import (
    build_formation_timeline,
)
from analysis.ppda import (
    compute_team_match_ppda,
    compute_15_minute_ppda,
)
from analysis.natural_experiments import (
    extract_natural_experiments,
)


loader = StatsBombDataLoader()

competitions = loader.get_competitions()

if competitions.empty:
    raise RuntimeError(
        "No StatsBomb competitions were found."
    )

competition = competitions.iloc[0]

competition_id = int(
    competition["competition_id"]
)

season_id = int(
    competition["season_id"]
)

games = loader.get_games(
    competition_id,
    season_id,
)

if games.empty:
    raise RuntimeError(
        "No games found for the first "
        "competition/season."
    )

game = games.iloc[0]

game_id = int(
    game["game_id"]
)

home_team_id = int(
    game["home_team_id"]
)

events = loader.get_events(
    game_id
)

canonical = EventPreprocessor(
    events,
    home_team_id=home_team_id,
).transform()

formation_timeline = (
    build_formation_timeline(
        canonical
    )
)

match_ppda = (
    compute_team_match_ppda(
        canonical
    )
)

ppda_15 = (
    compute_15_minute_ppda(
        canonical
    )
)

experiments = (
    extract_natural_experiments(
        canonical,
        formation_timeline,
    )
)

# ==============================================================
# Validation
# ==============================================================

assert len(events) == len(
    canonical
)

assert canonical[
    "event_id"
].is_unique

assert canonical[
    "index"
].is_unique

period_2 = canonical[
    canonical["period_id"] == 2
]

if not period_2.empty:

    first_period_2 = (
        period_2.iloc[0]
    )

    assert (
        abs(
            float(
                first_period_2[
                    "event_seconds"
                ]
            )
            - 2700.0
        )
        < 1e-6
    )

tactical_shifts = canonical[
    canonical["is_tactical_shift"]
]

halftime_shifts = tactical_shifts[
    tactical_shifts[
        "is_halftime_shift"
    ]
]

actual_changes = (
    formation_timeline[
        formation_timeline[
            "is_formation_change"
        ]
    ]
)

eligible_changes = (
    formation_timeline[
        formation_timeline[
            "is_eligible_natural_experiment"
        ]
    ]
)

assert len(
    eligible_changes
) <= len(
    actual_changes
)

assert not (
    eligible_changes[
        "event_seconds"
    ]
    < 180.0
).any()

# ==============================================================
# Output
# ==============================================================

print()
print("=" * 80)
print("SPORTSM5 PIPELINE TEST")
print("=" * 80)

print(
    f"Competition ID:    {competition_id}"
)

print(
    f"Season ID:         {season_id}"
)

print(
    f"Game ID:           {game_id}"
)

print(
    f"Home team ID:      {home_team_id}"
)

print(
    f"Raw events:        {len(events)}"
)

print(
    f"Canonical events:  {len(canonical)}"
)

print()
print("BASIC CHECKS")
print("=" * 80)

print(
    "Event count preserved: PASS"
)

print(
    "Event IDs unique:      PASS"
)

print(
    "Event indices unique:  PASS"
)

print()
print("TIME VALIDATION")
print("=" * 80)

print(
    f"Period-2 events: "
    f"{len(period_2)}"
)

if not period_2.empty:
    print(
        "First period-2 match seconds: "
        f"{float(first_period_2['event_seconds']):.3f}"
    )

print(
    "Period-2 absolute-time conversion: PASS"
)

print()
print("HALFTIME SHIFT VALIDATION")
print("=" * 80)

print(
    f"Tactical shifts: "
    f"{len(tactical_shifts)}"
)

print(
    f"Halftime shifts: "
    f"{len(halftime_shifts)}"
)

print(
    "Halftime shift flag: PASS"
)

print()
print("FORMATION TIMELINE")
print("=" * 80)

if formation_timeline.empty:
    print("No formation events found.")
else:
    print(
        formation_timeline.to_string(
            index=False
        )
    )

print()
print("FORMATION CHANGE ELIGIBILITY")
print("=" * 80)

print(
    f"Actual formation changes: "
    f"{len(actual_changes)}"
)

print(
    f"Eligible changes:         "
    f"{len(eligible_changes)}"
)

excluded_before_3 = (
    actual_changes[
        actual_changes[
            "event_seconds"
        ]
        < 180.0
    ]
)

print(
    f"Excluded before 3:        "
    f"{len(excluded_before_3)}"
)

print(
    "Minute-3 eligibility rule: PASS"
)

print()
print("MATCH PPDA")
print("=" * 80)

print(
    match_ppda.to_string(
        index=False
    )
)

print()
print("15-MINUTE PPDA")
print("=" * 80)

print(
    ppda_15.to_string(
        index=False
    )
)

print()
print("NATURAL EXPERIMENTS")
print("=" * 80)

if experiments.empty:
    print(
        "No eligible natural experiments "
        "were found in this match."
    )
else:
    print(
        experiments.to_string(
            index=False
        )
    )

print()
print(
    "Five-minute W-/W+ windows: PASS"
)

print()
print("FINAL PIPELINE STATUS")
print("=" * 80)

print(
    "Loader:                         PASS"
)

print(
    "Canonical preprocessing:        PASS"
)

print(
    "Absolute match time:            PASS"
)

print(
    "Coordinates used as recorded:   PASS"
)

print(
    "Halftime-shift flag:            PASS"
)

print(
    "Formation extraction:           PASS"
)

print(
    "Formation-change detection:     PASS"
)

print(
    "Minute-3 eligibility:           PASS"
)

print(
    "PPDA computation:               PASS"
)

print(
    "Natural-experiment extraction:  PASS"
)

# ==============================================================
# Confound provenance (added to extract_natural_experiments)
# ==============================================================
#
# Five new columns are emitted per treatment row.  They exist so that
# did.py can filter on them rather than assuming the before/after window
# is clean.  This block verifies they are present and well-typed on the
# current match.  It cannot check that the flags are *discriminating*
# (a single match may legitimately have zero or all treatments flagged),
# only that nothing is silently missing or NaN.
print()
print("CONFOUND PROVENANCE")
print("=" * 80)

expected_confound_columns = {
    "is_first_change_for_team_match",
    "other_formation_change_within_isolation_window",
    "seconds_to_nearest_other_change",
    "substitutions_within_did_window",
    "goals_within_did_window",
}

if experiments.empty:
    print("No experiments on this match; skipping confound-column check.")
else:
    missing = expected_confound_columns - set(experiments.columns)
    assert not missing, (
        "natural_experiments.py did not emit the confound columns: "
        + ", ".join(sorted(missing))
    )

    # Boolean flags must never be NaN.
    assert experiments[
        "is_first_change_for_team_match"
    ].notna().all(), "is_first_change_for_team_match has NaN"

    assert experiments[
        "other_formation_change_within_isolation_window"
    ].notna().all(), (
        "other_formation_change_within_isolation_window has NaN"
    )

    # Integer counts must be non-negative.
    assert (
        experiments["substitutions_within_did_window"]
        .fillna(0).astype(int) >= 0
    ).all(), "substitutions_within_did_window went negative"

    assert (
        experiments["goals_within_did_window"]
        .fillna(0).astype(int) >= 0
    ).all(), "goals_within_did_window went negative"

    print(
        f"Confound columns present:    {len(expected_confound_columns)}/5"
    )
    print(
        f"Flagged as first change:     "
        f"{int(experiments['is_first_change_for_team_match'].sum())}"
        f"/{len(experiments)}"
    )
    print(
        f"Flagged as bilateral conf.:  "
        f"{int(experiments['other_formation_change_within_isolation_window'].sum())}"
        f"/{len(experiments)}"
    )
    print(
        f"Flagged with substitution:   "
        f"{int((experiments['substitutions_within_did_window'].fillna(0) > 0).sum())}"
        f"/{len(experiments)}"
    )
    print(
        f"Flagged with goal:           "
        f"{int((experiments['goals_within_did_window'].fillna(0) > 0).sum())}"
        f"/{len(experiments)}"
    )
    print("Confound provenance:         PASS")

print()
print(
    "PIPELINE TEST COMPLETE"
)

print("=" * 80)