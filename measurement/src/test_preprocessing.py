from data.statsbomb_loader import StatsBombDataLoader
from data.preprocessing import EventPreprocessor


# ------------------------------------------------------------
# Load one game
# ------------------------------------------------------------

loader = StatsBombDataLoader()

competitions = loader.get_competitions()

if competitions.empty:
    raise RuntimeError(
        "No competitions were found in the local StatsBomb data."
    )

competition_id = competitions.iloc[0]["competition_id"]
season_id = competitions.iloc[0]["season_id"]

games = loader.get_games(
    competition_id=competition_id,
    season_id=season_id,
)

if games.empty:
    raise RuntimeError(
        "No games were found for the selected competition/season."
    )

game = games.iloc[0]

game_id = int(
    game["game_id"]
)

home_team_id = int(
    game["home_team_id"]
)

events = loader.get_events(game_id)


# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------

preprocessor = EventPreprocessor(
    events,
    home_team_id=home_team_id,
)

canonical_events = preprocessor.transform()


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PREPROCESSING RESULT")
print("=" * 80)

print(f"Game ID:       {game_id}")
print(f"Home team ID:  {home_team_id}")
print(f"Raw events:    {len(events)}")
print(f"Canonical:     {len(canonical_events)}")
print(f"Raw columns:   {len(events.columns)}")
print(f"Canonical cols:{len(canonical_events.columns)}")


# ------------------------------------------------------------
# Columns
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CANONICAL COLUMNS")
print("=" * 80)

for column in canonical_events.columns:
    print(f"  - {column}")


# ------------------------------------------------------------
# Event types
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("EVENT TYPES")
print("=" * 80)

print(
    canonical_events["event_type"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# Tactical shifts
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TACTICAL SHIFTS")
print("=" * 80)

tactical_shifts = canonical_events[
    canonical_events["is_tactical_shift"]
]

if tactical_shifts.empty:
    print("No Tactical Shift events found.")
else:
    print(
        tactical_shifts[
            [
                "event_seconds",
                "period_id",
                "team",
                "formation",
                "lineup",
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# Starting XI
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STARTING XI")
print("=" * 80)

starting_xi = canonical_events[
    canonical_events["is_starting_xi"]
]

if starting_xi.empty:
    print("No Starting XI events found.")
else:
    print(
        starting_xi[
            [
                "event_seconds",
                "period_id",
                "team",
                "formation",
                "lineup",
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# Location
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LOCATION CHECK")
print("=" * 80)

print(
    canonical_events[
        [
            "event_type",
            "x",
            "y",
            "ppda_zone",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# PPDA event flags
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PPDA EVENT FLAGS")
print("=" * 80)

print(
    canonical_events[
        [
            "event_type",
            "is_pass",
            "is_defensive_action",
            "ppda_zone",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Data integrity
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DATA INTEGRITY")
print("=" * 80)

print(
    f"Event count preserved: "
    f"{len(events) == len(canonical_events)}"
)

print(
    f"Event IDs unique: "
    f"{canonical_events['event_id'].is_unique}"
)

print(
    f"Event index unique: "
    f"{canonical_events['index'].is_unique}"
)

print(
    f"Missing event_seconds: "
    f"{canonical_events['event_seconds'].isna().sum()}"
)

print(
    f"Missing x/y: "
    f"{canonical_events[['x', 'y']].isna().all(axis=1).sum()}"
)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PREPROCESSING TEST COMPLETE")
print("=" * 80)