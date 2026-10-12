from data.statsbomb_loader import StatsBombDataLoader
import pandas as pd


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_nested_value(value, name: str):
    """Print useful information about a potentially nested value."""
    print(f"\n{name}:")
    print(f"  Python type: {type(value).__name__}")

    if isinstance(value, dict):
        print(f"  Dictionary keys: {list(value.keys())}")

        for key, item in value.items():
            print(
                f"    {key}: "
                f"type={type(item).__name__}, "
                f"value={repr(item)[:300]}"
            )

    elif isinstance(value, (list, tuple)):
        print(f"  Length: {len(value)}")
        print(f"  Value: {repr(value)[:500]}")

    else:
        print(f"  Value: {repr(value)[:500]}")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

loader = StatsBombDataLoader()

competitions = loader.get_competitions()

print_section("AVAILABLE COMPETITIONS")
print(competitions.to_string(index=False))

# For now, use the first competition/season exactly as the previous test did.
competition_id = competitions.iloc[0]["competition_id"]
season_id = competitions.iloc[0]["season_id"]

games = loader.get_games(competition_id, season_id)

print_section("SELECTED COMPETITION / SEASON")
print(f"Competition ID: {competition_id}")
print(f"Season ID:      {season_id}")

print_section("GAMES")
print(games.head(10).to_string(index=False))

game_id = games.iloc[0]["game_id"]

events = loader.get_events(game_id)

print_section("SELECTED GAME")
print(f"Game ID: {game_id}")

# ---------------------------------------------------------------------------
# 2. Basic dataframe information
# ---------------------------------------------------------------------------

print_section("DATAFRAME SHAPE")

print(f"Rows:    {events.shape[0]}")
print(f"Columns: {events.shape[1]}")

print_section("COLUMN NAMES")

for i, column in enumerate(events.columns):
    print(f"{i:2d}. {column}")

# ---------------------------------------------------------------------------
# 3. Data types
# ---------------------------------------------------------------------------

print_section("DATA TYPES")

print(events.dtypes.to_string())

# ---------------------------------------------------------------------------
# 4. Missing values
# ---------------------------------------------------------------------------

print_section("MISSING VALUES")

missing = events.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing.to_string())

# ---------------------------------------------------------------------------
# 5. Event type distribution
# ---------------------------------------------------------------------------

print_section("EVENT TYPES")

event_counts = (
    events["type_name"]
    .value_counts(dropna=False)
)

print(event_counts.to_string())

# ---------------------------------------------------------------------------
# 6. First few complete events
# ---------------------------------------------------------------------------

print_section("FIRST 10 EVENTS")

print(events.head(10).to_string())

# ---------------------------------------------------------------------------
# 7. Location examples
# ---------------------------------------------------------------------------

if "location" in events.columns:

    print_section("LOCATION EXAMPLES")

    location_rows = events[events["location"].notna()].head(10)

    for index, row in location_rows.iterrows():
        print(f"\nEvent dataframe index: {index}")
        print(f"Event type: {row.get('type_name')}")
        inspect_nested_value(row["location"], "location")

# ---------------------------------------------------------------------------
# 8. Extra-field examples
# ---------------------------------------------------------------------------

if "extra" in events.columns:

    print_section("EXTRA FIELD EXAMPLES")

    extra_rows = events[events["extra"].notna()].head(10)

    for index, row in extra_rows.iterrows():
        print(f"\nEvent dataframe index: {index}")
        print(f"Event type: {row.get('type_name')}")
        inspect_nested_value(row["extra"], "extra")

# ---------------------------------------------------------------------------
# 9. Tactical Shift events
# ---------------------------------------------------------------------------

print_section("TACTICAL SHIFT EVENTS")

if "type_name" in events.columns:

    tactical_shift = events[
        events["type_name"]
        .astype(str)
        .str.contains("Tactical Shift", case=False, na=False)
    ]

    print(f"Number of Tactical Shift events: {len(tactical_shift)}")

    if len(tactical_shift) > 0:
        print("\nColumns available for Tactical Shift events:")
        print(tactical_shift.columns.tolist())

        print("\nTactical Shift rows:")
        print(tactical_shift.to_string(index=False))

else:
    print("type_name column not found.")

# ---------------------------------------------------------------------------
# 10. Starting XI events
# ---------------------------------------------------------------------------

print_section("STARTING XI EVENTS")

if "type_name" in events.columns:

    starting_xi = events[
        events["type_name"]
        .astype(str)
        .str.contains("Starting XI", case=False, na=False)
    ]

    print(f"Number of Starting XI events: {len(starting_xi)}")

    if len(starting_xi) > 0:
        print("\nStarting XI rows:")
        print(starting_xi.to_string(index=False))

# ---------------------------------------------------------------------------
# 11. Pressure / defensive-action related events
# ---------------------------------------------------------------------------

print_section("PRESSURE / DEFENSIVE EVENTS")

defensive_types = {
    "Pressure",
    "Tackle",
    "Interception",
    "Block",
}

if "type_name" in events.columns:

    defensive_events = events[
        events["type_name"].isin(defensive_types)
    ]

    print(f"Number of defensive events: {len(defensive_events)}")

    print("\nCounts:")
    print(
        defensive_events["type_name"]
        .value_counts()
        .to_string()
    )

# ---------------------------------------------------------------------------
# 12. Pass events
# ---------------------------------------------------------------------------

print_section("PASS EVENTS")

if "type_name" in events.columns:

    passes = events[
        events["type_name"] == "Pass"
    ]

    print(f"Number of passes: {len(passes)}")

    print("\nFirst 10 passes:")
    print(passes.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# 13. Team distribution
# ---------------------------------------------------------------------------

print_section("TEAM DISTRIBUTION")

if "team_name" in events.columns:

    print(
        events["team_name"]
        .value_counts(dropna=False)
        .to_string()
    )

# ---------------------------------------------------------------------------
# 14. Period distribution
# ---------------------------------------------------------------------------

print_section("PERIOD DISTRIBUTION")

if "period_id" in events.columns:

    print(
        events["period_id"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# ---------------------------------------------------------------------------
# 15. Final summary
# ---------------------------------------------------------------------------

print_section("INSPECTION COMPLETE")

print(f"Game ID:       {game_id}")
print(f"Event count:   {len(events)}")
print(f"Column count:  {len(events.columns)}")
print(f"Event types:   {events['type_name'].nunique()}")