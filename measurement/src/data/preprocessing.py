from __future__ import annotations

import json
from typing import Any

import pandas as pd


class EventPreprocessor:
    """
    Canonical preprocessing for StatsBomb event data.

    The preprocessing:
      - preserves every raw event
      - creates continuous match time
      - extracts event coordinates
      - normalizes x-direction using the match home team
      - extracts tactical formation information
      - identifies tactical shifts
      - flags halftime tactical shifts
      - creates PPDA-related event flags
    """

    REQUIRED_COLUMNS = frozenset(
        {
            "game_id",
            "event_id",
            "index",
            "period_id",
            "timestamp",
            "minute",
            "second",
            "type_name",
            "team_id",
            "team_name",
        }
    )

    # FIX: "Tackle" is not a StatsBomb event type — tackles are Duel events
    # with duel.type.name == "Tackle", handled via the is_tackle flag below.
    # "Pressure" is excluded: StatsBomb logs ~117 pressures per team per match
    # against ~34 real defensive actions, which drives PPDA to ~3.6 and makes
    # it incomparable to published PPDA (8-15). Set INCLUDE_PRESSURE = True
    # only to reproduce the draft's definition for an ablation.
    INCLUDE_PRESSURE = False

    _BASE_DEFENSIVE_EVENT_TYPES = frozenset(
        {
            "Interception",
            "Foul Committed",
            "Block",
            "Clearance",
        }
    )

    DEFENSIVE_EVENT_TYPES = (
        _BASE_DEFENSIVE_EVENT_TYPES | {"Pressure"}
        if INCLUDE_PRESSURE
        else _BASE_DEFENSIVE_EVENT_TYPES
    )

    # StatsBomb pitch dimensions.
    FIELD_LENGTH = 120.0

    # Project requirement:
    # halftime tactical shifts are retained and flagged.
    HALFTIME_SHIFT_MAX_PERIOD_SECONDS = 5.0

    def __init__(
        self,
        events: pd.DataFrame,
        home_team_id: int | None = None,
    ) -> None:

        if not isinstance(events, pd.DataFrame):
            raise TypeError(
                "events must be a pandas DataFrame"
            )

        self.events = events.copy()
        self.home_team_id = (
            int(home_team_id)
            if home_team_id is not None
            else None
        )

    def transform(self) -> pd.DataFrame:
        self._validate_columns()

        df = self.events.copy()

        df = df.sort_values(
            ["index"],
            kind="stable",
        ).reset_index(drop=True)

        # --------------------------------------------------------------
        # Basic names
        # --------------------------------------------------------------

        df["event_type"] = (
            df["type_name"]
            .astype("string")
            .str.strip()
        )

        df["team"] = (
            df["team_name"]
            .astype("string")
            .str.strip()
        )

        # --------------------------------------------------------------
        # Time
        # --------------------------------------------------------------

        time_values = self._calculate_time(df)

        df["period_seconds"] = time_values[
            "period_seconds"
        ]

        df["event_seconds"] = time_values[
            "event_seconds"
        ]

        # --------------------------------------------------------------
        # Coordinates
        # --------------------------------------------------------------

        coordinates = df["location"].apply(
            self._location_xy
            if "location" in df.columns
            else lambda _: (None, None)
        )

        df["x_raw"] = coordinates.map(
            lambda value: value[0]
        )

        df["y_raw"] = coordinates.map(
            lambda value: value[1]
        )

        # Normalize x so that x=0 is the team's own goal
        # and x=120 is the opponent's goal.
        df["x"] = df.apply(
            self._normalize_x,
            axis=1,
        )

        df["y"] = df["y_raw"]

        # --------------------------------------------------------------
        # Time buckets
        # --------------------------------------------------------------

        df["minute_bucket_5"] = (
            pd.to_numeric(
                df["event_seconds"],
                errors="coerce",
            )
            .floordiv(300)
            .astype("Int64")
        )

        df["minute_bucket_15"] = (
            pd.to_numeric(
                df["event_seconds"],
                errors="coerce",
            )
            .floordiv(900)
            .astype("Int64")
        )

        # --------------------------------------------------------------
        # Event flags
        # --------------------------------------------------------------

        df["is_pass"] = (
            df["event_type"] == "Pass"
        )

        df["is_pressure"] = (
            df["event_type"] == "Pressure"
        )

        # FIX: recover tackles from Duel. The old comparison matched zero rows
        # in all 4,712 windows (ppda_window_diagnostics.csv: tackle_events = 0).
        df["duel_type"] = df["extra"].apply(
            lambda extra: (
                (self._as_dict(extra).get("duel") or {})
                .get("type", {})
                .get("name")
            )
        )

        df["is_tackle"] = (
            df["event_type"].eq("Duel")
            & df["duel_type"].eq("Tackle")
        )

        df["is_interception"] = (
            df["event_type"] == "Interception"
        )

        df["is_block"] = (
            df["event_type"] == "Block"
        )

        df["is_defensive_action"] = (
            df["event_type"].isin(
                self.DEFENSIVE_EVENT_TYPES
            )
            | df["is_tackle"]
        )

        df["is_tactical_shift"] = (
            df["event_type"] == "Tactical Shift"
        )

        df["is_starting_xi"] = (
            df["event_type"] == "Starting XI"
        )

        # --------------------------------------------------------------
        # Goals
        # --------------------------------------------------------------

        df["is_goal"] = df.apply(
            self._is_goal,
            axis=1,
        )

        # --------------------------------------------------------------
        # Tactical information
        # --------------------------------------------------------------

        df["formation_code"] = df["extra"].apply(
            self._extract_formation
        )

        df["formation"] = df[
            "formation_code"
        ].apply(
            self.formation_label
        )

        df["lineup"] = df["extra"].apply(
            self._extract_lineup
        )

        # --------------------------------------------------------------
        # Halftime tactical shift
        # --------------------------------------------------------------

        df["is_halftime_shift"] = (
            df["is_tactical_shift"]
            & (df["period_id"] == 2)
            & (
                df["period_seconds"]
                <= self.HALFTIME_SHIFT_MAX_PERIOD_SECONDS
            )
        )

        # --------------------------------------------------------------
        # PPDA zone
        # --------------------------------------------------------------

        # x is normalized so 0 is own goal.
        # The project's operational definition is x < 40.
        # FIX: removed. The pressing zone is not a property of an event — it
        # depends on whether the event's team is pressing or being pressed, and
        # the threshold is MIRRORED between them (presser x > t, opponent
        # passes x < 120 - t). A single boolean column cannot express that, and
        # it is what put the PPDA numerator and denominator ~60 m apart.
        # analysis/ppda.py now derives the zone itself.
        df["ppda_zone"] = pd.NA

        # --------------------------------------------------------------
        # Canonical columns
        # --------------------------------------------------------------

        columns = [
            "game_id",
            "event_id",
            "index",
            "period_id",
            "timestamp",
            "period_seconds",
            "event_seconds",
            "minute",
            "second",
            "minute_bucket_5",
            "minute_bucket_15",
            "team_id",
            "team",
            "player_id",
            "player_name",
            "position_id",
            "position_name",
            "event_type",
            "x_raw",
            "y_raw",
            "x",
            "y",
            "ppda_zone",
            "possession",
            "possession_team_id",
            "possession_team_name",
            "play_pattern_id",
            "play_pattern_name",
            "duration",
            "under_pressure",
            "counterpress",
            "is_pass",
            "is_pressure",
            "duel_type",
            "is_tackle",
            "is_interception",
            "is_block",
            "is_defensive_action",
            "is_tactical_shift",
            "is_starting_xi",
            "is_halftime_shift",
            "is_goal",
            "formation_code",
            "formation",
            "lineup",
            "extra",
            "related_events",
        ]

        columns = [
            column
            for column in columns
            if column in df.columns
        ]

        return df[columns].copy()

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_columns(self) -> None:
        missing = sorted(
            self.REQUIRED_COLUMNS
            - set(self.events.columns)
        )

        if missing:
            raise ValueError(
                "Missing required event columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing
                )
            )

        # FIX: home_team_id is no longer required. Coordinates arrive
        # pre-normalised from StatsBomb (see _normalize_x), so there is no
        # home/away direction to correct. The argument is kept for API
        # compatibility and is unused for coordinates.

        if "location" not in self.events.columns:
            raise ValueError(
                "StatsBomb event data must contain "
                "the 'location' column."
            )

        if "extra" not in self.events.columns:
            raise ValueError(
                "StatsBomb event data must contain "
                "the 'extra' column."
            )

    # ==================================================================
    # Time
    # ==================================================================

    @staticmethod
    def _calculate_time(
        df: pd.DataFrame,
    ) -> dict[str, pd.Series]:

        timestamp = pd.to_timedelta(
            df["timestamp"],
            errors="coerce",
        )

        timestamp_seconds = (
            timestamp.dt.total_seconds()
        )

        minute = pd.to_numeric(
            df["minute"],
            errors="coerce",
        )

        second = pd.to_numeric(
            df["second"],
            errors="coerce",
        )

        # StatsBomb's `minute` field is CUMULATIVE across the match: a period-2
        # event at 46:30 has minute = 46, not minute = 1. So minute*60 + second
        # is already on the continuous 0..5400+ match clock, and adding the
        # period offset to it again would double-count the 2700 s halftime gap.
        #
        # The `timestamp` field is PERIOD-RELATIVE (it resets to 00:00:00.000
        # at the start of each period), so the period offset MUST be added to it.
        #
        # The previous version did `timestamp_seconds.fillna(fallback)` and then
        # added `offset` to the result, which was correct for the timestamp branch
        # and wrong by 2700 s for every fallback row in period 2 (and by 5400 s
        # / 6300 s for periods 3 and 4). Every match-time window, every formation
        # timeline, and every natural-experiment window was affected whenever an
        # event's timestamp was missing.
        fallback_event_seconds = (
            minute * 60.0
            + second
        )

        # StatsBomb period identifiers:
        #   1: first half
        #   2: second half
        #   3: first extra-time period
        #   4: second extra-time period
        #   5: penalty shootout (not continuous match-play time)
        offsets = {
            1: 0.0,
            2: 2700.0,
            3: 5400.0,
            4: 6300.0,
        }

        period_numeric = pd.to_numeric(
            df["period_id"],
            errors="coerce",
        )

        offset = period_numeric.map(
            offsets
        )

        # Timestamp branch: period-relative value + period offset = continuous clock.
        event_seconds_from_timestamp = (
            timestamp_seconds
            + offset
        )

        # Choose per row: use the timestamp when it exists, otherwise fall back
        # to the cumulative minute/second. Do NOT add the offset to the fallback.
        event_seconds = event_seconds_from_timestamp.where(
            timestamp_seconds.notna(),
            fallback_event_seconds,
        )

        # period_seconds is the period-relative clock, used downstream by the
        # halftime-shift check. Derive it from event_seconds so it is consistent
        # whether the row used the timestamp branch or the fallback branch.
        period_seconds = event_seconds - offset

        # Period 5 (penalty shootout) is not continuous match-play time.
        event_seconds = event_seconds.where(
            period_numeric != 5
        )
        period_seconds = period_seconds.where(
            period_numeric != 5
        )

        return {
            "period_seconds": period_seconds,
            "event_seconds": event_seconds,
        }

    # ==================================================================
    # Coordinates
    # ==================================================================

    @staticmethod
    def _location_xy(
        location: Any,
    ) -> tuple[float | None, float | None]:

        if not isinstance(
            location,
            (list, tuple),
        ):
            return None, None

        if len(location) < 2:
            return None, None

        try:
            return (
                float(location[0]),
                float(location[1]),
            )
        except (TypeError, ValueError):
            return None, None

    def _normalize_x(
        self,
        row: pd.Series,
    ) -> float | None:

        raw_x = row["x_raw"]

        if pd.isna(raw_x):
            return None

        team_id = row["team_id"]

        if pd.isna(team_id):
            return None

        team_id = int(team_id)

        period = int(row["period_id"])

        # Standard StatsBomb/SoccerAction convention:
        # home team attacks right in the first half.
        #
        # Teams switch ends at halftime.
        home_attacks_right = (
            period in {1, 3}
        )

        if team_id == self.home_team_id:
            attacks_right = home_attacks_right
        else:
            attacks_right = not home_attacks_right

        # FIX: StatsBomb coordinates are ALREADY normalised — every event is
        # recorded with the acting team attacking towards x = 120, in both
        # halves. Verified on 9 La Liga matches of StatsBomb open data:
        #     shot x median        105.5 (both teams)
        #     goalkeeper x median    2.5 (both teams)
        #     shot x period 1 / 2  104.8 / 103.4  (no halftime flip)
        # The previous branch mirrored the away team in periods 1 and 3 and the
        # home team in periods 2 and 4 — roughly half of all events.
        # `attacks_right` is left computed above so the reasoning stays visible
        # in the diff, but it must not drive the coordinate any more.
        _ = attacks_right
        return float(raw_x)

    # ==================================================================
    # JSON helpers
    # ==================================================================

    @staticmethod
    def _as_dict(
        value: Any,
    ) -> dict[str, Any]:

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return {}

            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}

            if isinstance(parsed, dict):
                return parsed

        return {}

    @classmethod
    def _extract_formation(
        cls,
        extra: Any,
    ) -> int | None:

        data = cls._as_dict(extra)

        tactics = data.get("tactics")

        if not isinstance(tactics, dict):
            return None

        formation = tactics.get(
            "formation"
        )

        if formation is None:
            return None

        try:
            return int(formation)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_lineup(
        cls,
        extra: Any,
    ) -> Any:

        data = cls._as_dict(extra)

        tactics = data.get("tactics")

        if not isinstance(tactics, dict):
            return None

        return tactics.get("lineup")

    # ==================================================================
    # Formation
    # ==================================================================

    @staticmethod
    def formation_label(
        formation_code: int | float | None,
    ) -> str | None:

        if formation_code is None:
            return None

        try:
            digits = str(
                int(formation_code)
            )
        except (TypeError, ValueError):
            return None

        if len(digits) < 3:
            return None

        return "-".join(digits)

    # ==================================================================
    # Goals
    # ==================================================================

    @classmethod
    def _is_goal(
        cls,
        row: pd.Series,
    ) -> bool:

        event_type = str(
            row.get("event_type", "")
        )

        if event_type in {
            "Own Goal For",
            "Own Goal Against",
        }:
            return True

        extra = cls._as_dict(
            row.get("extra")
        )

        shot = extra.get("shot")

        if not isinstance(
            shot,
            dict,
        ):
            return False

        outcome = shot.get(
            "outcome"
        )

        if not isinstance(
            outcome,
            dict,
        ):
            return False

        return (
            outcome.get("name")
            == "Goal"
        )