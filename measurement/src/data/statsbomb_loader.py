from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from socceraction.data.statsbomb import StatsBombLoader


class StatsBombDataLoader:
    """
    Loader for the local StatsBomb Open Data repository.

    Expected structure:

        sportsm5/
        ├── open-data/
        │   └── data/
        │       ├── competitions.json
        │       ├── events/
        │       ├── lineups/
        │       ├── matches/
        │       └── three-sixty/
        │
        └── src/
            └── data/
                └── statsbomb_loader.py
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
    ) -> None:

        if data_root is None:
            project_root = Path(__file__).resolve().parents[2]
            data_root = project_root / "open-data" / "data"
        else:
            data_root = Path(data_root).expanduser().resolve()

        self.data_root = Path(data_root)

        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"StatsBomb data directory does not exist: "
                f"{self.data_root}"
            )

        required_directories = (
            "events",
            "lineups",
            "matches",
        )

        for directory in required_directories:
            directory_path = self.data_root / directory

            if not directory_path.is_dir():
                raise FileNotFoundError(
                    f"Required StatsBomb directory not found: "
                    f"{directory_path}"
                )

        competitions_file = self.data_root / "competitions.json"

        if not competitions_file.is_file():
            raise FileNotFoundError(
                f"StatsBomb competitions file not found: "
                f"{competitions_file}"
            )

        self.loader = StatsBombLoader(
            getter="local",
            root=str(self.data_root),
        )

    # ------------------------------------------------------------------
    # SoccerAction API
    # ------------------------------------------------------------------

    def get_competitions(self):
        """Return every competition/season available locally."""
        return self.loader.competitions()

    def get_games(
        self,
        competition_id: int,
        season_id: int,
    ):
        """Return every game in a competition/season."""
        return self.loader.games(
            competition_id=int(competition_id),
            season_id=int(season_id),
        )

    def get_teams(self, game_id: int):
        return self.loader.teams(game_id=int(game_id))

    def get_players(self, game_id: int):
        return self.loader.players(game_id=int(game_id))

    def get_events(self, game_id: int):
        """Load all StatsBomb events for one game."""
        return self.loader.events(game_id=int(game_id))

    # ------------------------------------------------------------------
    # Raw StatsBomb lineup JSON
    # ------------------------------------------------------------------

    def get_lineups(
        self,
        game_id: int,
    ) -> list[dict[str, Any]]:
        """
        Load the raw StatsBomb lineup JSON.

        SoccerAction 1.5.3 does not expose a lineups() method,
        so this is read directly from the Open Data file.
        """

        lineup_path = (
            self.data_root
            / "lineups"
            / f"{int(game_id)}.json"
        )

        if not lineup_path.is_file():
            raise FileNotFoundError(
                f"Lineup file not found for game {game_id}: "
                f"{lineup_path}"
            )

        with lineup_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                f"Unexpected lineup format in {lineup_path}: "
                f"expected a list."
            )

        return data

    # ------------------------------------------------------------------
    # File existence helpers
    # ------------------------------------------------------------------

    def event_file_exists(self, game_id: int) -> bool:
        return (
            self.data_root
            / "events"
            / f"{int(game_id)}.json"
        ).is_file()

    def lineup_file_exists(self, game_id: int) -> bool:
        return (
            self.data_root
            / "lineups"
            / f"{int(game_id)}.json"
        ).is_file()

    # ------------------------------------------------------------------
    # Dataset iterator
    # ------------------------------------------------------------------

    def iter_games(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over every competition/season/game available locally.

        Nothing is filtered by competition or season.
        """

        competitions = self.get_competitions()

        required_columns = {
            "competition_id",
            "season_id",
        }

        missing = required_columns - set(
            competitions.columns
        )

        if missing:
            raise ValueError(
                "Competition dataframe is missing columns: "
                + ", ".join(sorted(missing))
            )

        competitions = (
            competitions[
                ["competition_id", "season_id"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        for competition in competitions.itertuples(
            index=False
        ):
            competition_id = int(
                competition.competition_id
            )

            season_id = int(
                competition.season_id
            )

            games = self.get_games(
                competition_id,
                season_id,
            )

            for game in games.to_dict(
                orient="records"
            ):
                game["competition_id"] = competition_id
                game["season_id"] = season_id

                yield game