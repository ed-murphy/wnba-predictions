"""
src/data/odds_client.py

Fetches historical and upcoming WNBA betting spreads from The Odds API.
Requires an API key set as ODDS_API_KEY in your .env file.
Free tier: 500 requests/month. Historical odds require a paid plan.

Docs: https://the-odds-api.com/liveapi/guides/v4/
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.the-odds-api.com/v4"
_SPORT = "basketball_wnba"


class OddsAPIClient:
    """Thin wrapper around The Odds API for WNBA spread data."""

    def __init__(self, api_key: str | None = None, raw_dir: str | Path = "data/raw") -> None:
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key or self.api_key == "your_odds_api_key_here":
            logger.debug(
                "No ODDS_API_KEY found — Action Network will be used as the odds source."
            )
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_upcoming_spreads(self) -> pd.DataFrame:
        """Return spread lines for all upcoming WNBA games."""
        url = f"{_BASE_URL}/sports/{_SPORT}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "spreads",
            "oddsFormat": "american",
        }
        data = self._get(url, params)
        rows = self._parse_odds(data)
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(self.raw_dir / "upcoming_spreads.parquet", index=False)
        if df.empty:
            logger.debug("Odds API returned 0 upcoming spread lines")
        else:
            logger.info("Fetched %d upcoming spread lines", len(df))
        return df

    def fetch_historical_spreads(self, date: str) -> pd.DataFrame:
        """
        Return spreads for games on *date* (ISO-8601: YYYY-MM-DDTHH:MM:SSZ).
        NOTE: Historical endpoint requires a paid Odds API plan.
        """
        url = f"{_BASE_URL}/sports/{_SPORT}/odds-history"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "spreads",
            "oddsFormat": "american",
            "date": date,
        }
        data = self._get(url, params)
        rows = self._parse_odds(data.get("data", data) if isinstance(data, dict) else data)
        return pd.DataFrame(rows)

    def fetch_odds_snapshot(self, timestamp: str) -> pd.DataFrame:
        """
        Return spreads as they were at a specific *timestamp*
        (ISO-8601: YYYY-MM-DDTHH:MM:SSZ).

        This is a thin wrapper around fetch_historical_spreads that also tags
        each row with the snapshot timestamp so callers can distinguish opening
        from closing snapshots.

        NOTE: Requires a paid Odds API plan.
        """
        df = self.fetch_historical_spreads(timestamp)
        if not df.empty:
            df["snapshot_time"] = pd.Timestamp(timestamp, tz="UTC")
        return df

    def fetch_line_snapshots(
        self,
        game_dates: list[str],
        hours_before_open: float = 72.0,
        hours_before_close: float = 1.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch opening and closing line snapshots for a list of game dates.

        For each date in *game_dates* (YYYY-MM-DD), two snapshots are fetched:
        - Opening : ``hours_before_open`` hours before midnight on that date
        - Closing : ``hours_before_close`` hours before midnight on that date

        The two DataFrames returned are suitable for passing directly to
        ``LineMovementProcessor.compute(opening, closing)``.

        Parameters
        ----------
        game_dates : list of str
            Game dates in YYYY-MM-DD format.
        hours_before_open : float
            How many hours before tipoff to treat as the opening line (default 72).
        hours_before_close : float
            How many hours before tipoff to treat as the closing line (default 1).

        Returns
        -------
        opening_df, closing_df : tuple of DataFrames
        """
        import datetime

        open_frames: list[pd.DataFrame] = []
        close_frames: list[pd.DataFrame] = []

        for date_str in game_dates:
            game_day = datetime.date.fromisoformat(date_str)
            # Assume tipoff around 20:00 ET = midnight UTC (rough approximation)
            tipoff_utc = datetime.datetime(
                game_day.year, game_day.month, game_day.day, 0, 0, 0,
                tzinfo=datetime.timezone.utc,
            )

            open_ts = tipoff_utc - datetime.timedelta(hours=hours_before_open)
            close_ts = tipoff_utc - datetime.timedelta(hours=hours_before_close)

            open_snap = self.fetch_odds_snapshot(open_ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
            close_snap = self.fetch_odds_snapshot(close_ts.strftime("%Y-%m-%dT%H:%M:%SZ"))

            if not open_snap.empty:
                open_frames.append(open_snap)
            if not close_snap.empty:
                close_frames.append(close_snap)

        opening_df = pd.concat(open_frames, ignore_index=True) if open_frames else pd.DataFrame()
        closing_df = pd.concat(close_frames, ignore_index=True) if close_frames else pd.DataFrame()

        # Cache to disk for reuse
        if not opening_df.empty:
            opening_df.to_parquet(self.raw_dir / "opening_lines.parquet", index=False)
        if not closing_df.empty:
            closing_df.to_parquet(self.raw_dir / "closing_lines.parquet", index=False)

        logger.info(
            "Fetched %d opening and %d closing line rows across %d dates",
            len(opening_df), len(closing_df), len(game_dates),
        )
        return opening_df, closing_df

    def load_cached_snapshots(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously fetched opening and closing snapshots from disk.
        Returns (opening_df, closing_df); empty DataFrames if not found.
        """
        def _load(name: str) -> pd.DataFrame:
            path = self.raw_dir / name
            if path.exists():
                return pd.read_parquet(path)
            logger.debug("No cached snapshot at %s", path)
            return pd.DataFrame()

        return _load("opening_lines.parquet"), _load("closing_lines.parquet")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict) -> list | dict:
        """HTTP GET with error handling."""
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.debug("Odds API requests remaining: %s", remaining)
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                logger.debug("Odds API key invalid or quota exhausted — falling back to Action Network.")
            elif exc.response is not None and exc.response.status_code == 422:
                logger.debug("Historical odds require a paid Odds API plan.")
            else:
                logger.error("HTTP error fetching odds: %s", exc)
            return []
        except requests.RequestException as exc:
            logger.error("Request error fetching odds: %s", exc)
            return []

    @staticmethod
    def _parse_odds(payload: list | dict) -> list[dict]:
        """Normalise Odds API response to flat rows."""
        if isinstance(payload, dict):
            payload = payload.get("data", [])

        rows: list[dict] = []
        for game in payload or []:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            commence = game.get("commence_time", "")

            for bookmaker in game.get("bookmakers", []):
                book_key = bookmaker.get("key", "")
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "spreads":
                        continue
                    outcomes = {o["name"]: o for o in market.get("outcomes", [])}
                    home_outcome = outcomes.get(home, {})
                    away_outcome = outcomes.get(away, {})
                    rows.append(
                        {
                            "game_id": game.get("id"),
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "bookmaker": book_key,
                            "home_spread": home_outcome.get("point"),
                            "home_spread_price": home_outcome.get("price"),
                            "away_spread": away_outcome.get("point"),
                            "away_spread_price": away_outcome.get("price"),
                        }
                    )
        return rows
