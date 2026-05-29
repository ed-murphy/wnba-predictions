"""
src/data/action_network_scraper.py

Fetches historical WNBA betting odds from Action Network's public API
(no API key required).  For each game the API returns the full line history
for every bookmaker — multiple timestamped entries per book — so we can
derive *opening* lines (earliest entry) and *closing* lines (latest entry)
and feed them to LineMovementProcessor to compute steam moves, RLM, etc.

Endpoint
--------
  GET https://api.actionnetwork.com/web/v1/scoreboard/wnba?date=YYYYMMDD

Response (relevant fields)
--------------------------
  games[].id                  : int  — Action Network game id
  games[].start_time          : str  — UTC ISO datetime
  games[].home_team_id        : int
  games[].away_team_id        : int
  games[].teams[].id          : int
  games[].teams[].abbr        : str  — e.g. "MIN" (matches ESPN abbreviation)
  games[].teams[].full_name   : str  — e.g. "Minnesota Lynx"
  games[].odds[].book_id      : int  — bookmaker identifier
  games[].odds[].spread_home  : float | None — home team spread (neg = home fav)
  games[].odds[].inserted     : str  — UTC ISO datetime when line was recorded

Multiple odds entries share the same book_id; sorting by `inserted` gives the
line history.  First entry per book = opening line; last = closing line.

Caching
-------
Raw daily JSON is cached to data/raw/action_network/<YYYYMMDD>.json so we
never re-hit the API for dates we already have.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .line_movement import LineMovementProcessor

logger = logging.getLogger(__name__)

_SCOREBOARD_URL = "https://api.actionnetwork.com/web/v1/scoreboard/wnba"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


class ActionNetworkScraper:
    """
    Downloads WNBA betting lines from Action Network and builds the
    opening/closing snapshots needed by LineMovementProcessor.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        sleep: float = 0.4,
        steam_threshold: float = 2.0,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self._cache_dir = self.raw_dir / "action_network"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._lm = LineMovementProcessor(
            processed_dir=processed_dir, steam_threshold=steam_threshold
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_today_odds(self) -> pd.DataFrame:
        """
        Fetch today's WNBA games and current spreads from Action Network.

        Returns a DataFrame with columns:
            game_id, commence_time, home_team, away_team, home_spread

        The format is compatible with ``_build_upcoming_game_rows`` in
        predict.py so it can be used as a no-key alternative to the Odds API.
        Today's data is always fetched fresh (not from cache) so the spread
        reflects the current line rather than the morning open.
        """
        from datetime import datetime

        today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        raw = self._get(today_str)
        games_list = raw.get("games", [])
        if not games_list:
            return pd.DataFrame()

        rows = []
        for game in games_list:
            home_team_id = game.get("home_team_id")
            away_team_id = game.get("away_team_id")
            teams_by_id = {t["id"]: t for t in game.get("teams", [])}

            home_raw = teams_by_id.get(home_team_id, {})
            away_raw = teams_by_id.get(away_team_id, {})
            if not home_raw or not away_raw:
                continue

            home_abbr = _ABBR_MAP.get(home_raw["abbr"].upper(), home_raw["abbr"].upper())
            away_abbr = _ABBR_MAP.get(away_raw["abbr"].upper(), away_raw["abbr"].upper())

            home_name = _ABBR_TO_NAME.get(home_abbr, home_raw.get("full_name", home_abbr))
            away_name = _ABBR_TO_NAME.get(away_abbr, away_raw.get("full_name", away_abbr))

            # Collect spread, public %, and total across all books
            home_spread = None
            public_pcts: list[float] = []
            totals: list[float] = []
            for entry in game.get("odds", []):
                if entry.get("type") != "game":
                    continue
                if home_spread is None and entry.get("spread_home") is not None:
                    home_spread = float(entry["spread_home"])
                if entry.get("spread_home_public") is not None:
                    public_pcts.append(float(entry["spread_home_public"]))
                if entry.get("total") is not None:
                    totals.append(float(entry["total"]))

            rows.append({
                "game_id": f"an_{game.get('id', len(rows))}",
                "commence_time": game.get("start_time"),
                "home_team": home_name,
                "away_team": away_name,
                "home_spread": home_spread,
                "home_public_pct": float(np.median(public_pcts)) if public_pcts else None,
                "total": float(np.median(totals)) if totals else None,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(
            "Action Network: found %d game(s) today (%s)", len(df), today_str
        )
        return df

    def build_line_movement(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point.  Given an ESPN schedule DataFrame, fetches all
        available Action Network odds, matches games by date + team
        abbreviation, then computes and saves line_movement.parquet.

        Parameters
        ----------
        schedule : DataFrame
            Output of ESPNScraper.fetch_multiple_seasons().  Must contain
            columns: game_id, date, home_team_abbr, away_team_abbr, status.

        Returns
        -------
        DataFrame — line movement table (empty if no odds were found).
        """
        # Work only on completed games
        completed = schedule[
            schedule["status"].str.lower().str.contains("final")
        ].copy()
        if completed.empty:
            logger.warning("No completed games in schedule; nothing to fetch.")
            return pd.DataFrame()

        # Collect unique query dates from the ESPN schedule
        dates = self._extract_dates(completed)
        logger.info("Fetching Action Network odds for %d unique dates …", len(dates))

        raw_rows: list[dict] = []
        for i, d in enumerate(dates):
            if i and i % 50 == 0:
                logger.info("  … processed %d / %d dates", i, len(dates))
            rows = self._fetch_date(d)
            raw_rows.extend(rows)

        if not raw_rows:
            logger.warning("Action Network returned no odds data.")
            return pd.DataFrame()

        odds_raw = pd.DataFrame(raw_rows)
        logger.info("Raw odds rows: %d", len(odds_raw))

        # Match Action Network games → ESPN game_ids
        odds_with_ids = self._match_game_ids(odds_raw, completed)
        if odds_with_ids.empty:
            logger.warning("Could not match any Action Network games to ESPN schedule.")
            return pd.DataFrame()

        matched = odds_with_ids.dropna(subset=["game_id"])
        logger.info(
            "Matched %d / %d Action Network games to ESPN game_ids",
            matched["an_game_id"].nunique(),
            odds_raw["an_game_id"].nunique(),
        )

        # Build opening / closing snapshot DataFrames for LineMovementProcessor
        opening_df, closing_df = self._split_opening_closing(matched)

        # Save closing spreads (fallback) and market context (public %, total)
        self._save_historical_spreads(closing_df)
        self._save_market_context(matched)

        # Compute and persist line movement table
        movement = self._lm.compute(opening_df, closing_df)
        return movement

    # ------------------------------------------------------------------
    # Fetching & caching
    # ------------------------------------------------------------------

    def _fetch_date(self, d: date) -> list[dict]:
        """
        Fetch odds for a single date.  Returns a list of flat dicts, one per
        (an_game_id, book_id, inserted) entry.  Uses per-date JSON cache.
        """
        date_str = d.strftime("%Y%m%d")
        cache_path = self._cache_dir / f"{date_str}.json"

        if cache_path.exists():
            with cache_path.open("r") as fh:
                payload = json.load(fh)
        else:
            payload = self._get(date_str)
            if payload:
                with cache_path.open("w") as fh:
                    json.dump(payload, fh)
            time.sleep(self.sleep)  # only rate-limit live fetches

        return self._parse_payload(payload, d)

    def _get(self, date_str: str) -> dict:
        """HTTP GET with retry + 429 back-off."""
        for attempt in range(4):
            try:
                resp = self._session.get(
                    _SCOREBOARD_URL,
                    params={"date": date_str},
                    timeout=15,
                )
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 2)
                    logger.warning("Rate limited; waiting %ds …", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return {}
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("Request failed (attempt %d/4): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        return {}

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_payload(payload: dict, query_date: date) -> list[dict]:
        """
        Convert a raw Action Network scoreboard response into a flat list of
        dicts with one row per (game, book) — filtering to type='game' only
        so we use the main full-game spread and ignore halftime/quarter/live lines.
        """
        rows: list[dict] = []
        for game in payload.get("games", []):
            odds = game.get("odds")
            if not odds:
                continue

            an_id = game.get("id")
            home_team_id = game.get("home_team_id")
            away_team_id = game.get("away_team_id")
            teams = {t["id"]: t for t in game.get("teams", [])}

            home_info = teams.get(home_team_id, {})
            away_info = teams.get(away_team_id, {})
            home_abbr = home_info.get("abbr", "")
            away_abbr = away_info.get("abbr", "")

            if not home_abbr or not away_abbr:
                continue

            for entry in odds:
                # Only use the main full-game spread lines
                if entry.get("type") != "game":
                    continue
                spread = entry.get("spread_home")
                if spread is None:
                    continue
                pub = entry.get("spread_home_public")
                total = entry.get("total")
                rows.append(
                    {
                        "query_date": query_date,
                        "an_game_id": an_id,
                        "home_abbr": home_abbr,
                        "away_abbr": away_abbr,
                        "book_id": entry.get("book_id"),
                        "home_spread": float(spread),
                        "inserted": entry.get("inserted"),
                        "spread_home_public": float(pub) if pub is not None else None,
                        "total": float(total) if total is not None else None,
                    }
                )
        return rows

    # ------------------------------------------------------------------
    # Game-ID matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_game_ids(
        odds_raw: pd.DataFrame, schedule: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Map each Action Network game to an ESPN game_id by matching on
        (eastern_date, home_abbr, away_abbr).

        ESPN dates are UTC ISO strings; we convert to US/Eastern so they
        align with Action Network's Eastern-time date convention.
        """
        _ET = ZoneInfo("America/New_York")
        sched = schedule[
            ["game_id", "date", "home_team_abbr", "away_team_abbr"]
        ].copy()
        sched["_date"] = (
            pd.to_datetime(sched["date"], utc=True)
            .dt.tz_convert(_ET)
            .dt.date
        )

        # Build a lookup: (date, home_abbr, away_abbr) -> game_id
        lookup: dict[tuple, str] = {}
        for _, row in sched.iterrows():
            key = (row["_date"], row["home_team_abbr"].upper(), row["away_team_abbr"].upper())
            lookup[key] = str(row["game_id"])

        def resolve(row: pd.Series) -> str | None:
            home = _ABBR_MAP.get(row["home_abbr"].upper(), row["home_abbr"].upper())
            away = _ABBR_MAP.get(row["away_abbr"].upper(), row["away_abbr"].upper())
            # Try exact date first, then ±1 day for late-night UTC edge cases
            for delta in (0, 1, -1):
                d = row["query_date"] + timedelta(days=delta)
                gid = lookup.get((d, home, away))
                if gid:
                    return gid
            return None

        odds_copy = odds_raw.copy()
        odds_copy["home_abbr"] = odds_copy["home_abbr"].str.upper()
        odds_copy["away_abbr"] = odds_copy["away_abbr"].str.upper()
        odds_copy["game_id"] = odds_copy.apply(resolve, axis=1)
        return odds_copy

    # ------------------------------------------------------------------
    # Opening / closing split
    # ------------------------------------------------------------------

    @staticmethod
    def _split_opening_closing(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        The Action Network scoreboard endpoint provides exactly one type='game'
        entry per bookmaker per game — this represents the closing line (final
        pre-game spread as recorded in their system).

        To derive an *opening* line proxy we use the bookmaker that posted
        EARLIEST (lowest inserted timestamp) as the opening consensus, and the
        bookmaker that posted LATEST as the closing consensus.  When multiple
        books are present, this captures genuine early-vs-late market movement.

        Returns (opening_df, closing_df) each with columns:
            game_id, home_team, away_team, bookmaker, home_spread
        """
        df = df.copy()
        df["inserted_dt"] = pd.to_datetime(df["inserted"], utc=True, errors="coerce")
        df = df.dropna(subset=["inserted_dt", "home_spread", "game_id"])

        opening_rows: list[dict] = []
        closing_rows: list[dict] = []

        for gid, grp in df.groupby("game_id"):
            grp_sorted = grp.sort_values("inserted_dt")
            n = len(grp_sorted)

            # Use the earliest 50% of books as the "opening" snapshot and
            # all books as the "closing" snapshot.
            # With only 1 book, open == close; with 6 books, earliest 3 = open.
            split = max(1, n // 2)
            open_rows = grp_sorted.head(split)
            close_rows = grp_sorted  # all books for consensus closing

            home = grp_sorted.iloc[0]["home_abbr"]
            away = grp_sorted.iloc[0]["away_abbr"]

            for _, r in open_rows.iterrows():
                opening_rows.append(
                    {
                        "game_id": gid,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": str(r["book_id"]),
                        "home_spread": r["home_spread"],
                    }
                )
            for _, r in close_rows.iterrows():
                closing_rows.append(
                    {
                        "game_id": gid,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": str(r["book_id"]),
                        "home_spread": r["home_spread"],
                    }
                )

        return pd.DataFrame(opening_rows), pd.DataFrame(closing_rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _ET = ZoneInfo("America/New_York")

    @classmethod
    def _extract_dates(cls, schedule: pd.DataFrame) -> list[date]:
        """Return sorted unique game dates converted from UTC to US/Eastern."""
        # ESPN stores dates as UTC ISO strings; convert to ET to match Action
        # Network's date parameter convention (e.g. 2024-05-04T00:00Z = May 3 ET).
        dates = (
            pd.to_datetime(schedule["date"], utc=True)
            .dt.tz_convert(cls._ET)
            .dt.date.unique()
        )
        return sorted(dates)

    def _save_historical_spreads(self, closing_df: pd.DataFrame) -> None:
        """
        Save median closing spread per game as historical_spreads.parquet.
        This acts as fallback odds data when line_movement isn't available.
        """
        if closing_df.empty:
            return
        spreads = (
            closing_df.groupby("game_id")["home_spread"]
            .median()
            .reset_index()
            .rename(columns={"home_spread": "spread"})
        )
        out = self.raw_dir / "historical_spreads.parquet"
        spreads.to_parquet(out, index=False)
        logger.info(
            "Saved closing spreads for %d games to %s", len(spreads), out
        )

    def _save_market_context(self, df: pd.DataFrame) -> None:
        """
        Save per-game public betting % and over/under total to
        data/processed/market_context.parquet.

        home_public_pct  : median % of public spread bets on home team (0–100).
                           Values < 40 suggest public fades home; > 60 = public hammers home.
        away_public_pct  : 100 − home_public_pct (convenience column).
        total            : consensus over/under total (median across books).
        total_open       : total line from the earliest-inserted book (opening proxy).
        total_close      : total line from the latest-inserted book (closing).
        total_move       : total_close − total_open (positive = total moved up).
        """
        if df.empty:
            return

        # Base aggregations
        base_agg = (
            df.groupby("game_id")
            .agg(
                home_public_pct=("spread_home_public", "median"),
                total=("total", "median"),
            )
            .reset_index()
        )
        base_agg["away_public_pct"] = 100.0 - base_agg["home_public_pct"]

        # Total line open/close split (earliest vs latest insertion per game)
        if "inserted" in df.columns and "total" in df.columns:
            df_t = df.dropna(subset=["total", "inserted"]).copy()
            df_t["inserted_dt"] = pd.to_datetime(df_t["inserted"], utc=True, errors="coerce")
            df_t = df_t.dropna(subset=["inserted_dt"])

            open_tot = (
                df_t.sort_values("inserted_dt")
                .groupby("game_id")["total"]
                .first()
                .rename("total_open")
            )
            close_tot = (
                df_t.sort_values("inserted_dt")
                .groupby("game_id")["total"]
                .last()
                .rename("total_close")
            )
            tot_movement = pd.concat([open_tot, close_tot], axis=1).reset_index()
            tot_movement["total_move"] = tot_movement["total_close"] - tot_movement["total_open"]
            base_agg = base_agg.merge(tot_movement, on="game_id", how="left")

        out = self.processed_dir / "market_context.parquet"
        base_agg.to_parquet(out, index=False)
        logger.info(
            "Saved market context (public %%, total, total_move) for %d games to %s",
            len(base_agg), out,
        )


# ---------------------------------------------------------------------------
# Team abbreviation normalisation
# ---------------------------------------------------------------------------
# Action Network occasionally uses a different abbreviation than ESPN.
# Keys = Action Network abbr (upper), Values = ESPN abbr (upper).
_ABBR_MAP: dict[str, str] = {
    # Action Network abbreviation → ESPN abbreviation
    "CONN": "CON",  # Connecticut Sun (AN uses CONN, ESPN uses CON)
    "LVA": "LV",    # Las Vegas Aces  (AN uses LVA, ESPN uses LV)
    "LAS": "LA",    # Los Angeles Sparks (some books use LAS)
    "GSV": "GS",    # Golden State Valkyries (2025, if ESPN uses GS)
}

# ESPN abbreviation → ESPN displayName (for building upcoming game rows)
_ABBR_TO_NAME: dict[str, str] = {
    "ATL": "Atlanta Dream",
    "CHI": "Chicago Sky",
    "CON": "Connecticut Sun",
    "DAL": "Dallas Wings",
    "GS":  "Golden State Valkyries",
    "IND": "Indiana Fever",
    "LA":  "Los Angeles Sparks",
    "LV":  "Las Vegas Aces",
    "MIN": "Minnesota Lynx",
    "NY":  "New York Liberty",
    "PHX": "Phoenix Mercury",
    "SEA": "Seattle Storm",
    "WSH": "Washington Mystics",
}
