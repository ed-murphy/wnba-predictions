"""
src/data/espn_scraper.py

Fetches WNBA game data from ESPN's public (undocumented) JSON API.
No API key required. Data includes scores, team stats, and game metadata.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ESPN public API endpoints
_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
_SCOREBOARD = f"{_BASE}/scoreboard"
_SUMMARY = f"{_BASE}/summary"
_TEAMS = f"{_BASE}/teams"


class ESPNScraper:
    """Pulls WNBA game logs and box-score data from ESPN's public API."""

    def __init__(self, raw_dir: str | Path = "data/raw", sleep: float = 0.5) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "wnba-predictions-research/1.0"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_season_schedule(self, season: int, cache_ttl_hours: float = 4.0) -> pd.DataFrame:
        """Return every completed regular-season game for *season*.

        Past seasons are cached permanently.  The current season is cached
        with a TTL of ``cache_ttl_hours`` (default 4 h) so that repeated
        ``predict`` runs within a day skip the full ESPN re-scrape while still
        picking up last night's completed games on the first run of the day.
        """
        import datetime as _dt
        current_year = _dt.date.today().year
        cache = self.raw_dir / f"schedule_{season}.parquet"

        if cache.exists():
            if season < current_year:
                logger.info("Loading schedule from cache: %s", cache)
                return pd.read_parquet(cache)
            # Current season: use cache if it is fresh enough
            age_hours = (_dt.datetime.now().timestamp() - cache.stat().st_mtime) / 3600
            if age_hours < cache_ttl_hours:
                logger.info(
                    "Loading current-season schedule from cache (%.1f h old): %s",
                    age_hours, cache,
                )
                return pd.read_parquet(cache)

        logger.info("Fetching %d season schedule from ESPN …", season)
        games: list[dict] = []

        # Fetch the first page to get the calendar entries ESPN provides
        params = {
            "limit": 100,
            "dates": f"{season}",
            "seasontype": 2,  # regular season
        }
        resp = self._get(_SCOREBOARD, params=params)
        games.extend(self._parse_scoreboard(resp))

        # ESPN's scoreboard returns a calendar of date strings for the season.
        # Each entry is an ISO datetime; extract YYYYMMDD for the dates param.
        calendar_entries: list[str] = []
        for cal in resp.get("leagues", [{}])[0].get("calendar", []):
            if isinstance(cal, str):
                val = cal[:10].replace("-", "")  # "2021-05-05T07:00Z" -> "20210505"
            elif isinstance(cal, dict):
                raw = cal.get("value", "") or cal.get("startDate", "")
                val = raw[:10].replace("-", "")
            else:
                continue
            if val and val != params["dates"]:
                calendar_entries.append(val)

        seen_ids: set[str] = {g["game_id"] for g in games if g.get("game_id")}
        for date_str in calendar_entries:
            time.sleep(self.sleep)
            page_params = {**params, "dates": date_str}
            page_resp = self._get(_SCOREBOARD, params=page_params)
            for game in self._parse_scoreboard(page_resp):
                if game.get("game_id") not in seen_ids:
                    games.append(game)
                    seen_ids.add(game["game_id"])

        df = pd.DataFrame(games)
        if not df.empty:
            df.to_parquet(cache, index=False)
        logger.info("Fetched %d games for %d season", len(df), season)
        return df

    def fetch_game_boxscore(self, game_id: str) -> list[dict]:
        """Return team-level box-score stats for a single game as a list of dicts."""
        cache = self.raw_dir / "boxscores" / f"{game_id}.parquet"
        cache.parent.mkdir(parents=True, exist_ok=True)

        if cache.exists():
            return pd.read_parquet(cache).to_dict(orient="records")

        time.sleep(self.sleep)
        resp = self._get(_SUMMARY, params={"event": game_id})
        rows = self._parse_boxscore(game_id, resp)
        if rows:
            pd.DataFrame(rows).to_parquet(cache, index=False)
        return rows

    def fetch_all_boxscores(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Fetch box-scores for all completed games in *schedule*."""
        completed = schedule[schedule["status"].str.lower().str.contains("final")]
        records: list[dict] = []

        for _, row in completed.iterrows():
            gid = str(row["game_id"])
            rows = self.fetch_game_boxscore(gid)
            records.extend(rows)

        df = pd.DataFrame(records)
        out = self.raw_dir / "all_boxscores.parquet"
        if not df.empty:
            df.to_parquet(out, index=False)
        return df

    def fetch_multiple_seasons(self, seasons: list[int]) -> pd.DataFrame:
        """Fetch and concatenate schedules for multiple seasons."""
        frames = [self.fetch_season_schedule(s) for s in seasons]
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> dict:
        """HTTP GET with retry and 429 back-off logic."""
        for attempt in range(4):
            try:
                resp = self._session.get(url, params=params, timeout=20)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rate limited (429); waiting %ds …", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("Request failed (attempt %d/4): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        return {}

    @staticmethod
    def _parse_scoreboard(payload: dict) -> list[dict]:
        """Extract game metadata rows from a scoreboard response."""
        rows = []
        for event in payload.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])

            if len(competitors) != 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})

            status_type = comp.get("status", {}).get("type", {})
            rows.append(
                {
                    "game_id": event.get("id"),
                    "date": event.get("date"),
                    "season": event.get("season", {}).get("year"),
                    "status": status_type.get("name", "").lower(),
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team": home.get("team", {}).get("displayName"),
                    "home_team_abbr": home.get("team", {}).get("abbreviation"),
                    "home_score": int(home.get("score", 0) or 0),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team": away.get("team", {}).get("displayName"),
                    "away_team_abbr": away.get("team", {}).get("abbreviation"),
                    "away_score": int(away.get("score", 0) or 0),
                    "venue": comp.get("venue", {}).get("fullName"),
                    "neutral_site": comp.get("neutralSite", False),
                }
            )
        return rows

    @staticmethod
    def _parse_boxscore(game_id: str, payload: dict) -> list[dict]:
        """Extract team-level box-score rows from a summary response."""
        rows = []
        boxscore = payload.get("boxscore", {})
        teams = boxscore.get("teams", [])

        for team_data in teams:
            team = team_data.get("team", {})
            stats: dict = {}
            for s in team_data.get("statistics", []):
                name = s.get("name") or s.get("abbreviation")
                if name:
                    stats[name] = s.get("displayValue") or s.get("value")

            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team.get("id"),
                    "team_abbr": team.get("abbreviation"),
                    "home_away": team_data.get("homeAway"),
                    **stats,
                }
            )
        return rows

    def fetch_injuries(self) -> dict[str, list[dict]]:
        """Return current injury reports for all WNBA teams.

        Queries the live ESPN injuries endpoint (current-day data only — no
        historical archive is available from this endpoint).

        Returns
        -------
        dict
            Keys are team displayName strings (matching the `home_team` /
            `away_team` columns in the games table).  Values are lists of dicts:
            ``[{"name": str, "position": str, "status": str}, ...]``
            ``status`` is typically ``"Out"``, ``"Questionable"``, or
            ``"Doubtful"``.

        Returns an empty dict on any network or parse failure so that the
        prediction workflow degrades gracefully.
        """
        url = f"{_BASE}/injuries"
        try:
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Could not fetch injury data from ESPN: %s", exc)
            return {}

        result: dict[str, list[dict]] = {}
        for team_entry in data.get("injuries", []):
            team_info = team_entry.get("team", {})
            team_name = team_info.get("displayName", "")
            if not team_name:
                continue
            players: list[dict] = []
            for injury in team_entry.get("injuries", []):
                athlete = injury.get("athlete", {})
                pos_info = athlete.get("position", {}) or {}
                player: dict = {
                    "name": athlete.get("displayName", "Unknown"),
                    "position": pos_info.get("abbreviation", ""),
                    "status": injury.get("status", ""),
                    "comment": injury.get("longComment", ""),
                }
                players.append(player)
            if players:
                result[team_name] = players
        return result
