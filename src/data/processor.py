"""
src/data/processor.py

Merges raw ESPN game data with odds data, cleans columns, and produces a
single analytics-ready DataFrame for feature engineering.

Key output columns
------------------
game_id, date, season, home_team, away_team,
home_score, away_score, margin (home - away),
spread (home team closing spread line, e.g. -4.5 means home favoured by 4.5),
ats_result (1 = home covered, 0 = away covered),
total_points, home_win (1/0)

Line movement columns (present when line_movement data is supplied)
-------------------------------------------------------------------
spread_open, spread_close, line_move, abs_line_move,
steam_move, move_direction, pct_books_agree,
n_books_open, n_books_close,
line_range_open, line_range_close
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """Merges and cleans ESPN + odds data into a modelling-ready table."""

    def __init__(self, processed_dir: str | Path = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_games_table(
        self,
        schedule: pd.DataFrame,
        odds: pd.DataFrame | None = None,
        line_movement: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Combine schedule (from ESPNScraper) and optional odds / line movement
        data into a clean games table. Returns only completed games.

        Parameters
        ----------
        schedule : DataFrame
            Raw schedule from ESPNScraper.fetch_multiple_seasons().
        odds : DataFrame, optional
            Closing-line odds from OddsAPIClient. Used to compute spread and
            ats_result when line_movement is not supplied.
        line_movement : DataFrame, optional
            Output of LineMovementProcessor.compute(). When provided, the
            closing spread from here is used for ats_result and a full suite
            of movement features is added to the games table.
        """
        df = schedule.copy()

        # Keep only final games (ESPN uses "status_final"; also accept plain "final")
        if "status" in df.columns:
            df = df[df["status"].str.lower().str.contains("final")].copy()

        df = self._normalize_types(df)
        df = self._add_derived_columns(df)

        if line_movement is not None and not line_movement.empty:
            df = self._merge_line_movement(df, line_movement)
        elif odds is not None and not odds.empty:
            df = self._merge_odds(df, odds)

        # Merge market context (public betting %, game total) if available on disk
        mc_path = self.processed_dir / "market_context.parquet"
        if mc_path.exists():
            mc = pd.read_parquet(mc_path)
            if not mc.empty:
                df = df.merge(mc, on="game_id", how="left")

        # Over/Under result: 1 = total points exceeded the line, 0 = under, NaN = push/no line
        if "total" in df.columns and "total_points" in df.columns:
            tp = pd.to_numeric(df["total_points"], errors="coerce")
            tl = pd.to_numeric(df["total"], errors="coerce")
            df["over_result"] = np.where(
                tl.isna(),
                np.nan,
                np.where(tp > tl, 1.0, np.where(tp == tl, np.nan, 0.0)),
            )

        out_path = self.processed_dir / "games.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Games table saved to %s (%d rows)", out_path, len(df))
        return df

    def build_team_game_log(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape the wide games table into a long team-game log where each row
        represents one team's performance in one game.
        """
        home = games.rename(
            columns={
                "home_team": "team",
                "home_team_id": "team_id",
                "home_team_abbr": "team_abbr",
                "home_score": "pts_for",
                "away_team": "opponent",
                "away_team_id": "opp_id",
                "away_score": "pts_against",
            }
        ).copy()
        home["is_home"] = 1

        away = games.rename(
            columns={
                "away_team": "team",
                "away_team_id": "team_id",
                "away_team_abbr": "team_abbr",
                "away_score": "pts_for",
                "home_team": "opponent",
                "home_team_id": "opp_id",
                "home_score": "pts_against",
            }
        ).copy()
        away["is_home"] = 0

        keep = [
            "game_id", "date", "season", "team", "team_id", "team_abbr",
            "opponent", "opp_id", "pts_for", "pts_against", "is_home",
        ]
        # Add optional columns if present
        for col in ["spread", "ats_result", "home_win", "total_points", "venue"]:
            if col in games.columns:
                keep.append(col)

        home_cols = [c for c in keep if c in home.columns]
        away_cols = [c for c in keep if c in away.columns]

        log = pd.concat(
            [home[home_cols], away[away_cols]], ignore_index=True
        ).sort_values(["date", "game_id"]).reset_index(drop=True)

        log["win"] = (log["pts_for"] > log["pts_against"]).astype(int)
        log["margin"] = log["pts_for"] - log["pts_against"]

        out_path = self.processed_dir / "team_game_log.parquet"
        log.to_parquet(out_path, index=False)
        logger.info("Team game log saved to %s (%d rows)", out_path, len(log))
        return log

    def _compute_rtg_in_log(self, log: pd.DataFrame) -> pd.DataFrame:
        """
        After boxscores are merged into the game log, compute OffRtg/DefRtg/NetRtg.

        OffRtg = 100 * pts_for  / poss
        DefRtg = 100 * pts_against / opp_poss
        NetRtg = OffRtg - DefRtg

        `poss` and `opp_poss` are computed in merge_boxscores via _compute_advanced_box_stats.
        """
        if "poss" not in log.columns:
            return log
        log = log.copy()
        poss_c = log["poss"].clip(lower=1)
        log["off_rtg"] = 100.0 * log["pts_for"].astype(float) / poss_c
        if "opp_poss" in log.columns:
            opp_poss_c = log["opp_poss"].clip(lower=1)
            log["def_rtg"] = 100.0 * log["pts_against"].astype(float) / opp_poss_c
        else:
            log["def_rtg"] = 100.0 * log["pts_against"].astype(float) / poss_c
        log["net_rtg"] = log["off_rtg"] - log["def_rtg"]
        return log

    def merge_boxscores(
        self, game_log: pd.DataFrame, boxscores: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge per-team box score stats (from ESPNScraper.fetch_all_boxscores)
        into the team game log.

        Joins on (game_id, is_home) since boxscores carry a home_away column.
        All stat values are coerced to numeric; unparseable values become NaN.

        Also derives advanced efficiency stats (Four Factors, Pace, Rtg) that
        require both teams' raw counts from the same game.
        """
        if boxscores.empty:
            return game_log

        box = boxscores.copy()
        box["is_home"] = (box["home_away"] == "home").astype(int)

        key_cols = {"game_id", "team_id", "team_abbr", "home_away", "is_home"}
        stat_cols = [c for c in box.columns if c not in key_cols]

        # Parse "made-attempted" string columns (e.g. "29-64") into numeric made/attempted
        made_attempted_cols = [c for c in stat_cols if "-" in c and "Made" in c]
        for col in made_attempted_cols:
            # e.g. "fieldGoalsMade-fieldGoalsAttempted" → fgm, fga
            parts = col.split("-")
            if len(parts) == 2:
                made_name, att_name = parts
                made_vals = box[col].str.split("-", expand=True)
                box[made_name] = pd.to_numeric(made_vals[0], errors="coerce")
                box[att_name] = pd.to_numeric(made_vals[1], errors="coerce")

        # Remove the combined string columns
        box = box.drop(columns=made_attempted_cols, errors="ignore")

        # Refresh stat_cols after transformation
        stat_cols = [c for c in box.columns if c not in key_cols and c not in made_attempted_cols]

        for col in stat_cols:
            box[col] = pd.to_numeric(box[col], errors="coerce")

        # Drop columns that are entirely NaN (no useful data)
        stat_cols = [c for c in stat_cols if box[c].notna().any()]

        # --- Derive Four Factors + Pace/Rtg (need both teams' counts per game) ---
        box = self._compute_advanced_box_stats(box)
        advanced_cols = [
            c for c in box.columns
            if c not in key_cols and c not in made_attempted_cols
            and c not in stat_cols
        ]
        stat_cols = stat_cols + advanced_cols

        log = game_log.merge(
            box[["game_id", "is_home"] + stat_cols],
            on=["game_id", "is_home"],
            how="left",
        )

        # Compute OffRtg / DefRtg / NetRtg now that pts_for/pts_against are present
        log = self._compute_rtg_in_log(log)

        out_path = self.processed_dir / "team_game_log.parquet"
        log.to_parquet(out_path, index=False)
        logger.info(
            "Merged %d box score stat(s) (incl. Four Factors & Rtg) into game log (%d rows)",
            len(stat_cols), len(log),
        )
        return log

    @staticmethod
    def _compute_advanced_box_stats(box: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced efficiency stats (Four Factors, Pace, Offensive/Defensive Rtg)
        for each team row. Requires both teams' stats to be present for DefRtg.

        Four Factors (Dean Oliver):
          eFG%    = (FGM + 0.5 * FG3M) / FGA
          TOV%    = TOV / (FGA + 0.44 * FTA + TOV)
          FTA_rate = FTA / FGA
          ORB%    = ORB / (ORB + opp_DRB)   [requires cross-team join]

        Pace & Ratings:
          poss    ≈ FGA - ORB + TOV + 0.44 * FTA
          OffRtg  = 100 * pts_for / poss
          DefRtg  = 100 * pts_against / opp_poss
          NetRtg  = OffRtg - DefRtg
          Pace    = avg(poss, opp_poss) — a game-level constant per team row
        """
        box = box.copy()

        fgm = box.get("fieldGoalsMade", pd.Series(np.nan, index=box.index))
        fga = box.get("fieldGoalsAttempted", pd.Series(np.nan, index=box.index))
        fg3m = box.get("threePointFieldGoalsMade", pd.Series(np.nan, index=box.index))
        fta = box.get("freeThrowsAttempted", pd.Series(np.nan, index=box.index))
        tov = box.get("totalTurnovers", box.get("turnovers", pd.Series(np.nan, index=box.index)))
        orb = box.get("offensiveRebounds", pd.Series(np.nan, index=box.index))
        drb = box.get("defensiveRebounds", pd.Series(np.nan, index=box.index))

        fga_c = fga.clip(lower=1)
        fta_n = fta.fillna(0)
        tov_n = tov.fillna(0)
        orb_n = orb.fillna(0)

        # Four Factors
        box["efg_pct"] = (fgm.fillna(0) + 0.5 * fg3m.fillna(0)) / fga_c
        box["tov_pct"] = tov_n / (fga_c + 0.44 * fta_n + tov_n).clip(lower=1)
        box["fta_rate"] = fta_n / fga_c

        # Possessions estimate
        poss = fga - orb_n + tov_n + 0.44 * fta_n
        box["poss"] = poss.clip(lower=1)

        # ORB% requires opponent DRB — join within game
        if "game_id" in box.columns:
            # Build a map: game_id → opponent DRB (opponent = the other team in the game)
            opp_drb = (
                box.groupby("game_id")
                .apply(lambda g: g.set_index("is_home")["defensiveRebounds"].to_dict()
                       if "defensiveRebounds" in g.columns else {})
            )

            def _opp_drb(row):
                gid = row["game_id"]
                opp = 1 - row["is_home"]  # 0→1 or 1→0
                try:
                    return opp_drb[gid].get(opp, np.nan)
                except (KeyError, AttributeError):
                    return np.nan

            box["opp_drb"] = box.apply(_opp_drb, axis=1)
            orb_denom = (orb_n + box["opp_drb"].fillna(orb_n)).clip(lower=1)
            box["orb_pct"] = orb_n / orb_denom

            # Opponent possessions for DefRtg
            opp_poss_map = (
                box.groupby("game_id")
                .apply(lambda g: g.set_index("is_home")["poss"].to_dict()
                       if "poss" in g.columns else {})
            )

            def _opp_poss(row):
                gid = row["game_id"]
                opp = 1 - row["is_home"]
                try:
                    return opp_poss_map[gid].get(opp, np.nan)
                except (KeyError, AttributeError):
                    return np.nan

            box["opp_poss"] = box.apply(_opp_poss, axis=1)
        else:
            box["orb_pct"] = np.nan
            box["opp_poss"] = np.nan

        # OffRtg, DefRtg, NetRtg
        # pts_for comes from the game_log after merge, so approximate with fieldGoalsMade*2 + fg3m + ftm
        # We store raw for now and compute rtg in the game_log step
        box["pace"] = (box["poss"] + box["opp_poss"].fillna(box["poss"])) / 2

        return box

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        for col in ["home_score", "away_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    @staticmethod
    def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "home_score" in df.columns and "away_score" in df.columns:
            df["margin"] = df["home_score"] - df["away_score"]
            df["total_points"] = df["home_score"] + df["away_score"]
            df["home_win"] = (df["margin"] > 0).astype(int)
        return df

    @staticmethod
    def _merge_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
        """
        Left-join odds onto the games table.
        Consensus spread = median across bookmakers.
        """
        if odds.empty:
            return games

        # Compute consensus (median) spread per game
        consensus = (
            odds.groupby("game_id")["home_spread"]
            .median()
            .reset_index()
            .rename(columns={"home_spread": "spread"})
        )

        merged = games.merge(consensus, on="game_id", how="left")
        merged = DataProcessor._compute_ats_result(merged, spread_col="spread")
        return merged

    @staticmethod
    def _merge_line_movement(
        games: pd.DataFrame, line_movement: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge a LineMovementProcessor output table onto the games table.

        Uses spread_close as the authoritative spread for ATS calculation,
        and attaches all movement columns (line_move, steam_move, etc.).
        """
        if line_movement.empty:
            return games

        # All movement columns except game_id
        movement_cols = [c for c in line_movement.columns if c != "game_id"]

        merged = games.merge(
            line_movement[["game_id"] + movement_cols],
            on="game_id",
            how="left",
        )

        # spread_close is the closing line — use it as the canonical spread
        if "spread_close" in merged.columns:
            merged["spread"] = merged["spread_close"]
            merged = DataProcessor._compute_ats_result(merged, spread_col="spread_close")

        return merged

    @staticmethod
    def _compute_ats_result(df: pd.DataFrame, spread_col: str = "spread") -> pd.DataFrame:
        """Compute ats_result from margin and a spread column (in-place copy)."""
        df = df.copy()
        if "margin" in df.columns and spread_col in df.columns:
            cover = df["margin"] + df[spread_col]
            # Rows where spread is missing → NaN (not False)
            df["ats_result"] = np.where(
                df[spread_col].isna(),
                np.nan,
                np.where(cover > 0, 1.0, np.where(cover == 0, np.nan, 0.0)),
            )
        return df
