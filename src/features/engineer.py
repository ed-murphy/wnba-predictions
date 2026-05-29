"""
src/features/engineer.py

Builds a modelling-ready feature matrix from the team game log produced by
DataProcessor.

Feature groups
--------------
1.  Season-to-date averages   — cumulative stats up to (not including) current game.
2.  Rolling form              — rolling mean over last N games (default 3, 5, 10).
3.  Rest / schedule           — days since last game, back-to-back flags.
4.  Home-court advantage      — is_home flag.
5.  Head-to-head              — season H2H record.
6.  Spread context            — opening/closing spread, size flags.
7.  Line movement             — magnitude, direction, steam flag, book consensus.
8.  Four Factors              — eFG%, TOV%, FTA_rate, ORB% (Dean Oliver).
9.  Pace & Efficiency Ratings — OffRtg, DefRtg, NetRtg, Pace (per-possession).
10. Pythagorean expectation   — pts^exp / (pts^exp + opp_pts^exp).
11. Opponent-adjusted stats   — efficiency vs opponent defensive quality.
12. Schedule density          — games played in last 7 days.
13. Consistency               — rolling std dev for key metrics.
14. Reverse Line Movement     — line moved opposite to public betting direction.
15. Totals context            — combined pace/scoring for O/U modelling.
16. Home/Away splits          — ATS cover rates at home vs away.
17. Over/Under tracking       — team over rate history.

Output
------
One row per game. Features are named with home_ / away_ prefix.
Target columns:
  - ats_result   : 1 = home covered, 0 = away covered (NaN = push)
  - over_result  : 1 = total exceeded line, 0 = under (NaN = push/no line)
  - home_win     : 1 = home team won outright
  - margin       : home_score - away_score
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

logger = logging.getLogger(__name__)

_DEFAULT_WINDOWS = [3, 5, 10]

# Stats computed per team per game (from score data)
_TEAM_STATS = [
    "pts_for",
    "pts_against",
    "margin",
    "win",
]

# Box score stats to include when available (ESPN API column names).
# NOTE: Raw counts (FGM, FGA, etc.) are intentionally excluded because we
# already have the derived rates (FG%, eFG%, FTA_rate) which carry the same
# information without the high collinearity with volume stats.
_BOX_SCORE_STATS = [
    # Shooting percentages (rate stats — lower collinearity than raw counts)
    "fieldGoalPct",
    "threePointFieldGoalPct",
    "freeThrowPct",
    # Rebounding (split; totalRebounds excluded as it equals ORB+DRB)
    "offensiveRebounds",
    "defensiveRebounds",
    # Playmaking & ball security
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    # Scoring breakdown
    "fastBreakPoints",
    "pointsInPaint",
    "largestLead",
    "leadChanges",
    # Four Factors (derived in DataProcessor.merge_boxscores)
    "efg_pct",
    "tov_pct",
    "fta_rate",
    "orb_pct",
    "poss",
    # Efficiency ratings (derived in DataProcessor._compute_rtg_in_log)
    "off_rtg",
    "def_rtg",
    "net_rtg",
    "pace",
]


class FeatureEngineer:
    """Transforms a team game log into a game-level feature matrix."""

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        windows: list[int] | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.windows = windows or _DEFAULT_WINDOWS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features(
        self,
        game_log: pd.DataFrame,
        games: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the full feature matrix.

        Parameters
        ----------
        game_log : DataFrame
            Long-format team game log from DataProcessor.build_team_game_log().
        games : DataFrame
            Wide-format games table from DataProcessor.build_games_table().

        Returns
        -------
        DataFrame with one row per game and all engineered features.
        """
        log = game_log.sort_values(["team", "date"]).copy()

        # 1. Per-team rolling / cumulative features
        team_features = self._compute_team_features(log)

        # 2. Join home and away team features onto the games table
        feature_df = self._join_to_games(games, team_features)

        # 3. Spread-level features
        if "spread" in feature_df.columns:
            s = feature_df["spread"].fillna(0)
            feature_df["abs_spread"] = s.abs()
            feature_df["home_is_favorite"] = (s < 0).astype(float)
            feature_df["spread_sq"] = s ** 2
            feature_df["big_spread"] = (s.abs() >= 7).astype(float)
            feature_df["pick_em"] = (s.abs() <= 1).astype(float)
            # Market-implied home win probability (σ ≈ 12 pts for WNBA margins)
            # P(home wins) ≈ Φ(−spread / σ); negative spread = home favored
            feature_df["home_implied_prob"] = _norm.cdf(-s / 12.0)
            # How much edge our team-based pyth estimate has vs the market
            if "home_season_pyth_exp" in feature_df.columns:
                feature_df["pyth_vs_market"] = (
                    feature_df["home_season_pyth_exp"] - feature_df["home_implied_prob"]
                )

        # 4. Rest & schedule
        feature_df = self._add_rest_days(feature_df, log)

        # 5. Head-to-head
        feature_df = self._add_h2h(feature_df, log)

        # 6. Differential features (home − away relative strength)
        feature_df = self._add_differential_features(feature_df)

        # 7. Opponent-adjusted efficiency
        feature_df = self._add_opponent_adjusted(feature_df)

        # 8. Totals context (for O/U modelling)
        feature_df = self._add_totals_context(feature_df, games)

        # 9. Line movement + Reverse Line Movement
        feature_df = self._add_line_movement(feature_df, games)

        # 10. Travel distance (away team's home arena → game venue)
        feature_df = self._add_travel_distance(feature_df)

        # 11. Public betting context (directly from games table)
        for col in ["home_public_pct", "total"]:
            if col in games.columns and col not in feature_df.columns:
                feature_df = feature_df.merge(
                    games[["game_id", col]], on="game_id", how="left"
                )
        if "home_public_pct" in feature_df.columns:
            # Extreme public betting = potential contrarian fade signal
            feature_df["public_pct_extreme"] = (
                (feature_df["home_public_pct"] > 65) |
                (feature_df["home_public_pct"] < 35)
            ).astype(float)

        # 11. Ensure target columns are retained
        for col in ["ats_result", "over_result", "home_win", "margin", "spread", "total"]:
            if col in games.columns and col not in feature_df.columns:
                feature_df = feature_df.merge(
                    games[["game_id", col]], on="game_id", how="left"
                )

        out_path = self.processed_dir / "features.parquet"
        feature_df.to_parquet(out_path, index=False)
        logger.info(
            "Feature matrix saved to %s (%d rows, %d cols)",
            out_path, len(feature_df), feature_df.shape[1],
        )
        return feature_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_team_features(self, log: pd.DataFrame) -> pd.DataFrame:
        """For each team-game, compute rolling/cumulative stats."""
        frames: list[pd.DataFrame] = []

        # Include box score stats when present in the log
        box_stats = [s for s in _BOX_SCORE_STATS if s in log.columns]
        all_stats = _TEAM_STATS + box_stats

        for team, grp in log.groupby("team"):
            grp = grp.sort_values("date").copy()

            # Build all feature columns as a dict first, then construct DataFrame once.
            # This avoids the "fragmented DataFrame" PerformanceWarning.
            cols: dict = {
                "game_id": grp["game_id"].values,
                "team": grp["team"].values,
                "date": grp["date"].values,
            }

            # Games played this season before this game
            if "season" in grp.columns:
                cols["games_played_this_season"] = (
                    grp.groupby("season").cumcount().values
                )

            # Prior-season warm-start baseline (leak-free):
            # For a game in season Y, use full mean from season Y-1.
            if "season" in grp.columns:
                season_means = (
                    grp.groupby("season")[all_stats]
                    .mean()
                    .rename_axis("season")
                    .reset_index()
                )
                season_means["season"] = season_means["season"] + 1
                prior_map = season_means.set_index("season")
                for stat in all_stats:
                    if stat in prior_map.columns:
                        cols[f"prior_season_{stat}_mean"] = (
                            grp["season"].map(prior_map[stat]).values
                        )

            # Win/loss streak (positive = win streak, negative = loss streak)
            if "win" in grp.columns:
                streak = []
                current = 0
                for w in grp["win"]:
                    if pd.isna(w):
                        streak.append(np.nan)
                        continue
                    if w == 1:
                        current = max(current + 1, 1)
                    else:
                        current = min(current - 1, -1)
                    streak.append(current)
                # Shift to exclude the current game
                streak_s = pd.Series(streak, index=grp.index).shift(1)
                cols["streak"] = streak_s.values

            # Rolling ATS cover rate + cover margin + fav/dog split
            if "ats_result" in grp.columns and "is_home" in grp.columns:
                ats_vals = grp["ats_result"].astype(float)
                is_home_vals = grp["is_home"].astype(float)
                team_covered = np.where(
                    ats_vals.isna(),
                    np.nan,
                    np.where(is_home_vals == 1, ats_vals.values, 1.0 - ats_vals.values),
                )
                tc = pd.Series(team_covered, index=grp.index)
                cols["season_ats_cover_mean"] = tc.expanding().mean().shift(1).values
                for w in self.windows:
                    cols[f"roll{w}_ats_cover_mean"] = (
                        tc.rolling(w, min_periods=1).mean().shift(1).values
                    )

                if "spread" in grp.columns and "margin" in grp.columns:
                    spread_arr = grp["spread"].astype(float).values
                    margin_arr = grp["margin"].astype(float).values

                    # Cover margin: how many pts above/below the line this team covered by
                    # home: margin + spread (spread=-4.5 → need +5 margin → cm = margin+spread)
                    # away: margin - spread (away margin = -home_margin)
                    sign = np.where(is_home_vals.values == 1, 1.0, -1.0)
                    cm_arr = np.where(
                        np.isnan(spread_arr), np.nan, margin_arr + sign * spread_arr
                    )
                    cm = pd.Series(cm_arr, index=grp.index)
                    cols["season_cover_margin_mean"] = cm.expanding().mean().shift(1).values
                    cols["season_cover_margin_std"] = cm.expanding().std().shift(1).values
                    for w in self.windows:
                        cols[f"roll{w}_cover_margin_mean"] = (
                            cm.rolling(w, min_periods=1).mean().shift(1).values
                        )

                    # ATS cover rate when favored vs. when underdog
                    is_fav = np.where(
                        is_home_vals.values == 1,
                        (spread_arr < 0).astype(float),
                        (spread_arr > 0).astype(float),
                    )
                    is_fav = np.where(np.isnan(spread_arr), np.nan, is_fav)

                    fav_tc = pd.Series(
                        np.where(is_fav == 1, team_covered, np.nan), index=grp.index
                    )
                    dog_tc = pd.Series(
                        np.where(is_fav == 0, team_covered, np.nan), index=grp.index
                    )
                    cols["season_ats_as_fav"] = fav_tc.expanding().mean().shift(1).values
                    cols["season_ats_as_dog"] = dog_tc.expanding().mean().shift(1).values
                    for w in self.windows:
                        cols[f"roll{w}_ats_as_fav"] = (
                            fav_tc.rolling(w, min_periods=1).mean().shift(1).values
                        )
                        cols[f"roll{w}_ats_as_dog"] = (
                            dog_tc.rolling(w, min_periods=1).mean().shift(1).values
                        )

                # Home-specific vs away-specific ATS cover rates
                home_tc = pd.Series(
                    np.where(is_home_vals == 1, team_covered, np.nan), index=grp.index
                )
                away_tc = pd.Series(
                    np.where(is_home_vals == 0, team_covered, np.nan), index=grp.index
                )
                cols["season_ats_home_cover"] = home_tc.expanding().mean().shift(1).values
                cols["season_ats_away_cover"] = away_tc.expanding().mean().shift(1).values

            # ---- Over/Under result tracking ----
            if "over_result" in grp.columns:
                over_vals = grp["over_result"].astype(float)
                ov = pd.Series(over_vals.values, index=grp.index)
                cols["season_over_rate"] = ov.expanding().mean().shift(1).values
                for w in self.windows:
                    cols[f"roll{w}_over_rate"] = (
                        ov.rolling(w, min_periods=1).mean().shift(1).values
                    )

            for stat in all_stats:
                if stat not in grp.columns:
                    continue
                s = grp[stat].astype(float)

                # Career-to-date (expanding, shift 1 to exclude current game)
                cols[f"season_{stat}_mean"] = s.expanding().mean().shift(1).values
                cols[f"season_{stat}_std"] = s.expanding().std().shift(1).values

                # Cross-season rolling windows
                for w in self.windows:
                    cols[f"roll{w}_{stat}_mean"] = (
                        s.rolling(w, min_periods=1).mean().shift(1).values
                    )

                # Season-scoped rolling windows (reset each season)
                if "season" in grp.columns:
                    for w in self.windows:
                        cols[f"szn_roll{w}_{stat}_mean"] = (
                            grp.groupby("season")[stat]
                            .transform(
                                lambda x, _w=w: x.astype(float)
                                .rolling(_w, min_periods=1)
                                .mean()
                                .shift(1)
                            )
                            .values
                        )

            # ---- Rolling consistency (std) for key stats ----
            for stat in ["pts_for", "pts_against", "margin", "off_rtg", "def_rtg", "net_rtg"]:
                if stat not in grp.columns:
                    continue
                s = grp[stat].astype(float)
                for w in [5, 10]:
                    cols[f"roll{w}_{stat}_std"] = (
                        s.rolling(w, min_periods=3).std().shift(1).values
                    )

            # ---- Pythagorean expectation (16.5 exponent calibrated for basketball) ----
            if "pts_for" in grp.columns and "pts_against" in grp.columns:
                pts_f = grp["pts_for"].astype(float).clip(lower=0.1)
                pts_a = grp["pts_against"].astype(float).clip(lower=0.1)
                exp = 16.5
                denom = (pts_f ** exp + pts_a ** exp).clip(lower=1e-9)
                pyth = pts_f ** exp / denom
                cols["season_pyth_exp"] = pyth.expanding().mean().shift(1).values
                for w in self.windows:
                    cols[f"roll{w}_pyth_exp"] = (
                        pyth.rolling(w, min_periods=1).mean().shift(1).values
                    )

            # ---- Scoring momentum: recent form vs. season average ----
            for stat in ["pts_for", "pts_against", "margin", "off_rtg", "net_rtg"]:
                if stat not in grp.columns:
                    continue
                s = grp[stat].astype(float)
                roll3 = s.rolling(3, min_periods=1).mean().shift(1)
                szn_mean = s.expanding().mean().shift(1)
                cols[f"{stat}_momentum"] = (roll3 - szn_mean).values

            # ---- Home/Away scoring splits ----
            if "is_home" in grp.columns and "pts_for" in grp.columns:
                is_home_v = grp["is_home"].astype(float)
                pts_f_v = grp["pts_for"].astype(float)
                home_pts = pd.Series(np.where(is_home_v == 1, pts_f_v, np.nan), index=grp.index)
                away_pts = pd.Series(np.where(is_home_v == 0, pts_f_v, np.nan), index=grp.index)
                cols["season_home_pts_mean"] = home_pts.expanding().mean().shift(1).values
                cols["season_away_pts_mean"] = away_pts.expanding().mean().shift(1).values

            # ---- Schedule density: games played in last 7 days ----
            dates_arr = pd.to_datetime(grp["date"].values)
            g7 = []
            for i, dt in enumerate(dates_arr):
                cutoff = dt - pd.Timedelta(days=7)
                g7.append(int(sum(1 for d in dates_arr[:i] if d > cutoff)))
            cols["games_last_7"] = g7

            frames.append(pd.DataFrame(cols))

        return pd.concat(frames, ignore_index=True)

    def _join_to_games(
        self, games: pd.DataFrame, team_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge per-team features onto the games table for home and away sides."""
        base_cols = [
            c for c in [
                "game_id", "date", "season", "home_team", "away_team",
                "venue", "neutral_site", "spread", "ats_result",
                "home_win", "margin", "total_points", "home_score", "away_score",
            ]
            if c in games.columns
        ]
        df = games[base_cols].copy()

        # Cast neutral_site to int (XGBoost requires numeric, not object/bool)
        if "neutral_site" in df.columns:
            df["neutral_site"] = df["neutral_site"].astype(float)

        # Stat columns are everything except the join key columns
        stat_cols = [
            c for c in team_features.columns
            if c not in {"game_id", "team", "date"}
        ]

        tf_no_date = team_features.drop(columns=["date"])

        # Home: match where team_features.team == games.home_team
        home_tf = tf_no_date.rename(columns={c: f"home_{c}" for c in stat_cols})
        df = df.merge(
            home_tf,
            left_on=["game_id", "home_team"],
            right_on=["game_id", "team"],
            how="left",
        ).drop(columns=["team"])

        # Away: match where team_features.team == games.away_team
        away_tf = tf_no_date.rename(columns={c: f"away_{c}" for c in stat_cols})
        df = df.merge(
            away_tf,
            left_on=["game_id", "away_team"],
            right_on=["game_id", "team"],
            how="left",
        ).drop(columns=["team"])

        return df

    @staticmethod
    def _add_rest_days(
        feature_df: pd.DataFrame, log: pd.DataFrame
    ) -> pd.DataFrame:
        """Add days_rest, back-to-back flags, rest advantage, and schedule burden."""
        sorted_log = log.sort_values(["team", "date"]).copy()
        sorted_log["prev_date"] = sorted_log.groupby("team")["date"].shift(1)
        sorted_log["days_rest"] = (
            sorted_log["date"] - sorted_log["prev_date"]
        ).dt.days

        home_rest = sorted_log[["game_id", "team", "days_rest"]].rename(
            columns={"team": "home_team", "days_rest": "days_rest_home"}
        )
        away_rest = sorted_log[["game_id", "team", "days_rest"]].rename(
            columns={"team": "away_team", "days_rest": "days_rest_away"}
        )

        df = feature_df.merge(home_rest, on=["game_id", "home_team"], how="left")
        df = df.merge(away_rest, on=["game_id", "away_team"], how="left")

        df["home_b2b"] = (df["days_rest_home"] == 1).astype(float)
        df["away_b2b"] = (df["days_rest_away"] == 1).astype(float)
        # Positive = away is more fatigued (good for home); negative = home more fatigued
        df["b2b_advantage"] = df["away_b2b"] - df["home_b2b"]
        df["rest_advantage"] = (
            df["days_rest_home"].clip(upper=7) - df["days_rest_away"].clip(upper=7)
        )
        # Both teams on short rest (tired game → unpredictable, may skew under)
        df["both_b2b"] = ((df["home_b2b"] == 1) & (df["away_b2b"] == 1)).astype(float)
        return df

    @staticmethod
    def _add_differential_features(feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Home-minus-away differentials for key rolling stats.
        Encodes relative strength in a single number that is directly predictive.
        """
        df = feature_df
        new_cols: dict[str, pd.Series] = {}

        home_cols = {c[5:]: c for c in df.columns if c.startswith("home_roll")}
        away_cols = {c[5:]: c for c in df.columns if c.startswith("away_roll")}
        shared = home_cols.keys() & away_cols.keys()
        for key in shared:
            new_cols[f"diff_{key}"] = df[home_cols[key]] - df[away_cols[key]]

        for stat in ["pts_for", "pts_against", "margin", "win", "off_rtg", "def_rtg", "net_rtg"]:
            hcol, acol = f"home_season_{stat}_mean", f"away_season_{stat}_mean"
            if hcol in df.columns and acol in df.columns:
                new_cols[f"diff_season_{stat}_mean"] = df[hcol] - df[acol]

        derived = [
            "season_pyth_exp", "season_cover_margin_mean", "season_ats_cover_mean",
            "season_ats_as_fav", "season_ats_as_dog",
            "season_ats_home_cover", "season_ats_away_cover",
            "roll3_pyth_exp", "roll5_pyth_exp", "roll10_pyth_exp",
            "roll3_cover_margin_mean", "roll5_cover_margin_mean",
            "pts_for_momentum", "margin_momentum", "off_rtg_momentum", "net_rtg_momentum",
            "season_over_rate", "roll3_over_rate", "roll5_over_rate",
        ]
        for stat in derived:
            hcol, acol = f"home_{stat}", f"away_{stat}"
            if hcol in df.columns and acol in df.columns:
                new_cols[f"diff_{stat}"] = df[hcol] - df[acol]

        if "home_games_last_7" in df.columns and "away_games_last_7" in df.columns:
            new_cols["diff_games_last_7"] = (
                df["home_games_last_7"].fillna(0) - df["away_games_last_7"].fillna(0)
            )
        if "home_streak" in df.columns and "away_streak" in df.columns:
            new_cols["diff_streak"] = df["home_streak"].fillna(0) - df["away_streak"].fillna(0)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    @staticmethod
    def _add_opponent_adjusted(feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Opponent-adjusted efficiency features.

        adj_off_edge = home_recent_OffRtg − away_recent_DefRtg
            Positive → home offence exceeds away defence → home has scoring edge
        adj_def_edge = away_recent_OffRtg − home_recent_DefRtg
            Positive → away offence exceeds home defence → away has scoring edge
        net_edge = adj_off_edge − adj_def_edge → overall game edge for home
        """
        df = feature_df
        new_cols: dict[str, pd.Series] = {}

        for w in [5, 10]:
            home_off = f"home_roll{w}_off_rtg_mean"
            away_def = f"away_roll{w}_def_rtg_mean"
            away_off = f"away_roll{w}_off_rtg_mean"
            home_def = f"home_roll{w}_def_rtg_mean"
            if home_off in df.columns and away_def in df.columns:
                new_cols[f"adj_off_edge_{w}"] = df[home_off] - df[away_def]
            if away_off in df.columns and home_def in df.columns:
                new_cols[f"adj_def_edge_{w}"] = df[away_off] - df[home_def]
            off_k, def_k = f"adj_off_edge_{w}", f"adj_def_edge_{w}"
            if off_k in new_cols and def_k in new_cols:
                new_cols[f"net_edge_{w}"] = new_cols[off_k] - new_cols[def_k]

        # eFG% shooting matchup
        for w in [5]:
            h_efg = f"home_roll{w}_efg_pct_mean"
            a_efg = f"away_roll{w}_efg_pct_mean"
            if h_efg in df.columns and a_efg in df.columns:
                new_cols[f"efg_edge_{w}"] = df[h_efg] - df[a_efg]

        # Turnover pressure: home's tov% vs away's steal tendency
        for w in [5]:
            h_tov = f"home_roll{w}_tov_pct_mean"
            a_stl = f"away_roll{w}_steals_mean"
            if h_tov in df.columns and a_stl in df.columns:
                new_cols[f"tov_pressure_{w}"] = df[h_tov] + df[a_stl] * 0.1

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    @staticmethod
    def _add_totals_context(
        feature_df: pd.DataFrame, games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Features for predicting Over/Under results.

        Combines both teams' offensive/defensive rolling averages to estimate
        likely combined scoring, then compares to the total line.
        """
        df = feature_df
        new_cols: dict[str, pd.Series] = {}

        for w in [3, 5, 10]:
            hp, ap = f"home_roll{w}_pts_for_mean", f"away_roll{w}_pts_for_mean"
            hpa, apa = f"home_roll{w}_pts_against_mean", f"away_roll{w}_pts_against_mean"
            if hp in df.columns and ap in df.columns:
                new_cols[f"combined_pts_for_{w}"] = df[hp] + df[ap]
            if hpa in df.columns and apa in df.columns:
                new_cols[f"combined_pts_against_{w}"] = df[hpa] + df[apa]
            off_k, def_k = f"combined_pts_for_{w}", f"combined_pts_against_{w}"
            if off_k in new_cols and def_k in new_cols:
                new_cols[f"expected_total_{w}"] = (new_cols[off_k] + new_cols[def_k]) / 2

        for w in [5, 10]:
            hp, ap = f"home_roll{w}_pace_mean", f"away_roll{w}_pace_mean"
            if hp in df.columns and ap in df.columns:
                new_cols[f"combined_pace_{w}"] = df[hp] + df[ap]
            ho, ao = f"home_roll{w}_off_rtg_mean", f"away_roll{w}_off_rtg_mean"
            if ho in df.columns and ao in df.columns:
                new_cols[f"combined_off_rtg_{w}"] = df[ho] + df[ao]

        # Total line vs expected total (positive = line is set higher than teams' trend)
        if "total" in games.columns:
            total_series = games.set_index("game_id")["total"]
            new_cols["total_line"] = df["game_id"].map(total_series)
            for w in [5]:
                exp_k = f"expected_total_{w}"
                if exp_k in new_cols:
                    new_cols[f"total_line_vs_expected_{w}"] = new_cols["total_line"] - new_cols[exp_k]

        # Combined over rate (both teams' historical over tendency)
        for stat in ["season_over_rate", "roll5_over_rate"]:
            hc, ac = f"home_{stat}", f"away_{stat}"
            if hc in df.columns and ac in df.columns:
                new_cols[f"combined_{stat}"] = (df[hc].fillna(0.5) + df[ac].fillna(0.5)) / 2

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    @staticmethod
    def _add_travel_distance(feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add away-team travel distance features.

        Uses the haversine formula to compute the great-circle distance (miles)
        between the away team's home arena and the home team's arena (game venue).
        Features:
          away_travel_miles   : great-circle miles travelled by the away team.
          long_haul           : 1 if away_travel_miles > 1500 (cross-timezone trip).
          cross_country       : 1 if away_travel_miles > 2500 (coast-to-coast).
          travel_b2b_burden   : away_travel_miles * away_b2b (combined fatigue signal).
        """
        # lat/lon of each WNBA franchise's primary home arena (2018-present)
        _ARENAS: dict[str, tuple[float, float]] = {
            "Atlanta Dream":            (33.757, -84.396),   # State Farm Arena
            "Chicago Sky":              (41.880, -87.674),   # Wintrust Arena
            "Connecticut Sun":          (41.481, -72.090),   # Mohegan Sun Arena
            "Dallas Wings":             (32.748, -97.094),   # College Park Center
            "Golden State Valkyries":   (37.768, -122.388),  # Chase Center
            "Indiana Fever":            (39.764, -86.156),   # Gainbridge Fieldhouse
            "Los Angeles Sparks":       (34.043, -118.267),  # Crypto.com Arena
            "Las Vegas Aces":           (36.131, -115.166),  # Michelob Ultra Arena
            "Minnesota Lynx":           (44.979, -93.276),   # Target Center
            "New York Liberty":         (40.683, -73.975),   # Barclays Center
            "Phoenix Mercury":          (33.446, -112.071),  # Footprint Center
            "Seattle Storm":            (47.622, -122.354),  # Climate Pledge Arena
            "Washington Mystics":       (38.879, -76.998),   # Entertainment & Sports Arena
            # Historical teams (2018-2020)
            "San Antonio Stars":        (29.427, -98.438),   # AT&T Center
        }

        def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Return great-circle distance in miles between two lat/lon points."""
            r = 3958.8  # Earth radius in miles
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlam = np.radians(lon2 - lon1)
            a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
            return 2 * r * np.arcsin(np.sqrt(a))

        miles: list[float] = []
        for _, row in feature_df.iterrows():
            home_loc = _ARENAS.get(row.get("home_team", ""))
            away_loc = _ARENAS.get(row.get("away_team", ""))
            if home_loc and away_loc:
                miles.append(_haversine(*away_loc, *home_loc))
            else:
                miles.append(np.nan)

        new_cols: dict[str, pd.Series] = {}
        miles_s = pd.Series(miles, index=feature_df.index)
        new_cols["away_travel_miles"] = miles_s
        new_cols["long_haul"] = (miles_s > 1500).astype(float)
        new_cols["cross_country"] = (miles_s > 2500).astype(float)

        if "away_b2b" in feature_df.columns:
            new_cols["travel_b2b_burden"] = (
                miles_s.fillna(0) * feature_df["away_b2b"].fillna(0)
            )

        return pd.concat([feature_df, pd.DataFrame(new_cols, index=feature_df.index)], axis=1)

    @staticmethod
    def _add_h2h(feature_df: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
        """Add season head-to-head win-rate and ATS cover rate."""
        matchup_log = log.merge(
            log[["game_id", "team", "win"]].rename(
                columns={"team": "opponent_check", "win": "opp_win"}
            ),
            on="game_id",
        )
        matchup_log = matchup_log[
            matchup_log["team"] != matchup_log["opponent_check"]
        ]

        h2h_records: list[dict] = []
        for (team, opp, season), grp in matchup_log.groupby(
            ["team", "opponent_check", "season"]
        ):
            grp = grp.sort_values("date")
            cum_wins = grp["win"].expanding().sum().shift(1).fillna(0)
            cum_games = pd.Series(range(len(grp)), index=grp.index)

            if "ats_result" in grp.columns and "is_home" in grp.columns:
                ats_v = grp["ats_result"].astype(float)
                ih_v = grp["is_home"].astype(float)
                tc_h2h = np.where(
                    ats_v.isna(), np.nan,
                    np.where(ih_v == 1, ats_v.values, 1.0 - ats_v.values)
                )
                cum_ats = pd.Series(tc_h2h, index=grp.index).expanding().mean().shift(1)
            else:
                cum_ats = pd.Series(np.nan, index=grp.index)

            for i, (idx, row) in enumerate(grp.iterrows()):
                h2h_records.append({
                    "game_id": row["game_id"],
                    "team": team,
                    "opponent": opp,
                    "h2h_wins": cum_wins.iloc[i],
                    "h2h_games": cum_games.iloc[i],
                    "h2h_ats_cover": cum_ats.iloc[i],
                })

        if not h2h_records:
            return feature_df

        h2h = pd.DataFrame(h2h_records)
        h2h["h2h_win_rate"] = np.where(
            h2h["h2h_games"] > 0, h2h["h2h_wins"] / h2h["h2h_games"], 0.5,
        )

        home_h2h = h2h.rename(columns={
            "team": "home_team", "opponent": "away_team",
            "h2h_win_rate": "home_h2h_win_rate",
            "h2h_games": "h2h_games_season",
            "h2h_ats_cover": "home_h2h_ats_cover",
        })[["game_id", "home_team", "away_team", "home_h2h_win_rate",
            "h2h_games_season", "home_h2h_ats_cover"]]

        df = feature_df.merge(home_h2h, on=["game_id", "home_team", "away_team"], how="left")
        df["home_h2h_win_rate"] = df["home_h2h_win_rate"].fillna(0.5)
        df["h2h_games_season"] = df["h2h_games_season"].fillna(0)
        df["home_h2h_ats_cover"] = df["home_h2h_ats_cover"].fillna(0.5)
        return df

    @staticmethod
    def _add_line_movement(
        feature_df: pd.DataFrame, games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attach line movement columns and compute Reverse Line Movement (RLM).

        RLM = public bets on one side but the line moves the other way.
        This is the strongest known signal of sharp (syndicate) money.

        rlm_home = 1 when public backs home but line moved away (or vice versa).
        rlm_signal = continuous: (public_pct - 0.5) * (-move_direction)
            Positive values → sharp money opposing the public, on the away side.
        """
        lm_cols = [
            "line_move", "abs_line_move", "steam_move", "move_direction",
            "n_books_open", "n_books_close", "pct_books_agree",
            "spread_open", "spread_close", "line_range_open", "line_range_close",
        ]
        available = [c for c in lm_cols if c in games.columns]
        if not available:
            return feature_df

        movement_slice = games[["game_id"] + available].copy()
        if "steam_move" in movement_slice.columns:
            movement_slice["steam_move"] = (
                movement_slice["steam_move"].fillna(False).astype(float)
            )

        df = feature_df.merge(movement_slice, on="game_id", how="left")

        # Reverse Line Movement
        pub = df.get("home_public_pct")
        mv = df.get("move_direction")
        if pub is not None and mv is not None:
            df["rlm_home"] = np.where(
                pub.isna() | mv.isna(), np.nan,
                np.where(
                    ((pub > 55) & (mv > 0)) | ((pub < 45) & (mv < 0)),
                    1.0, 0.0,
                ),
            )
            df["rlm_signal"] = np.where(
                pub.isna() | mv.isna(), np.nan,
                (pub / 100.0 - 0.5) * (-mv),
            )

        # Book disagreement at close (uncertainty)
        if "line_range_close" in df.columns:
            df["sharp_disagreement"] = (df["line_range_close"] > 1.5).astype(float)

        return df
