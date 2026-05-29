"""
src/data/line_movement.py

Processes betting line movement — the change in spread from opening to closing.

Line movement is one of the strongest signals in sports betting because sharp
(professional) money tends to move lines, while public money tends not to.

Key concepts
------------
- Opening line  : spread first posted by bookmakers (days before tipoff)
- Closing line  : spread at game time (or last available snapshot)
- Line move     : closing_spread - opening_spread
                  Positive → line moved in favour of away team (home became
                  less favoured / away became cheaper to bet).
                  Negative → line moved in favour of home team.
- Steam move    : a significant, fast move ≥ 2 points — typically a sign of
                  sharp/syndicate action.
- Reverse line movement (RLM): line moves opposite to the direction of public
                  betting percentage — strong signal of sharp money.

Workflow
--------
1. Capture an opening snapshot shortly after lines are posted (e.g. 72h before tipoff).
2. Capture a closing snapshot as close to game time as possible (e.g. 1h before).
3. Call LineMovementProcessor.compute(opening_df, closing_df) to get movement table.
4. Pass the result to DataProcessor.build_games_table() via the `line_movement`
   parameter to attach it to the games table.

Output columns (per game_id)
-----------------------------
spread_open         float  consensus opening spread (home team)
spread_close        float  consensus closing spread (home team)
line_move           float  spread_close - spread_open
abs_line_move       float  absolute value of line_move
steam_move          bool   |line_move| >= steam_threshold (default 2.0)
move_direction      int    +1 = moved toward away, -1 = toward home, 0 = no move
n_books_open        int    number of bookmakers with opening line
n_books_close       int    number of bookmakers with closing line
pct_books_agree     float  fraction of books that moved in the same direction
spread_open_min     float  sharpest (most extreme) opening line
spread_open_max     float
spread_close_min    float
spread_close_max    float
line_range_open     float  max - min opening (book disagreement at open)
line_range_close    float  max - min closing (book disagreement at close)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_STEAM_THRESHOLD = 2.0  # points


class LineMovementProcessor:
    """Computes and stores line movement features from paired odds snapshots."""

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        steam_threshold: float = _DEFAULT_STEAM_THRESHOLD,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.steam_threshold = steam_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        opening: pd.DataFrame,
        closing: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute line movement from two odds snapshots.

        Parameters
        ----------
        opening : DataFrame
            Raw odds snapshot taken near line open. Expected columns:
            game_id, home_team, away_team, bookmaker, home_spread.
        closing : DataFrame
            Raw odds snapshot taken near game time. Same schema as opening.

        Returns
        -------
        DataFrame with one row per game_id and all movement features.
        """
        if opening.empty or closing.empty:
            logger.warning("One or both snapshots are empty; returning empty movement table.")
            return pd.DataFrame()

        open_agg = self._aggregate_snapshot(opening, label="open")
        close_agg = self._aggregate_snapshot(closing, label="close")

        movement = open_agg.merge(close_agg, on="game_id", how="inner")

        movement["line_move"] = movement["spread_close"] - movement["spread_open"]
        movement["abs_line_move"] = movement["line_move"].abs()
        movement["steam_move"] = (movement["abs_line_move"] >= self.steam_threshold)

        movement["move_direction"] = np.sign(movement["line_move"]).astype(int)

        # Book-level agreement: what fraction of books moved in the same direction?
        movement["pct_books_agree"] = self._book_agreement(opening, closing)

        out_path = self.processed_dir / "line_movement.parquet"
        movement.to_parquet(out_path, index=False)
        logger.info(
            "Line movement table saved to %s (%d games, %d steam moves)",
            out_path,
            len(movement),
            int(movement["steam_move"].sum()),
        )
        return movement

    def load(self) -> pd.DataFrame:
        """Load a previously computed line movement table from disk."""
        path = self.processed_dir / "line_movement.parquet"
        if not path.exists():
            logger.warning("No line movement table found at %s", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_snapshot(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Reduce a raw multi-bookmaker snapshot to one row per game_id.
        Consensus = median spread across bookmakers.
        """
        agg = (
            df.groupby("game_id")["home_spread"]
            .agg(
                **{
                    f"spread_{label}": "median",
                    f"spread_{label}_min": "min",
                    f"spread_{label}_max": "max",
                    f"n_books_{label}": "count",
                }
            )
            .reset_index()
        )
        agg[f"line_range_{label}"] = (
            agg[f"spread_{label}_max"] - agg[f"spread_{label}_min"]
        )
        return agg

    @staticmethod
    def _book_agreement(opening: pd.DataFrame, closing: pd.DataFrame) -> pd.Series:
        """
        For each game, compute the fraction of bookmakers whose line moved in
        the same direction as the consensus movement.

        Returns a Series indexed by game_id.
        """
        # Per-book movement
        merged = opening[["game_id", "bookmaker", "home_spread"]].merge(
            closing[["game_id", "bookmaker", "home_spread"]],
            on=["game_id", "bookmaker"],
            suffixes=("_open", "_close"),
        )
        if merged.empty:
            return pd.Series(dtype=float, name="pct_books_agree")

        merged["book_move"] = merged["home_spread_close"] - merged["home_spread_open"]

        # Consensus direction per game
        consensus = merged.groupby("game_id")["book_move"].median().rename("consensus_move")
        merged = merged.join(consensus, on="game_id")

        merged["same_dir"] = (
            np.sign(merged["book_move"]) == np.sign(merged["consensus_move"])
        ) | (merged["book_move"] == 0)

        agreement = merged.groupby("game_id")["same_dir"].mean()
        return agreement
