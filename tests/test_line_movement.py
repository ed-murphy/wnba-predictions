"""
tests/test_line_movement.py

Unit tests for LineMovementProcessor and the DataProcessor line movement
integration — no network calls, no file I/O beyond tmp_path.
"""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from src.data.line_movement import LineMovementProcessor
from src.data.processor import DataProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def opening_odds() -> pd.DataFrame:
    """Simulated opening-line snapshot (72h before games)."""
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g2", "g2"],
            "home_team": ["Seattle Storm"] * 3 + ["Las Vegas Aces"] * 2,
            "away_team": ["Las Vegas Aces"] * 3 + ["Seattle Storm"] * 2,
            "bookmaker": ["fanduel", "draftkings", "betmgm", "fanduel", "draftkings"],
            "home_spread": [-4.5, -4.5, -5.0, -3.0, -3.5],
        }
    )


@pytest.fixture
def closing_odds() -> pd.DataFrame:
    """Simulated closing-line snapshot (1h before games)."""
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g2", "g2"],
            "home_team": ["Seattle Storm"] * 3 + ["Las Vegas Aces"] * 2,
            "away_team": ["Las Vegas Aces"] * 3 + ["Seattle Storm"] * 2,
            "bookmaker": ["fanduel", "draftkings", "betmgm", "fanduel", "draftkings"],
            # g1: line moved from ~-4.5 to ~-6.5 → home became MORE favoured
            # g2: line moved from ~-3.0 to ~-1.5 → home became LESS favoured
            "home_spread": [-6.5, -6.5, -6.0, -1.5, -1.5],
        }
    )


@pytest.fixture
def sample_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "date": ["2024-06-01T20:00:00Z", "2024-06-02T20:00:00Z"],
            "season": [2024, 2024],
            "status": ["final", "final"],
            "home_team": ["Seattle Storm", "Las Vegas Aces"],
            "home_team_id": ["16", "3"],
            "home_team_abbr": ["SEA", "LV"],
            "home_score": [90, 80],
            "away_team": ["Las Vegas Aces", "Seattle Storm"],
            "away_team_id": ["3", "16"],
            "away_team_abbr": ["LV", "SEA"],
            "away_score": [80, 78],
            "venue": ["Arena A", "Arena B"],
            "neutral_site": [False, False],
        }
    )


# ---------------------------------------------------------------------------
# LineMovementProcessor tests
# ---------------------------------------------------------------------------

def test_compute_returns_one_row_per_game(tmp_path, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    result = lmp.compute(opening_odds, closing_odds)
    assert len(result) == 2
    assert set(result["game_id"]) == {"g1", "g2"}


def test_consensus_spread_is_median(tmp_path, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    result = lmp.compute(opening_odds, closing_odds)
    g1 = result[result["game_id"] == "g1"].iloc[0]
    # Opening for g1: [-4.5, -4.5, -5.0] → median = -4.5
    assert g1["spread_open"] == pytest.approx(-4.5)
    # Closing for g1: [-6.5, -6.5, -6.0] → median = -6.5
    assert g1["spread_close"] == pytest.approx(-6.5)


def test_line_move_direction(tmp_path, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    result = lmp.compute(opening_odds, closing_odds)

    g1 = result[result["game_id"] == "g1"].iloc[0]
    g2 = result[result["game_id"] == "g2"].iloc[0]

    # g1: -6.5 - (-4.5) = -2.0 → moved toward home (more negative)
    assert g1["line_move"] == pytest.approx(-2.0)
    assert g1["move_direction"] == -1

    # g2: -1.5 - (-3.25) = +1.75 → moved toward away (less negative)
    assert g2["line_move"] > 0
    assert g2["move_direction"] == 1


def test_steam_move_flag(tmp_path, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path), steam_threshold=2.0)
    result = lmp.compute(opening_odds, closing_odds)

    g1 = result[result["game_id"] == "g1"].iloc[0]
    g2 = result[result["game_id"] == "g2"].iloc[0]

    # g1 abs_move = 2.0 → exactly at threshold → steam
    assert g1["steam_move"] is True or g1["steam_move"] == 1
    # g2 abs_move < 2.0 → not a steam move
    assert g2["steam_move"] is False or g2["steam_move"] == 0


def test_n_books_columns(tmp_path, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    result = lmp.compute(opening_odds, closing_odds)
    g1 = result[result["game_id"] == "g1"].iloc[0]
    assert g1["n_books_open"] == 3
    assert g1["n_books_close"] == 3


def test_line_range(tmp_path, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    result = lmp.compute(opening_odds, closing_odds)
    g1 = result[result["game_id"] == "g1"].iloc[0]
    # Opening range: max(-4.5,-4.5,-5.0) - min(-4.5,-4.5,-5.0) = -4.5 - (-5.0) = 0.5
    assert g1["line_range_open"] == pytest.approx(0.5)


def test_empty_snapshot_returns_empty(tmp_path):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    result = lmp.compute(pd.DataFrame(), pd.DataFrame())
    assert result.empty


def test_persists_to_disk(tmp_path, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    lmp.compute(opening_odds, closing_odds)
    loaded = lmp.load()
    assert not loaded.empty
    assert "line_move" in loaded.columns


# ---------------------------------------------------------------------------
# DataProcessor integration tests
# ---------------------------------------------------------------------------

def test_processor_attaches_movement_columns(tmp_path, sample_schedule, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    movement = lmp.compute(opening_odds, closing_odds)

    processor = DataProcessor(processed_dir=str(tmp_path))
    games = processor.build_games_table(sample_schedule, line_movement=movement)

    assert "spread_close" in games.columns
    assert "line_move" in games.columns
    assert "steam_move" in games.columns
    assert "spread" in games.columns  # alias for spread_close


def test_processor_ats_result_uses_closing_spread(tmp_path, sample_schedule, opening_odds, closing_odds):
    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    movement = lmp.compute(opening_odds, closing_odds)

    processor = DataProcessor(processed_dir=str(tmp_path))
    games = processor.build_games_table(sample_schedule, line_movement=movement)

    # g1: home_score=90, away_score=80 → margin=10
    # spread_close for g1 = -6.5
    # 10 + (-6.5) = 3.5 > 0 → home covered → ats_result = 1
    g1 = games[games["game_id"] == "g1"].iloc[0]
    assert g1["ats_result"] == 1.0


def test_processor_falls_back_to_odds_without_movement(tmp_path, sample_schedule, closing_odds):
    """When line_movement is None, should still work with plain odds."""
    processor = DataProcessor(processed_dir=str(tmp_path))
    # closing_odds only has game_id, home_spread; pass as plain odds
    games = processor.build_games_table(sample_schedule, odds=closing_odds)
    assert "spread" in games.columns
    assert "line_move" not in games.columns


# ---------------------------------------------------------------------------
# FeatureEngineer integration test
# ---------------------------------------------------------------------------

def test_feature_engineer_includes_movement_cols(tmp_path, sample_schedule, opening_odds, closing_odds):
    from src.data.processor import DataProcessor
    from src.features.engineer import FeatureEngineer

    # Build a bigger schedule so feature engineer has enough rows
    rows = []
    for i in range(10):
        rows.append(
            {
                "game_id": f"pre{i}",
                "date": f"2024-05-{i + 1:02d}T20:00:00Z",
                "season": 2024,
                "status": "final",
                "home_team": "Seattle Storm",
                "home_team_id": "16",
                "home_team_abbr": "SEA",
                "home_score": 80 + i,
                "away_team": "Las Vegas Aces",
                "away_team_id": "3",
                "away_team_abbr": "LV",
                "away_score": 75,
                "venue": "Arena",
                "neutral_site": False,
            }
        )
    full_schedule = pd.concat(
        [pd.DataFrame(rows), sample_schedule], ignore_index=True
    )

    lmp = LineMovementProcessor(processed_dir=str(tmp_path))
    movement = lmp.compute(opening_odds, closing_odds)

    processor = DataProcessor(processed_dir=str(tmp_path))
    games = processor.build_games_table(full_schedule, line_movement=movement)
    log = processor.build_team_game_log(games)

    engineer = FeatureEngineer(processed_dir=str(tmp_path), windows=[3])
    features = engineer.build_features(log, games)

    movement_cols = [c for c in features.columns if c in ["line_move", "steam_move", "abs_line_move"]]
    assert len(movement_cols) > 0, "Expected line movement columns in feature matrix"
