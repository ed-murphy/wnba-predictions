"""
tests/test_processor.py

Unit tests for DataProcessor — no network calls, no file I/O.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.processor import DataProcessor


@pytest.fixture
def sample_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "date": ["2024-06-01T20:00:00Z", "2024-06-02T18:00:00Z", "2024-06-03T20:00:00Z"],
            "season": [2024, 2024, 2024],
            "status": ["final", "final", "in_progress"],
            "home_team": ["Seattle Storm", "Las Vegas Aces", "New York Liberty"],
            "home_team_id": ["16", "3", "5"],
            "home_team_abbr": ["SEA", "LV", "NY"],
            "home_score": [85, 92, 0],
            "away_team": ["Las Vegas Aces", "Seattle Storm", "Chicago Sky"],
            "away_team_id": ["3", "16", "8"],
            "away_team_abbr": ["LV", "SEA", "CHI"],
            "away_score": [78, 88, 0],
            "venue": ["Climate Pledge Arena", "Michelob ULTRA Arena", "Barclays Center"],
            "neutral_site": [False, False, False],
        }
    )


@pytest.fixture
def sample_odds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "home_team": ["Seattle Storm", "Seattle Storm", "Las Vegas Aces"],
            "away_team": ["Las Vegas Aces", "Las Vegas Aces", "Seattle Storm"],
            "bookmaker": ["fanduel", "draftkings", "fanduel"],
            "home_spread": [-4.5, -5.0, -3.5],
            "home_spread_price": [-110, -110, -110],
            "away_spread": [4.5, 5.0, 3.5],
            "away_spread_price": [-110, -110, -110],
        }
    )


def test_build_games_table_filters_non_final(tmp_path, sample_schedule):
    processor = DataProcessor(processed_dir=str(tmp_path))
    games = processor.build_games_table(sample_schedule)
    assert len(games) == 2
    assert all(games["status"] == "final")


def test_build_games_table_derived_columns(tmp_path, sample_schedule):
    processor = DataProcessor(processed_dir=str(tmp_path))
    games = processor.build_games_table(sample_schedule)
    assert "margin" in games.columns
    assert "total_points" in games.columns
    assert "home_win" in games.columns
    # game g1: home 85, away 78 → margin = 7, home_win = 1
    g1 = games[games["game_id"] == "g1"].iloc[0]
    assert g1["margin"] == 7
    assert g1["total_points"] == 163
    assert g1["home_win"] == 1


def test_build_games_table_ats_result(tmp_path, sample_schedule, sample_odds):
    processor = DataProcessor(processed_dir=str(tmp_path))
    games = processor.build_games_table(sample_schedule, odds=sample_odds)
    assert "spread" in games.columns
    assert "ats_result" in games.columns
    # g1: consensus spread = median(-4.5, -5.0) = -4.75, margin = 7
    # 7 + (-4.75) = 2.25 > 0 → home covered → ats_result = 1
    g1 = games[games["game_id"] == "g1"].iloc[0]
    assert g1["ats_result"] == 1.0


def test_build_team_game_log_shape(tmp_path, sample_schedule):
    processor = DataProcessor(processed_dir=str(tmp_path))
    games = processor.build_games_table(sample_schedule)
    log = processor.build_team_game_log(games)
    # 2 final games × 2 teams = 4 rows
    assert len(log) == 4


def test_build_team_game_log_columns(tmp_path, sample_schedule):
    processor = DataProcessor(processed_dir=str(tmp_path))
    games = processor.build_games_table(sample_schedule)
    log = processor.build_team_game_log(games)
    required = {"game_id", "team", "pts_for", "pts_against", "is_home", "win", "margin"}
    assert required.issubset(set(log.columns))
