"""
tests/test_models.py

Unit tests for ModelTrainer and Evaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import Evaluator
from src.models.train import ModelTrainer


def _make_feature_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic feature DataFrame with a learnable signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "game_id": [str(i) for i in range(n)],
            "date": pd.date_range("2022-06-01", periods=n, freq="D"),
            "season": [2022] * (n // 2) + [2023] * (n - n // 2),
            "home_team": ["Team A"] * n,
            "away_team": ["Team B"] * n,
            # Features with signal
            "home_roll5_pts_for_mean": rng.normal(85, 5, n),
            "away_roll5_pts_for_mean": rng.normal(80, 5, n),
            "home_roll5_pts_against_mean": rng.normal(78, 4, n),
            "away_roll5_pts_against_mean": rng.normal(82, 4, n),
            "days_rest_home": rng.integers(1, 5, n).astype(float),
            "days_rest_away": rng.integers(1, 5, n).astype(float),
        }
    )
    # Synthetic target: home covers if home offence > away offence
    signal = (
        df["home_roll5_pts_for_mean"] - df["away_roll5_pts_for_mean"]
        - df["home_roll5_pts_against_mean"] + df["away_roll5_pts_against_mean"]
    )
    noise = rng.normal(0, 3, n)
    df["ats_result"] = (signal + noise > 0).astype(float)
    return df


def test_trainer_logistic(tmp_path):
    """ModelTrainer should train and return metrics dict."""
    df = _make_feature_df(n=100)
    trainer = ModelTrainer(
        artifacts_dir=str(tmp_path / "artifacts"),
        model_type="logistic",
    )
    metrics = trainer.train(df, target="ats_result")

    assert "test_accuracy" in metrics
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert "flat_bet_roi" in metrics


def test_trainer_saves_artifacts(tmp_path):
    """Trained model artifacts should be persisted to disk."""
    df = _make_feature_df(n=100)
    trainer = ModelTrainer(
        artifacts_dir=str(tmp_path / "artifacts"),
        model_type="logistic",
    )
    trainer.train(df, target="ats_result")

    assert (tmp_path / "artifacts" / "model.pkl").exists()
    assert (tmp_path / "artifacts" / "feature_names.txt").exists()


def test_trainer_load_roundtrip(tmp_path):
    """Loaded model should produce identical predictions as trained model."""
    df = _make_feature_df(n=100)
    trainer = ModelTrainer(
        artifacts_dir=str(tmp_path / "artifacts"),
        model_type="logistic",
    )
    trainer.train(df, target="ats_result")

    loaded = ModelTrainer.load(artifacts_dir=str(tmp_path / "artifacts"))
    feat_cols = loaded.feature_names
    X = df[feat_cols].fillna(0)

    orig_preds = trainer.model.predict(X)
    loaded_preds = loaded.model.predict(X)
    np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_evaluator_flat_bet_roi():
    """ROI should be 0 if all picks are wrong, positive if all are right."""
    evaluator = Evaluator()

    # All correct → ROI = 100/110 ≈ 0.909
    preds = np.ones(10, dtype=int)
    probs = np.full(10, 0.8)
    actuals = np.ones(10, dtype=int)
    report = evaluator.full_report(preds, probs, actuals)
    assert report["flat_bet_roi"] > 0

    # All wrong → ROI = -1.0
    actuals_wrong = np.zeros(10, dtype=int)
    report_bad = evaluator.full_report(preds, probs, actuals_wrong)
    assert report_bad["flat_bet_roi"] < 0


def test_evaluator_calibration_bins():
    """Calibration table should have correct structure."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 200)
    actuals = (rng.uniform(0, 1, 200) < probs).astype(int)
    preds = (probs > 0.5).astype(int)

    evaluator = Evaluator()
    report = evaluator.full_report(preds, probs, actuals)

    assert "calibration" in report
    for row in report["calibration"]:
        assert "prob_bucket" in row
        assert "n" in row
        assert 0.0 <= row["actual_cover_rate"] <= 1.0


def test_trainer_no_valid_target_raises(tmp_path):
    """Training with no valid target rows should raise ValueError."""
    df = _make_feature_df(n=50)
    df["ats_result"] = float("nan")  # No valid labels

    trainer = ModelTrainer(
        artifacts_dir=str(tmp_path / "artifacts"),
        model_type="logistic",
    )
    with pytest.raises(ValueError, match="No rows with valid"):
        trainer.train(df, target="ats_result")
