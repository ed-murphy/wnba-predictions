"""
src/models/evaluate.py

Betting-focused evaluation utilities for ATS prediction models.

Key metrics
-----------
- ATS accuracy           : % of picks that covered
- Brier score            : probability calibration
- Flat-bet ROI           : return on investment assuming $110 to win $100
- Kelly-optimal ROI      : ROI using fractional Kelly sizing
- ATS record by season   : year-by-year breakdown
- Calibration table      : bucketed probability vs actual cover rate
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VIG = 110  # risk $110 to win $100 (standard -110 line)
_WIN_PAYOUT = 100


class Evaluator:
    """Computes ATS betting-model evaluation metrics."""

    def __init__(self, kelly_fraction: float = 0.25) -> None:
        """
        Parameters
        ----------
        kelly_fraction : float
            Fraction of full Kelly bet to use (default 0.25 = quarter-Kelly).
        """
        self.kelly_fraction = kelly_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def full_report(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        actuals: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Parameters
        ----------
        predictions : array of int
            Binary predicted labels (1 = home covers, 0 = away covers).
        probabilities : array of float
            Model-predicted probability that home covers.
        actuals : array of int
            Ground-truth ATS results.
        meta : DataFrame, optional
            Game metadata (must include 'season' and 'date' columns) for
            breakdowns.

        Returns
        -------
        dict with all metrics.
        """
        mask = ~np.isnan(actuals.astype(float))
        predictions = predictions[mask]
        probabilities = probabilities[mask]
        actuals = actuals[mask].astype(int)

        report: dict[str, Any] = {}
        report["n_games"] = int(len(actuals))
        report["ats_accuracy"] = float(np.mean(predictions == actuals))
        report["cover_rate"] = float(np.mean(actuals))  # base rate

        report.update(self._roi_metrics(predictions, probabilities, actuals))
        report["calibration"] = self._calibration_table(probabilities, actuals)

        if meta is not None:
            filt_meta = meta.iloc[mask] if hasattr(meta, "iloc") else meta
            if "season" in filt_meta.columns:
                report["by_season"] = self._by_season(
                    predictions, probabilities, actuals, filt_meta
                )

        return report

    def print_report(self, report: dict[str, Any]) -> None:
        """Pretty-print the evaluation report."""
        print("\n" + "=" * 55)
        print("  WNBA ATS Model Evaluation Report")
        print("=" * 55)
        print(f"  Games evaluated   : {report['n_games']}")
        print(f"  Base cover rate   : {report['cover_rate']:.1%}")
        print(f"  ATS accuracy      : {report['ats_accuracy']:.1%}")
        print(f"  Flat-bet ROI      : {report['flat_bet_roi']:+.1%}")
        print(f"  Kelly ROI         : {report['kelly_roi']:+.1%}")
        print(f"  Brier score       : {report['brier_score']:.4f}")
        print(f"  Log loss          : {report['log_loss']:.4f}")

        if "by_season" in report:
            print("\n  Season Breakdown:")
            print(f"  {'Season':>8}  {'N':>5}  {'Acc':>6}  {'ROI':>8}")
            print("  " + "-" * 35)
            for row in report["by_season"]:
                print(
                    f"  {row['season']:>8}  {row['n']:>5}  "
                    f"{row['ats_accuracy']:>6.1%}  {row['flat_bet_roi']:>+8.1%}"
                )

        if "calibration" in report:
            print("\n  Calibration Table:")
            print(f"  {'Prob Bucket':>12}  {'N':>5}  {'Actual Cover%':>14}")
            print("  " + "-" * 38)
            for row in report["calibration"]:
                print(
                    f"  {row['prob_bucket']:>12}  {row['n']:>5}  "
                    f"{row['actual_cover_rate']:>14.1%}"
                )

        print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _roi_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        actuals: np.ndarray,
    ) -> dict:
        """Compute flat-bet and Kelly-bet ROI metrics."""
        from sklearn.metrics import brier_score_loss, log_loss

        flat_roi = self._flat_bet_roi(predictions, actuals)
        kelly_roi = self._kelly_roi(probabilities, actuals)

        return {
            "flat_bet_roi": flat_roi,
            "kelly_roi": kelly_roi,
            "brier_score": float(brier_score_loss(actuals, probabilities)),
            "log_loss": float(log_loss(actuals, probabilities, labels=[0, 1])),
        }

    @staticmethod
    def _flat_bet_roi(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """ROI for flat $110 bets on every predicted home-cover."""
        bets = predictions == 1
        if not bets.any():
            return 0.0
        wins = bets & (actuals == 1)
        losses = bets & (actuals == 0)
        net = wins.sum() * _WIN_PAYOUT - losses.sum() * _VIG
        return float(net / (bets.sum() * _VIG))

    def _kelly_roi(
        self, probabilities: np.ndarray, actuals: np.ndarray
    ) -> float:
        """
        ROI using fractional Kelly criterion.
        Kelly fraction: f = (b*p - q) / b  where b = WIN_PAYOUT/VIG, p = prob, q = 1-p.
        """
        b = _WIN_PAYOUT / _VIG
        p = probabilities
        q = 1.0 - p
        raw_kelly = (b * p - q) / b
        kelly_bets = raw_kelly * self.kelly_fraction
        kelly_bets = np.clip(kelly_bets, 0.0, 1.0)  # no negative bets

        # Only bet when Kelly fraction > 0
        bet_mask = kelly_bets > 0
        if not bet_mask.any():
            return 0.0

        bankroll = 1.0
        for i in np.where(bet_mask)[0]:
            stake = bankroll * kelly_bets[i]
            if actuals[i] == 1:
                bankroll += stake * b
            else:
                bankroll -= stake

        return float(bankroll - 1.0)

    @staticmethod
    def _calibration_table(
        probabilities: np.ndarray, actuals: np.ndarray, n_bins: int = 10
    ) -> list[dict]:
        """Bucket predicted probabilities and compute actual cover rate per bucket."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probabilities >= lo) & (probabilities < hi)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "prob_bucket": f"{lo:.1f}–{hi:.1f}",
                    "n": int(mask.sum()),
                    "mean_prob": float(probabilities[mask].mean()),
                    "actual_cover_rate": float(actuals[mask].mean()),
                }
            )
        return rows

    @staticmethod
    def _by_season(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        actuals: np.ndarray,
        meta: pd.DataFrame,
    ) -> list[dict]:
        """ATS accuracy and ROI broken down by season."""
        rows = []
        for season, grp in meta.groupby("season"):
            idx = grp.index
            preds_s = predictions[idx] if isinstance(predictions, np.ndarray) else predictions.iloc[idx]
            probs_s = probabilities[idx] if isinstance(probabilities, np.ndarray) else probabilities.iloc[idx]
            acts_s = actuals[idx] if isinstance(actuals, np.ndarray) else actuals.iloc[idx]

            bets = preds_s == 1
            wins = bets & (acts_s == 1)
            losses = bets & (acts_s == 0)
            net = wins.sum() * _WIN_PAYOUT - losses.sum() * _VIG
            roi = net / (bets.sum() * _VIG) if bets.sum() > 0 else 0.0

            rows.append(
                {
                    "season": season,
                    "n": len(acts_s),
                    "ats_accuracy": float((preds_s == acts_s).mean()),
                    "flat_bet_roi": float(roi),
                }
            )
        return rows
