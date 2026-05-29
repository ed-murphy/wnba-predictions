"""
src/models/train.py

Trains a classifier to predict whether the home team covers the spread
(ats_result == 1) and an optional regressor to predict the final margin.

Supported model types: "xgboost", "lightgbm", "logistic"

The trainer uses a temporal train/test split (no shuffling) to simulate
real-world deployment where we only train on past data.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns that are targets or identifiers — never used as features
_NON_FEATURE_COLS = {
    "game_id", "date", "season", "home_team", "away_team",
    "venue", "ats_result", "over_result", "home_win", "margin", "total_points",
    "home_score", "away_score", "status",
}


class AveragingEnsemble:
    """
    Averages predict_proba output from a list of pre-fitted estimators.

    Each estimator must implement predict_proba(X) returning shape (n, 2).
    Predictions are the simple unweighted mean across all estimators.
    """

    def __init__(self, models: list) -> None:
        self.models = models

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _get_model(model_type: str, params: dict) -> Any:
    """Instantiate a model by type name."""
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                eval_metric="logloss",
                random_state=params.get("random_state", 42),
                **{
                    k: v for k, v in params.get("xgboost", {}).items()
                },
            )
        except ImportError:
            logger.warning("xgboost not installed, falling back to logistic.")
            model_type = "logistic"

    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                random_state=params.get("random_state", 42),
                verbose=-1,
                **{k: v for k, v in params.get("lightgbm", {}).items()},
            )
        except ImportError:
            logger.warning("lightgbm not installed, falling back to logistic.")
            model_type = "logistic"

    # Logistic regression (always available via scikit-learn)
    lr_params = params.get("logistic", {})
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=params.get("random_state", 42),
            **lr_params,
        )),
    ])


class ModelTrainer:
    """Trains ATS spread-covering classifiers on a feature matrix."""

    def __init__(
        self,
        artifacts_dir: str | Path = "models/artifacts",
        model_type: str = "xgboost",
        model_params: dict | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model_params.setdefault("random_state", random_state)
        self.test_size = test_size
        self.random_state = random_state
        self.model: Any = None
        self.feature_names: list[str] = []
        self.train_medians: pd.Series | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, features: pd.DataFrame, target: str = "ats_result") -> dict:
        """
        Train the ATS classifier.

        Parameters
        ----------
        features : DataFrame
            Output of FeatureEngineer.build_features().
        target : str
            Target column name (default: "ats_result").

        Returns
        -------
        dict with train/test metrics.
        """
        df = features.dropna(subset=[target]).copy()
        if df.empty:
            raise ValueError(
                f"No rows with valid '{target}' labels. "
                "Have you merged odds data? See DataProcessor.build_games_table()."
            )

        # Temporal split: first 80% = training, last 20% = held-out test.
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        test_start = int(n * (1 - self.test_size))

        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:]

        # For the Over/Under model, exclude ATS-specific features — they have
        # no causal relationship to game totals and cause severe overfitting.
        ou_excl = ["_ats_"] if target == "over_result" else []
        X_train, y_train = self._split_xy(train_df, target, exclude_substr=ou_excl)
        X_test, y_test = self._split_xy(test_df, target, exclude_substr=ou_excl)

        self.feature_names = list(X_train.columns)
        logger.info(
            "Training on %d samples, testing on %d samples, %d features",
            len(X_train), len(X_test), len(self.feature_names),
        )

        # Impute NaNs with column medians (fit on train only to prevent leakage)
        # then apply feature selection via mutual information.
        self.train_medians = X_train.median()
        X_train = X_train.fillna(self.train_medians).fillna(0)
        X_test = X_test.fillna(self.train_medians).fillna(0)

        # Feature selection: keep top-K features by mutual information with the
        # target, fit only on training data to prevent leakage.  With ~1000
        # training rows and 150+ engineered features the models overfit badly
        # without this step.  Use a lower k for O/U since the signal is weaker.
        k_features = min(50 if target == "over_result" else 65, len(self.feature_names))
        selector = SelectKBest(mutual_info_classif, k=k_features)
        selector.fit(X_train, y_train)
        selected_mask = selector.get_support()
        X_train = X_train.loc[:, selected_mask]
        X_test = X_test.loc[:, selected_mask]
        self.feature_names = list(X_train.columns)
        logger.info(
            "Feature selection: kept %d / %d features",
            len(self.feature_names), len(selected_mask),
        )

        # Calibrate with TimeSeriesSplit: 5 expanding-window folds each train a
        # base model and fit sigmoid calibration on the next window.  Averaging
        # 5 calibrated models avoids the collapse-to-all-zeros failure mode seen
        # with a single PredefinedSplit fold (where one small cal set causes the
        # sigmoid to learn a near-zero slope and map everything below 0.5).
        tscv = TimeSeriesSplit(n_splits=5)

        if self.model_type == "ensemble":
            # Train XGBoost + LightGBM independently, then average their
            # calibrated probabilities.  Diverse inductive biases give better
            # probability estimates than any single model.
            base_types = ["xgboost", "lightgbm"]
            calibrated_models = []
            for bt in base_types:
                bm = _get_model(bt, self.model_params)
                cal = CalibratedClassifierCV(bm, cv=tscv, method="sigmoid")
                cal.fit(X_train, y_train)
                calibrated_models.append(cal)
            self.model = AveragingEnsemble(calibrated_models)
        else:
            base_model = _get_model(self.model_type, self.model_params)
            self.model = CalibratedClassifierCV(base_model, cv=tscv, method="sigmoid")
            self.model.fit(X_train, y_train)

        metrics = self._compute_metrics(
            X_train, y_train, X_test, y_test, test_df
        )
        self._log_feature_importance()
        self._save_artifacts()
        return metrics

    def cross_validate(
        self, features: pd.DataFrame, target: str = "ats_result", n_folds: int = 5
    ) -> dict:
        """
        Temporal k-fold cross-validation (folds are chronological, not shuffled).
        Returns mean ± std for key metrics.
        """
        df = features.dropna(subset=[target]).sort_values("date").reset_index(drop=True)
        X, y = self._split_xy(df, target)
        self.feature_names = list(X.columns)

        fold_size = len(df) // n_folds
        accs, brls = [], []

        for fold in range(1, n_folds):
            train_end = fold * fold_size
            test_end = min(train_end + fold_size, len(df))

            X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
            X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

            if len(X_tr) < 20 or len(X_te) < 5:
                continue

            m = _get_model(self.model_type, self.model_params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            probs = m.predict_proba(X_te)[:, 1]

            accs.append(accuracy_score(y_te, preds))
            brls.append(brier_score_loss(y_te, probs))

        return {
            "cv_accuracy_mean": float(np.mean(accs)),
            "cv_accuracy_std": float(np.std(accs)),
            "cv_brier_mean": float(np.mean(brls)),
            "cv_brier_std": float(np.std(brls)),
            "n_folds": len(accs),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_xy(
        self, df: pd.DataFrame, target: str,
        exclude_substr: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and label series."""
        exclude_substr = exclude_substr or []
        feature_cols = [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])
            and not any(sub in c for sub in exclude_substr)
        ]
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target].astype(int)
        return X, y

    def _compute_metrics(
        self,
        X_train, y_train, X_test, y_test,
        test_df: pd.DataFrame,
    ) -> dict:
        """Compute train and test metrics including ATS ROI simulation."""
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_preds = self.model.predict(X_test)
        test_probs = self.model.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, test_preds)
        brier = brier_score_loss(y_test, test_probs)

        # Simulated flat-bet ROI at -110 vig
        roi = self._simulate_roi(test_preds, y_test.values)

        metrics = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "brier_score": round(brier, 4),
            "flat_bet_roi": round(roi, 4),
            "test_n": len(y_test),
            "model_type": self.model_type,
        }

        # ROI filtered by model confidence — bet only when edge is clearest
        for thresh in [0.55, 0.60, 0.65]:
            mask_h = test_probs >= thresh
            mask_a = (1.0 - test_probs) >= thresh
            total = int(mask_h.sum() + mask_a.sum())
            if total > 0:
                wins = float(y_test.values[mask_h].sum() + (1.0 - y_test.values[mask_a]).sum())
                thresh_roi = (wins * (100.0 / 110.0) - (total - wins)) / total
                key = f"roi_ge{int(thresh * 100)}pct"
                metrics[key] = round(thresh_roi, 4)
                metrics[f"bets_{key}"] = total

        # Closing Line Value (CLV) — market-disagreement betting strategy.
        # Instead of betting blindly on model > 0.5, only bet when the model
        # meaningfully disagrees with the closing line:
        #   Bet HOME  when model_prob > closing_implied + threshold  (model finds home value)
        #   Bet AWAY  when closing_implied > model_prob + threshold  (model fades home)
        if "spread" in test_df.columns:
            try:
                from scipy.stats import norm as _norm
                spreads = test_df["spread"].values.astype(float)
                closing_implied = _norm.cdf(-spreads / 12.0)
                clv = test_probs - closing_implied  # + = model more bullish on home than market
                valid = ~np.isnan(closing_implied)

                # Overall mean CLV across all games with valid spreads (diagnostic)
                if valid.sum() > 0:
                    metrics["mean_clv_all"] = round(float(np.nanmean(clv[valid])), 4)

                # CLV-aware ROI at several edge thresholds
                for edge in [0.03, 0.05, 0.07]:
                    bet_home = valid & (clv > edge)
                    bet_away = valid & (clv < -edge)
                    n_bets = int(bet_home.sum() + bet_away.sum())
                    if n_bets > 0:
                        wins = float(
                            y_test.values[bet_home].sum()
                            + (1 - y_test.values[bet_away]).sum()
                        )
                        clv_roi = (wins * (100.0 / 110.0) - (n_bets - wins)) / n_bets
                        tag = f"clv{int(edge * 100):02d}"
                        metrics[f"roi_{tag}"] = round(clv_roi, 4)
                        metrics[f"bets_{tag}"] = n_bets
                        metrics[f"mean_clv_{tag}"] = round(
                            float(np.nanmean(clv[bet_home | bet_away])), 4
                        )
            except Exception:
                pass  # CLV is supplementary; never block training

        logger.info("Training complete: %s", metrics)
        return metrics

    @staticmethod
    def _simulate_roi(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Flat-bet ROI at standard -110 vig.
        Risk $110 to win $100 on every predicted home-cover.
        Returns ROI as a fraction (e.g. 0.05 = 5%).
        """
        total_risked = len(predictions) * 110
        total_won = sum(
            100 for pred, actual in zip(predictions, actuals)
            if pred == 1 and actual == 1
        )
        total_lost = sum(
            110 for pred, actual in zip(predictions, actuals)
            if pred == 1 and actual == 0
        )
        net = total_won - total_lost
        return net / total_risked if total_risked > 0 else 0.0

    def _save_artifacts(self) -> None:
        """Persist model, feature names, and train medians to disk."""
        model_path = self.artifacts_dir / "model.pkl"
        features_path = self.artifacts_dir / "feature_names.txt"
        medians_path = self.artifacts_dir / "train_medians.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        with open(features_path, "w") as f:
            f.write("\n".join(self.feature_names))

        if self.train_medians is not None:
            with open(medians_path, "wb") as f:
                pickle.dump(self.train_medians, f)

        logger.info("Model saved to %s", model_path)

    def _log_feature_importance(self) -> None:
        """Log the top-20 most important features (tree models only)."""
        model = self.model

        # Collect all base estimators (handles AveragingEnsemble + CalibratedClassifierCV)
        def _extract_importances(m) -> np.ndarray | None:
            """Return feature_importances_ array or None if not applicable."""
            if isinstance(m, AveragingEnsemble):
                arrays = [a for sub in m.models for a in [_extract_importances(sub)] if a is not None]
                return np.mean(arrays, axis=0) if arrays else None
            if hasattr(m, "calibrated_classifiers_"):
                arrays = []
                for cc in m.calibrated_classifiers_:
                    inner = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
                    if inner is not None:
                        if hasattr(inner, "named_steps"):
                            inner = inner.named_steps.get("clf", inner)
                        if hasattr(inner, "feature_importances_"):
                            arrays.append(inner.feature_importances_)
                return np.mean(arrays, axis=0) if arrays else None
            if hasattr(m, "named_steps"):
                m = m.named_steps.get("clf", m)
            if hasattr(m, "feature_importances_"):
                return m.feature_importances_
            return None

        importances = _extract_importances(model)
        if importances is None:
            return
        pairs = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        top = pairs[:20]
        logger.info("Top-20 feature importances:")
        for name, score in top:
            logger.info("  %-45s %.4f", name, score)

        # Also save full importance table for inspection
        imp_df = pd.DataFrame(pairs, columns=["feature", "importance"])
        imp_path = self.artifacts_dir / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        logger.info("Full feature importance saved to %s", imp_path)

    @classmethod
    def load(cls, artifacts_dir: str | Path = "models/artifacts") -> "ModelTrainer":
        """Load a previously trained model from disk."""
        artifacts_dir = Path(artifacts_dir)
        model_path = artifacts_dir / "model.pkl"
        features_path = artifacts_dir / "feature_names.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found at {model_path}")

        trainer = cls(artifacts_dir=artifacts_dir)
        with open(model_path, "rb") as f:
            trainer.model = pickle.load(f)

        if features_path.exists():
            trainer.feature_names = features_path.read_text().splitlines()

        medians_path = artifacts_dir / "train_medians.pkl"
        if medians_path.exists():
            with open(medians_path, "rb") as f:
                trainer.train_medians = pickle.load(f)

        logger.info("Model loaded from %s", model_path)
        return trainer
