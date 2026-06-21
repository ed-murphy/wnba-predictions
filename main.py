"""
main.py — CLI entry point for WNBA ATS predictions.

Usage examples
--------------
# Full pipeline: fetch data, engineer features, train model, evaluate
python main.py train --seasons 2022 2023 2024

# Predict today's games (requires ODDS_API_KEY in .env)
python main.py predict

# Show evaluation report on held-out test set
python main.py evaluate --seasons 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))

from src.data import ESPNScraper, LineMovementProcessor, OddsAPIClient, DataProcessor, ActionNetworkScraper
from src.features import FeatureEngineer
from src.models import ModelTrainer, Evaluator
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def cmd_train(args: argparse.Namespace, cfg: dict) -> None:
    """Fetch data, engineer features, train and save an ATS model."""
    seasons = args.seasons or cfg["data"]["seasons"]
    model_type = args.model or cfg["model"]["type"]

    setup_logging(cfg["logging"]["level"])
    logger.info("Starting training pipeline for seasons: %s", seasons)

    # 1. Collect data
    scraper = ESPNScraper(raw_dir=cfg["data"]["raw_dir"])
    schedule = scraper.fetch_multiple_seasons(seasons)
    logger.info("Fetched %d total games", len(schedule))

    # 2. Load odds / line movement (if available)
    odds_client = OddsAPIClient(raw_dir=cfg["data"]["raw_dir"])
    lm_processor = LineMovementProcessor(processed_dir=cfg["data"]["processed_dir"])

    # Always (re-)build the line movement table from Action Network so that it
    # covers exactly the seasons we're training on.  Per-date JSON is cached on
    # disk so only new/missing dates hit the network.
    logger.info("Fetching historical odds from Action Network …")
    an_scraper = ActionNetworkScraper(
        raw_dir=cfg["data"]["raw_dir"],
        processed_dir=cfg["data"]["processed_dir"],
    )
    line_movement = an_scraper.build_line_movement(schedule)
    if not line_movement.empty:
        logger.info(
            "Action Network: built line movement for %d games", len(line_movement)
        )
        odds = pd.DataFrame()
    else:
        # Fallback: try a pre-saved closing-spreads file or cached lm table
        line_movement = lm_processor.load()
        if not line_movement.empty:
            logger.info("Using cached line movement data (%d games)", len(line_movement))
            odds = pd.DataFrame()
        else:
            odds_path = Path(cfg["data"]["raw_dir"]) / "historical_spreads.parquet"
            odds = pd.read_parquet(odds_path) if odds_path.exists() else pd.DataFrame()
            if odds.empty:
                logger.warning(
                    "No historical odds or line movement data found. "
                    "ATS metrics will not be available. "
                    "See README for how to add odds data.",
                )

    # 3. Process
    processor = DataProcessor(processed_dir=cfg["data"]["processed_dir"])
    games = processor.build_games_table(
        schedule,
        odds=odds if not odds.empty else None,
        line_movement=line_movement if not line_movement.empty else None,
    )
    game_log = processor.build_team_game_log(games)

    # Fetch box scores and merge richer per-game stats into the game log
    # (results are cached per game_id in data/raw/boxscores/ after first run)
    logger.info("Fetching box score data (cached after first run) …")
    boxscores = scraper.fetch_all_boxscores(schedule)
    if not boxscores.empty:
        game_log = processor.merge_boxscores(game_log, boxscores)
        logger.info("Box score stats merged into game log")

    # 4. Engineer features
    engineer = FeatureEngineer(
        processed_dir=cfg["data"]["processed_dir"],
        windows=cfg["features"]["rolling_windows"],
    )
    features = engineer.build_features(game_log, games)
    logger.info("Feature matrix shape: %s", features.shape)

    # 5. Train
    trainer = ModelTrainer(
        model_type=model_type,
        model_params=cfg["model"],
        artifacts_dir=cfg["model"].get("artifacts_dir", "models/artifacts"),
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
    )

    target = "ats_result" if "ats_result" in features.columns else "home_win"
    if target == "home_win":
        logger.warning(
            "No 'ats_result' column found; training on 'home_win' instead. "
            "Provide odds data for true ATS modelling."
        )

    metrics = trainer.train(features, target=target)
    print("\nTraining complete!")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    # Also train an Over/Under model if the target is available
    ou_target = "over_result"
    if ou_target in features.columns and features[ou_target].notna().sum() > 50:
        logger.info("Training Over/Under model on '%s' …", ou_target)
        ou_trainer = ModelTrainer(
            model_type=model_type,
            model_params=cfg["model"],
            artifacts_dir=cfg.get("model", {}).get("ou_artifacts_dir", "models/artifacts_ou"),
            test_size=cfg["model"]["test_size"],
            random_state=cfg["model"]["random_state"],
        )
        try:
            ou_metrics = ou_trainer.train(features, target=ou_target)
            print(f"\nOver/Under model training complete!")
            for k, v in ou_metrics.items():
                print(f"  {k:<25}: {v}")
        except Exception as exc:
            logger.warning("O/U model training failed: %s", exc)


def cmd_predict(args: argparse.Namespace, cfg: dict) -> None:
    """Predict tonight's games and print picks with confidence."""
    setup_logging(cfg["logging"]["level"])

    # Load trained model
    trainer = ModelTrainer.load()
    logger.info("Loaded trained model")

    live_movement = pd.DataFrame()

    if getattr(args, "games", None):
        # Manual mode: games supplied via --games flag, no API call needed
        upcoming_games = _build_manual_game_rows(args.games)
        if upcoming_games.empty:
            print("Could not parse --games input. Format: 'Away Team:Home Team:spread'")
            return
    else:
        # Fetch upcoming odds and live line snapshots
        odds_client = OddsAPIClient(raw_dir=cfg["data"]["raw_dir"])
        upcoming_odds = odds_client.fetch_upcoming_spreads()

        if upcoming_odds.empty:
            # Odds API unavailable (no key / quota exhausted) — fall back to
            # Action Network which is free and requires no API key.
            logger.info("Odds API unavailable — trying Action Network for today's games …")
            an_scraper = ActionNetworkScraper(
                raw_dir=cfg["data"]["raw_dir"],
                processed_dir=cfg["data"]["processed_dir"],
            )
            upcoming_odds = an_scraper.fetch_today_odds()

        if upcoming_odds.empty:
            print(
                "No upcoming WNBA games found.\n"
                "Either set ODDS_API_KEY in .env, or supply games manually:\n"
                "  python main.py predict --games 'Away Team:Home Team:spread' ..."
            )
            return

        # Filter to today's games only (Odds API returns all upcoming games across
        # multiple days; we only want games starting today in US/Eastern time).
        _ET = ZoneInfo("America/New_York")
        _today_et = pd.Timestamp.now(tz=_ET).date()
        _ct = pd.to_datetime(upcoming_odds["commence_time"], utc=True)
        upcoming_odds = upcoming_odds[
            _ct.dt.tz_convert(_ET).dt.date == _today_et
        ].copy()
        if upcoming_odds.empty:
            print("No WNBA games scheduled for today.")
            return

        # Attempt to fetch live opening vs. current (closing proxy) snapshots
        lm_processor = LineMovementProcessor(processed_dir=cfg["data"]["processed_dir"])
        opening, closing = odds_client.load_cached_snapshots()
        if not opening.empty and not closing.empty:
            live_movement = lm_processor.compute(opening, closing)
            logger.info("Live line movement computed for %d games", len(live_movement))
        else:
            logger.debug(
                "No Odds API snapshots — live line movement features will be absent."
            )

        upcoming_games = _build_upcoming_game_rows(upcoming_odds)

    # Fetch recent game data to build features for upcoming games
    scraper = ESPNScraper(raw_dir=cfg["data"]["raw_dir"])
    schedule = scraper.fetch_multiple_seasons(cfg["data"]["seasons"][-2:])

    processor = DataProcessor(processed_dir=cfg["data"]["processed_dir"])
    games = processor.build_games_table(schedule)
    game_log = processor.build_team_game_log(games)

    # Merge box scores for richer rolling features (uses cache from train run)
    boxscores = scraper.fetch_all_boxscores(schedule)
    if not boxscores.empty:
        game_log = processor.merge_boxscores(game_log, boxscores)

    engineer = FeatureEngineer(
        processed_dir=cfg["data"]["processed_dir"],
        windows=cfg["features"]["rolling_windows"],
    )
    # Attach live movement to upcoming games if available
    upcoming_games_with_movement = (
        upcoming_games.merge(live_movement, on="game_id", how="left")
        if not live_movement.empty
        else upcoming_games
    )
    all_games = pd.concat([games, upcoming_games_with_movement], ignore_index=True)
    all_log = pd.concat(
        [game_log, processor.build_team_game_log(upcoming_games)],
        ignore_index=True,
    )

    features = engineer.build_features(all_log, all_games)
    upcoming_features = features[features["game_id"].isin(upcoming_games["game_id"])]

    if upcoming_features.empty:
        print("Could not build features for upcoming games.")
        return

    feat_cols = [
        c for c in trainer.feature_names
        if c in upcoming_features.columns
    ]
    X = upcoming_features.reindex(columns=trainer.feature_names, fill_value=0)
    # Apply the same median imputation used at training time
    if trainer.train_medians is not None:
        X = X.fillna(trainer.train_medians.reindex(X.columns))
    X = X.fillna(0)
    # Ensure no object-dtype columns sneak through (e.g. neutral_site stored as bool)
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    probs = trainer.model.predict_proba(X)[:, 1]
    min_edge = cfg["betting"]["min_edge"]

    print("\n" + "=" * 80)
    print("  WNBA ATS Picks")
    print("=" * 80)
    print(f"  {'Away':^22} {'Home':^22} {'Favorite':^14} {'Pick':^10} {'Conf':^6}")
    print("  " + "-" * 75)

    for i, (_, row) in enumerate(upcoming_features.iterrows()):
        spread_val = row.get("spread", float("nan"))
        prob = probs[i]

        # Identify the favorite and show as "TEAM -X.X"
        if pd.notna(spread_val):
            if spread_val < 0:
                fav_team = row["home_team"].split()[-1]
                fav_str = f"{fav_team} {spread_val:+.1f}"
            elif spread_val > 0:
                fav_team = row["away_team"].split()[-1]
                fav_str = f"{fav_team} {-spread_val:+.1f}"
            else:
                fav_str = "Pick'em"
        else:
            fav_str = "N/A"

        pick = "HOME CVR" if prob >= 0.5 else "AWAY CVR"
        conf = prob if prob >= 0.5 else 1.0 - prob
        flag = " ★" if abs(conf - 0.5) >= min_edge else ""

        home_str = row["home_team"].split()[-1]
        away_str = row["away_team"].split()[-1]
        print(
            f"  {away_str:^22} {home_str:^22} "
            f"{fav_str:^14} {pick:^10} {conf:.1%}{flag}"
        )

    print()
    print("  ★ = model confidence exceeds edge threshold")
    print("=" * 80 + "\n")

    # --- Over/Under picks (load separate O/U model if available) ---
    ou_artifacts_dir = Path(cfg.get("model", {}).get("ou_artifacts_dir", "models/artifacts_ou"))
    ou_model_path = ou_artifacts_dir / "model.pkl"
    if ou_model_path.exists():
        try:
            ou_trainer = ModelTrainer.load(artifacts_dir=ou_artifacts_dir)
            X_ou = upcoming_features.reindex(columns=ou_trainer.feature_names, fill_value=0)
            if ou_trainer.train_medians is not None:
                X_ou = X_ou.fillna(ou_trainer.train_medians.reindex(X_ou.columns))
            X_ou = X_ou.fillna(0)
            for col in X_ou.select_dtypes(include="object").columns:
                X_ou[col] = pd.to_numeric(X_ou[col], errors="coerce").fillna(0)
            ou_probs = ou_trainer.model.predict_proba(X_ou)[:, 1]

            print("=" * 80)
            print("  WNBA O/U Picks")
            print("=" * 80)
            print(f"  {'Away':^22} {'Home':^22} {'Total':^10} {'Pick':^8} {'Conf':^6}")
            print("  " + "-" * 75)

            for i, (_, row) in enumerate(upcoming_features.iterrows()):
                total_line = row.get("total_line", row.get("total", float("nan")))
                total_str = f"O/U {total_line:.1f}" if pd.notna(total_line) else "N/A"
                ou_prob = ou_probs[i]
                ou_pred = "OVER" if ou_prob >= 0.5 else "UNDER"
                ou_conf = ou_prob if ou_prob >= 0.5 else 1.0 - ou_prob
                flag = " ★" if abs(ou_conf - 0.5) >= min_edge else ""
                home_str2 = row["home_team"].split()[-1]
                away_str2 = row["away_team"].split()[-1]
                print(
                    f"  {away_str2:^22} {home_str2:^22} "
                    f"{total_str:^10} {ou_pred:^8} {ou_conf:.1%}{flag}"
                )
            print()
            print("  ★ = model confidence exceeds edge threshold")
            print("=" * 80 + "\n")
        except Exception as exc:
            logger.warning("Could not load O/U model: %s", exc)

    # --- Injury Report (live ESPN data, no historical archive) ---
    try:
        injury_data = scraper.fetch_injuries()
        if injury_data:
            teams_in_play: set[str] = set()
            for _, row in upcoming_features.iterrows():
                teams_in_play.add(row.get("home_team", ""))
                teams_in_play.add(row.get("away_team", ""))
            teams_in_play.discard("")

            relevant: dict[str, list[dict]] = {
                t: injury_data[t] for t in teams_in_play if t in injury_data
            }

            if relevant:
                print("=" * 85)
                print("  INJURY REPORT  (live ESPN data)")
                print("=" * 85)
                for team_name, players in sorted(relevant.items()):
                    for p in players:
                        status = p.get("status", "Unknown")
                        name = p.get("name", "")
                        pos = p.get("position", "")
                        flag = "⚠ " if status.lower() in ("out", "doubtful") else "  "
                        pos_str = f"[{pos}] " if pos else ""
                        print(f"  {flag}{team_name}: {name} {pos_str}— {status}")
                print("=" * 85 + "\n")
    except Exception as exc:
        logger.debug("Injury fetch skipped: %s", exc)

    print("=" * 77 + "\n")


def cmd_evaluate(args: argparse.Namespace, cfg: dict) -> None:
    """Run evaluation report on existing feature data."""
    setup_logging(cfg["logging"]["level"])
    features_path = Path(cfg["data"]["processed_dir"]) / "features.parquet"

    if not features_path.exists():
        print(f"No features found at {features_path}. Run 'train' first.")
        return

    features = pd.read_parquet(features_path)
    trainer = ModelTrainer.load()

    target = "ats_result" if "ats_result" in features.columns else "home_win"
    df = features.dropna(subset=[target]).sort_values("date")
    split_idx = int(len(df) * (1 - cfg["model"]["test_size"]))
    test_df = df.iloc[split_idx:]

    feat_cols = [c for c in trainer.feature_names if c in test_df.columns]
    X_test = test_df[feat_cols].fillna(test_df[feat_cols].median())
    y_test = test_df[target].astype(int).values

    preds = trainer.model.predict(X_test)
    probs = trainer.model.predict_proba(X_test)[:, 1]

    evaluator = Evaluator()
    # Pass closing spread for CLV computation (feature "spread" = closing spread)
    report = evaluator.full_report(preds, probs, y_test, meta=test_df)
    evaluator.print_report(report)


def _build_upcoming_game_rows(odds: pd.DataFrame) -> pd.DataFrame:
    """Convert upcoming odds rows into a minimal games-table format."""
    rows = []
    for _, row in odds.drop_duplicates("game_id").iterrows():
        ts = pd.Timestamp(row["commence_time"])
        # Normalise to tz-naive so it can be sorted alongside historical dates
        date = ts.tz_convert(None) if ts.tzinfo is not None else ts
        rows.append(
            {
                "game_id": row["game_id"],
                "date": date,
                "season": date.year,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": 0,
                "away_score": 0,
                "status": "upcoming",
                "spread": row.get("home_spread"),
                "home_public_pct": row.get("home_public_pct"),
                "total": row.get("total"),
            }
        )
    return pd.DataFrame(rows)


def _build_manual_game_rows(game_specs: list[str]) -> pd.DataFrame:
    """
    Parse manual game specs into a minimal games-table format.

    Each spec must be:  "Away Team:Home Team:spread"
    e.g.  "Las Vegas Aces:New York Liberty:-3.5"
    Spread is from the home team's perspective (negative = home favoured).
    """
    rows = []
    today = pd.Timestamp.now().normalize()
    for i, spec in enumerate(game_specs):
        parts = spec.rsplit(":", maxsplit=2)
        if len(parts) != 3:
            logger.warning(
                "Skipping malformed game spec %r — expected 'Away:Home:spread'", spec
            )
            continue
        away, home, spread_str = [p.strip() for p in parts]
        try:
            spread = float(spread_str)
        except ValueError:
            logger.warning("Skipping %r — spread %r is not a number", spec, spread_str)
            continue
        rows.append(
            {
                "game_id": f"manual_{i}",
                "date": today,
                "season": today.year,
                "home_team": home,
                "away_team": away,
                "home_score": 0,
                "away_score": 0,
                "status": "upcoming",
                "spread": spread,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WNBA ATS Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train the ATS model")
    train_parser.add_argument(
        "--seasons", nargs="+", type=int, help="Seasons to train on (e.g. 2022 2023)"
    )
    train_parser.add_argument(
        "--model", choices=["xgboost", "lightgbm", "logistic"],
        help="Model type override"
    )

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Generate picks for upcoming games")
    predict_parser.add_argument(
        "--games",
        nargs="+",
        metavar="AWAY:HOME:SPREAD",
        help=(
            "Manually supply today's games instead of fetching from the Odds API. "
            "Format: 'Away Team:Home Team:spread' (spread is home-team perspective). "
            "Example: 'Las Vegas Aces:New York Liberty:-3.5'"
        ),
    )

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on test set")
    eval_parser.add_argument(
        "--seasons", nargs="+", type=int, help="Seasons to evaluate"
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "train":
        cmd_train(args, cfg)
    elif args.command == "predict":
        cmd_predict(args, cfg)
    elif args.command == "evaluate":
        cmd_evaluate(args, cfg)


if __name__ == "__main__":
    main()
