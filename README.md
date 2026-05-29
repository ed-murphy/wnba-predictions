# wnba-predictions

A machine learning system that reads the WNBA betting market, builds a statistical model of each team's form, and tells you when the model disagrees with the bookmakers â€” and by how much.

**No API keys required.** All data is sourced from ESPN and Action Network, both free and public.

---

## What this does (no statistics background required)

Every WNBA game has a **spread** â€” a handicap set by bookmakers like DraftKings or FanDuel. For example, if the Las Vegas Aces are playing the New York Liberty and the spread is `Aces -5.5`, that means the bookmakers believe Las Vegas is good enough to win by more than 5.5 points. If you bet on the Aces, they need to win by 6 or more for your bet to win. If you bet on Liberty +5.5, Liberty just needs to lose by fewer than 6 (or win outright).

The spread is set by professional oddsmakers, refined by millions of dollars in wagers, and is extremely hard to beat. But it's not perfect.

This system tries to find the cracks. It does three things:

**1. Collects everything.** It automatically downloads WNBA game scores and box-score stats from ESPN going back to 2018 â€” over 2,000 games. It also downloads every betting line from Action Network, including how the line moved between when it opened and when it closed.

**2. Builds a picture of each team.** For every game, it calculates nearly 800 numbers describing both teams: how they've performed over the last 3, 5, and 10 games; how efficient they've been offensively and defensively; how tired they might be from travel or back-to-back games; how well they've been covering spreads historically; and whether sharp ("smart") money moved the line in a particular direction before tip-off.

**3. Finds where the model and market disagree.** The core insight is this: if the bookmakers have the home team as a 65% probability to cover, and our model thinks it's only 50%, that's a meaningful disagreement. Betting on the away team in that situation is what's called "getting good value" â€” you're exploiting a gap between the market price and what the data suggests is true. The system flags those games with a â˜….

Backtesting on 2018â€“2026 data shows that games flagged this way (where the model disagrees with the closing line by 5+ percentage points) have returned roughly **+5% ROI** on the ATS side. That's meaningful â€” the standard expectation for a random bettor is about âˆ’5% per bet (the bookmaker's cut, called the "vig").

---

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/yourname/wnba-predictions
cd wnba-predictions
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model

Downloads all data automatically on first run. Subsequent runs use cached data and are fast (~2 minutes).

```bash
python predict.py train --seasons 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

### 3. Generate today's picks

```bash
python predict.py predict
```

Output example:
```
=============================================================================================================================================
  WNBA ATS Picks
=============================================================================================================================================
  Away                   Home                   Favorite       Pick       Conf    Mkt Edge
  --------------------------------------------------------------------------------------------
  Storm                  Aces                   Aces -5.5    AWAY CVR   54.2%   -0.087 â˜…
  Sky                    Dream                  Pick'em      HOME CVR   51.8%   +0.018
  Fever                  Lynx                   Lynx -3.0    HOME CVR   53.1%   +0.063 â˜…

  â˜… = model disagrees with market line by â‰¥ min_clv_edge (value bet)
  Mkt Edge = model_prob âˆ’ market_implied (+ favours home, âˆ’ favours away)
```

Games marked â˜… are the value bets â€” where the model and market disagree by at least 5 percentage points.

### 4. Evaluate on held-out test data

```bash
python predict.py evaluate
```

---

## Project Structure

```
wnba-predictions/
â”œâ”€â”€ predict.py                        # CLI entry point (train / predict / evaluate)
â”œâ”€â”€ config/
â”‚   â””â”€â”€ config.yaml                   # All tuneable parameters
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ espn_scraper.py           # ESPN public API â€” schedules, box scores, injuries
â”‚   â”‚   â”œâ”€â”€ action_network_scraper.py # Action Network â€” historical & live odds
â”‚   â”‚   â”œâ”€â”€ line_movement.py          # Opening/closing line movement & steam detection
â”‚   â”‚   â”œâ”€â”€ odds_client.py            # The Odds API client (optional live lines)
â”‚   â”‚   â””â”€â”€ processor.py              # Data cleaning, merging, ATS/O/U labelling
â”‚   â”œâ”€â”€ features/
â”‚   â”‚   â””â”€â”€ engineer.py               # Feature engineering pipeline (~778 features)
â”‚   â”œâ”€â”€ models/
â”‚   â”‚   â”œâ”€â”€ train.py                  # Ensemble training + probability calibration
â”‚   â”‚   â””â”€â”€ evaluate.py               # ATS/CLV-focused evaluation metrics
â”‚   â””â”€â”€ utils/
â”‚       â”œâ”€â”€ config.py                 # YAML config loader
â”‚       â””â”€â”€ logging.py                # Logging setup
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                          # Cached ESPN & Action Network data (gitignored)
â”‚   â”‚   â”œâ”€â”€ action_network/           # Per-date JSON files (one per game-date)
â”‚   â”‚   â”œâ”€â”€ boxscores/                # Per-game box score cache
â”‚   â”‚   â”œâ”€â”€ schedule_<year>.parquet   # Season schedule cache
â”‚   â”‚   â””â”€â”€ historical_spreads.parquet
â”‚   â””â”€â”€ processed/                    # Cleaned tables & feature matrix (gitignored)
â”‚       â”œâ”€â”€ games.parquet
â”‚       â”œâ”€â”€ team_game_log.parquet
â”‚       â”œâ”€â”€ features.parquet
â”‚       â”œâ”€â”€ market_context.parquet
â”‚       â””â”€â”€ line_movement.parquet
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ artifacts/                    # ATS model (gitignored) + feature_names.txt
â”‚   â””â”€â”€ artifacts_ou/                 # O/U model (gitignored) + feature_names.txt
â””â”€â”€ tests/
    â”œâ”€â”€ test_features.py
    â”œâ”€â”€ test_line_movement.py
    â”œâ”€â”€ test_models.py
    â””â”€â”€ test_processor.py
```

---

## Data Sources

| Source | What it provides | API key? |
|--------|-----------------|----------|
| ESPN public API | Schedules, scores, box scores, injury reports | No |
| Action Network public API | Historical spreads, line movement, public betting % | No |
| The Odds API | Optional live lines for in-season predictions | Yes (free tier) |

### ESPN (`src/data/espn_scraper.py`)
Hits `site.api.espn.com/apis/site/v2/sports/basketball/wnba`. Past seasons are cached permanently as Parquet files. The current season is always re-fetched to capture completed games. Box scores are cached per game ID after first fetch.

### Action Network (`src/data/action_network_scraper.py`)
Hits `api.actionnetwork.com/web/v1/scoreboard/wnba?date=YYYYMMDD`. Returns the full line-history for every bookmaker on every game for that date. Per-date JSON is cached in `data/raw/action_network/`. Only dates not already cached hit the network. From the raw history the scraper derives:

- **Opening spread** â€” earliest recorded line per book
- **Closing spread** â€” latest recorded line per book
- **Total (O/U)** â€” closing total line
- **Public betting %** â€” percentage of bets on the home side
- **Line movement magnitude** â€” how far the line moved open â†’ close
- **Steam move flag** â€” rapid sharp-money movement across multiple books simultaneously

---

## Feature Engineering

Source: `src/features/engineer.py` â€” produces ~778 features per game row.

### Feature Groups

| Group | Description | Examples |
|-------|-------------|---------|
| **Season-to-date** | Cumulative stats up to (not including) the current game | `home_season_margin_mean`, `away_season_win_mean` |
| **Rolling form** | Mean over last 3, 5, and 10 games | `home_roll5_off_rtg_mean`, `diff_roll10_net_rtg_std` |
| **Four Factors** | eFG%, TOV%, FTA rate, ORB% (Dean Oliver's framework) | `home_roll5_efg_pct_mean`, `away_season_tov_pct_mean` |
| **Pace & Ratings** | Per-possession offensive/defensive/net ratings, pace | `home_roll5_off_rtg_mean`, `diff_roll10_def_rtg_mean` |
| **Pythagorean expectation** | Points-based win probability estimate | `home_season_pyth_exp` |
| **Implied probability** | Market-implied home win probability from closing spread | `home_implied_prob`, `pyth_vs_market` |
| **Rest / schedule density** | Days rest, back-to-back flag, games in last 7 days | `home_rest_days`, `away_is_b2b`, `home_games_last_7` |
| **Travel distance** | Haversine distance flown by away team, cross-country flag | `away_travel_miles`, `cross_country`, `travel_b2b_burden` |
| **ATS history** | Cover rate season-to-date, home/away splits *(ATS model only â€” excluded from O/U)* | `home_season_ats_cover_mean`, `home_season_ats_home_cover` |
| **Over/Under tracking** | Team over rate history, pace-based scoring expectation | `home_season_over_rate`, `expected_total_10` |
| **Consistency** | Rolling standard deviation (measures variance/volatility) | `home_season_margin_std`, `diff_roll10_net_rtg_std` |
| **Head-to-head** | Season H2H record and ATS cover rate between these two teams | `home_h2h_wins`, `home_h2h_ats_cover` |
| **Line movement** | Open-to-close movement magnitude, steam flag, reverse line movement | `spread_move`, `steam_move`, `reverse_line_movement` |
| **Public betting %** | % of bets on home side (proxy for public money direction) | `home_public_pct` |

All features are prefixed `home_`, `away_`, or `diff_` (home minus away differential). Differentials capture relative matchup quality.

### Temporal Safety
All season-to-date and rolling features use `.shift(1)` before any cumulative operation â€” a game at index *t* only ever sees data from games at indices *0 â€¦ tâˆ’1*. This prevents data leakage from future games.

---

## Models

### Architecture

Two independent classifiers are trained:

| Model | Target | Notes |
|-------|--------|-------|
| **ATS model** | `ats_result` (1 = home covered) | All features included |
| **O/U model** | `over_result` (1 = total exceeded line) | ATS-specific features (`*_ats_*`) excluded to prevent target leakage |

Both use the same **ensemble** approach:

1. **XGBoost** + **LightGBM** trained independently on the same feature set
2. Each is wrapped in `CalibratedClassifierCV` with **TimeSeriesSplit (5 folds)** and sigmoid calibration, so probability outputs reflect true historical frequencies
3. The two calibrated models' probabilities are **averaged** (`AveragingEnsemble`) to reduce variance

### Feature Selection

Before fitting, `SelectKBest` with mutual information scoring selects the most predictive features. Feature selection is fit on training data only.

- ATS model: top **65** features
- O/U model: top **50** features (lower to reduce overfit risk on the noisier totals target)

### Train/Test Split

A strict **temporal 80/20 split** â€” no shuffling. The last 20% of games by date form the held-out test set, simulating real deployment where you never train on future data.

### Hyperparameters

```yaml
xgboost:
  n_estimators: 200
  max_depth: 3          # shallow trees prevent overfit on ~1500 samples
  learning_rate: 0.02
  subsample: 0.65
  colsample_bytree: 0.4
  min_child_weight: 20
  gamma: 1.5
  reg_alpha: 2.0        # L1 regularization
  reg_lambda: 8.0       # L2 regularization (strong, given small dataset)

lightgbm:
  n_estimators: 200
  max_depth: 3
  learning_rate: 0.02
  num_leaves: 15
  subsample: 0.65
  colsample_bytree: 0.4
  min_child_samples: 30
  reg_alpha: 2.0
  reg_lambda: 8.0
```

---

## Evaluation & Backtesting

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **ATS accuracy** | % of test-set picks where the predicted side covered |
| **Brier score** | Probability calibration quality (lower = better; 0.25 = coin flip baseline) |
| **Flat-bet ROI** | Return betting $110 to win $100 on every pick |
| **Kelly ROI** | ROI using quarter-Kelly position sizing (sized by model confidence) |
| **mean_clv_all** | Mean `model_prob âˆ’ closing_implied` across all test games |
| **roi_clv03/05/07** | ROI when betting only games with â‰¥3%/5%/7% modelâ€“market disagreement |

Breakeven at âˆ’110 vig requires â‰¥ 52.4% win rate.

### Current Results (2018â€“2026, last 20% as held-out test, ~388 games)

**ATS Model**

| Metric | Value |
|--------|-------|
| Train accuracy | 51.3% |
| Test accuracy | 53.1% |
| Brier score | 0.2488 |
| Flat-bet ROI | +0.45% |
| CLV-aware ROI (â‰¥3% edge) | +3.2% on 383 bets |
| CLV-aware ROI (â‰¥5% edge) | +5.2% on 372 bets |
| CLV-aware ROI (â‰¥7% edge) | +6.2% on 354 bets |

**O/U Model**

| Metric | Value |
|--------|-------|
| Train accuracy | 66.4% |
| Test accuracy | 52.3% |
| Brier score | 0.2497 |
| Flat-bet ROI | âˆ’1.9% |
| CLV-aware ROI (â‰¥3% edge) | +2.7% on 381 bets |
| CLV-aware ROI (â‰¥5% edge) | +2.6% on 372 bets |

The O/U model's 14-point train/test gap indicates remaining overfit; results should be treated with more caution than the ATS model.

---

## Closing Line Value (CLV)

CLV is the single most important long-run indicator of genuine market edge.

$$\text{CLV} = p_{\text{model}} - p_{\text{closing line}}$$

The closing line's implied probability for the home team is:

$$p_{\text{closing}} = \Phi\!\left(\frac{-\text{spread}}{12}\right)$$

where Ïƒ = 12 points matches historical WNBA scoring-margin variance, and Î¦ is the standard normal CDF.

The closing line reflects all sharp-money action and is the most efficient betting price. The system's overall CLV is negative (the model is typically less bullish on home favorites than the market), but it generates positive ROI by betting **against** overpriced favorites â€” a contrarian pattern that is structurally sound.

### CLV-Aware Betting Strategy

| Condition | Action |
|-----------|--------|
| `model_prob âˆ’ closing_implied > min_clv_edge` | Bet HOME (model finds home value) |
| `closing_implied âˆ’ model_prob > min_clv_edge` | Bet AWAY (model fades overpriced home) |
| `\|model_prob âˆ’ closing_implied\| â‰¤ min_clv_edge` | No bet (no meaningful disagreement) |

Default `min_clv_edge = 0.05` (5 percentage points). Adjustable in `config.yaml`.

---

## Configuration Reference

`config/config.yaml`:

```yaml
data:
  seasons: [2018, ..., 2026]   # Seasons to train on
  raw_dir: data/raw
  processed_dir: data/processed

features:
  rolling_windows: [3, 5, 10]  # Windows for rolling-form features
  min_games: 5                  # Min games before a team is included in predictions

model:
  type: ensemble                # xgboost | lightgbm | logistic | ensemble
  test_size: 0.2                # Fraction of games held out for testing
  random_state: 42
  artifacts_dir: models/artifacts
  ou_artifacts_dir: models/artifacts_ou

betting:
  vig: -110                     # Standard -110 juice assumed on all bets
  min_edge: 0.05                # Confidence threshold when no market line is available
  min_clv_edge: 0.05            # Modelâ€“market disagreement threshold for â˜… flag
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: feature engineering (no-leakage checks, output shape), line movement detection, model training (sanity checks), and data processor output schema.

---

## Disclaimer

This is a research tool. Sports betting involves substantial financial risk. Past backtested performance does not guarantee future results. The bookmakers' closing line is extremely hard to beat consistently at scale. Use responsibly.
