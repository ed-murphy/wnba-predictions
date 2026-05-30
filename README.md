# wnba-predictions

A machine learning system that models WNBA team form and flags games where the model disagrees with the closing betting line -- and by how much.

**No API keys required.** Data is sourced from ESPN and Action Network, both free and public.

---

## How it works

Every WNBA game has a spread set by bookmakers. This system builds ~780 features per game -- rolling form, four factors, pace/ratings, rest, travel, line movement, and public betting percentages -- then trains an XGBoost/LightGBM ensemble to predict the probability that each side covers.

Games where the model's probability disagrees with the market's implied probability by 5+ percentage points are flagged as value bets (*). Backtesting on 2018-2026 data shows roughly **+5% ROI** on those flagged games on the ATS side, compared to the roughly -5% a random bettor expects after vig.

A strict temporal 80/20 split (no shuffling) is used throughout to prevent data leakage.

---

## Quickstart

\\ash
git clone https://github.com/yourname/wnba-predictions
cd wnba-predictions
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
\
**Train** (downloads all data automatically on first run, ~2 min):
\\ash
python predict.py train --seasons 2018 2019 2020 2021 2022 2023 2024 2025 2026
\
**Generate today's picks:**
\\ash
python predict.py predict
\
\  Away      Home      Favorite     Pick       Conf    Mkt Edge
  Storm     Aces      Aces -5.5    AWAY CVR   54.2%   -0.087 *
  Sky       Dream     Pick'em      HOME CVR   51.8%   +0.018
  Fever     Lynx      Lynx -3.0    HOME CVR   53.1%   +0.063 *

  * = model disagrees with market by >= 5 pp (value bet)
  Mkt Edge = model_prob - market_implied  (+ favors home, - favors away)
\
**Evaluate on held-out test data:**
\\ash
python predict.py evaluate
\
---

## Results (2018-2026, last 20% held out, ~388 games)

**ATS model**

| Metric | Value |
|---|---|
| Test accuracy | 53.1% |
| Flat-bet ROI | +0.45% |
| ROI on >=5% edge bets | +5.2% (372 bets) |
| ROI on >=7% edge bets | +6.2% (354 bets) |

**O/U model** (treat with more caution -- larger train/test gap)

| Metric | Value |
|---|---|
| Test accuracy | 52.3% |
| Flat-bet ROI | -1.9% |
| ROI on >=5% edge bets | +2.6% (372 bets) |

Breakeven at -110 vig requires >= 52.4% win rate.

---

## Project structure

\predict.py              # CLI: train / predict / evaluate
config/config.yaml      # Tunable parameters
src/
  data/
    espn_scraper.py     # Schedules, box scores (ESPN public API)
    action_network_scraper.py  # Historical odds & line movement
    line_movement.py    # Open/close movement, steam detection
    processor.py        # Cleaning, merging, ATS/O/U labeling
  features/engineer.py  # Feature pipeline (~780 features)
  models/
    train.py            # Ensemble training + calibration
    evaluate.py         # ATS/CLV metrics
data/
  raw/                  # Cached ESPN & Action Network data (gitignored)
  processed/            # Feature matrix (gitignored)
models/
  artifacts/            # ATS model weights (gitignored)
  artifacts_ou/         # O/U model weights (gitignored)
tests/                  # pytest suite
\
---

## Running tests

\\ash
pytest tests/ -v
\
---

## Disclaimer

This is a research tool. Sports betting involves substantial financial risk. Past backtested performance does not guarantee future results. Use responsibly.
