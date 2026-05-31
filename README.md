# wnba-predictions

Models WNBA team form and flags disagreements with betting lines.

**No API keys required.** Data is sourced from ESPN and Action Network, both free and public.

---

## How it works

The system builds ~780 features per game -- rolling form, four factors, pace and efficiency ratings, rest, travel, line movement, and public betting percentages -- then trains separate XGBoost/LightGBM ensembles to predict spread coverage (ATS) and totals (O/U).

Backtesting on data back to 2018:

- **ATS**: when the model's probability disagrees with the market's closing implied probability by 5+ percentage points, those flagged games return roughly **+5% ROI** -- compared to the roughly -5% a random bettor expects after vig.
- **O/U**: games where the model's confidence exceeds 55% (i.e. 5+ pp from 50/50) return roughly **+2.6% ROI**, though this is a raw confidence threshold rather than a true market comparison -- the O/U model has no equivalent closing-line implied probability to compare against.

All features use only data available before tip-off, and training uses a strict temporal 80/20 split with no shuffling, so the backtest reflects real deployment conditions.

---

## Quickstart

```bash
git clone https://github.com/yourname/wnba-predictions
cd wnba-predictions
python -m venv .venv
```

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Train

Downloads all data automatically on first run (~2 minutes).

```bash
python predict.py train --seasons 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

### Generate Today's Picks

```bash
python predict.py predict
```

Example output:

```text
Away      Home      Favorite     Pick       Conf    Mkt Edge
Storm     Aces      Aces -5.5    AWAY CVR   54.2%   -0.087 *
Sky       Dream     Pick'em      HOME CVR   51.8%   +0.018
Fever     Lynx      Lynx -3.0    HOME CVR   53.1%   +0.063 *

* = model disagrees with market by >= 5 pp (value bet)
Mkt Edge = model_prob - market_implied
           (+ favors home, - favors away)
```

### Evaluate Model on Held-Out Test Data

```bash
python predict.py evaluate
```

---

## Results (2018-2026, last 20% held out, ~388 games)

**ATS model**

| Metric | Value |
|---|---|
| Test accuracy | 53.1% |
| Flat-bet ROI | +0.45% |
| ROI on >=5% edge bets | +5.2% (372 bets) |
| ROI on >=7% edge bets | +6.2% (354 bets) |

**O/U model**

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

```bash
pytest tests/ -v
```
---

## Disclaimer

This is a research tool. Sports betting involves financial risk. Past backtested performance does not guarantee future results.
