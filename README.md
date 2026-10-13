# Quest #3: The Compass 🧭 — Travel Destination Personalization

Predict which country a new PlaceUp user will book — balancing predictive
accuracy against the business need for **diverse, explainable** recommendations.

📓 **[View the fully-rendered notebook on nbviewer](https://nbviewer.org/github/nitayv658/placeup-travel-personalization/blob/main/notebooks/quest3_the_compass.ipynb)** — recommended (GitHub's inline notebook preview can be unreliable for larger notebooks).

## Problem
10,000 users, one row each. The target `country_destination` has 8 heavily
imbalanced classes, ~48% of them `NDF` ("no destination found" — the user never
booked). "Always predict NDF" scores ~48% accuracy and is worthless, so accuracy
alone is the wrong yardstick.

## Approach
A **two-stage cascade**:
1. **Gate** — will this user book at all? (binary: book vs `NDF`)
2. **Ranker** — for likely bookers, which of the 7 real countries? Trained on
   booked users only (leakage guard), surfaced as a **top-3 shortlist**.

Each stage benchmarks a **LogisticRegression baseline vs. LightGBM**, selected on
validation and reported once on test. Metrics go beyond accuracy: PR-AUC for the
gate; top-3 accuracy **with a diversity veto** for the ranker (a model that
recommends the same country to everyone fails, even at higher top-1). SHAP
explains each stage.

## Headline result (held-out test)
- **Gate:** PR-AUC 0.570 (LightGBM).
- **Ranker:** top-3 0.784, recommending all 7 destinations — vs. the baseline's
  4 destinations with 91% funneled into one.
- Label-ambiguity ceiling ~57% top-1 (look-alike users disagree 47% of the time)
  → the business case for a shortlist over a single guess.

## Repo layout
| Path | What |
|---|---|
| `notebooks/quest3_the_compass.ipynb` | The submission — Part A summary · Part B cleaning+EDA · Part C models+SHAP |
| `src/data_prep.py` | Loading, cleaning, leakage-safe split & preprocessing |
| `src/kpis.py` | Business + model KPIs |
| `data/` | Raw dataset + processed train/val/test splits |
| `reports/figures/` | SHAP + EDA plots |

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/quest3_the_compass.ipynb   # then Restart & Run All
```

---
*Author: Nitay Vitkin · [GitHub](https://github.com/nitayv658)*
