"""Business-facing KPI computations for the PlaceUp travel dataset.

Segment-level KPIs (conversion rate, destination affinity, data trust) only need
the raw/cleaned dataframe. Model-dependent KPIs (personalization lift, confidence
coverage) take a fitted model's predicted probabilities as input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_prep import TARGET

NDF_LABEL = "NDF"


def add_age_bucket(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    df = df.copy()
    bins = [0, 24, 34, 44, 54, 200]
    labels = ["<25", "25-34", "35-44", "45-54", "55+"]
    df["age_bucket"] = pd.cut(df[age_col], bins=bins, labels=labels)
    return df


def conversion_rate(df: pd.DataFrame, by: str | None = None):
    """Share of users whose country_destination != NDF, overall or per segment."""
    converted = df[TARGET] != NDF_LABEL
    if by is None:
        return converted.mean()
    return converted.groupby(df[by], observed=True).mean().sort_values(ascending=False)


def destination_affinity(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Normalized country_destination distribution within each value of `by`."""
    return pd.crosstab(df[by], df[TARGET], normalize="index").round(3).sort_index()


def top_destination_per_segment(df: pd.DataFrame, by: str, exclude_ndf: bool = True) -> pd.DataFrame:
    """For each segment value, the most common (non-NDF) destination and its share."""
    data = df[df[TARGET] != NDF_LABEL] if exclude_ndf else df
    affinity = destination_affinity(data, by)
    return pd.DataFrame({"top_destination": affinity.idxmax(axis=1), "share": affinity.max(axis=1)})


def data_trust_kpi(df: pd.DataFrame, flag_col: str = "age_missing_or_invalid") -> pd.DataFrame:
    """Data-quality flag rate overall and its relationship to conversion."""
    conversion_by_flag = conversion_rate(df, by=flag_col)
    return pd.DataFrame(
        {
            "flag_rate": [df[flag_col].mean()],
            "conversion_when_flagged": [conversion_by_flag.get(1, np.nan)],
            "conversion_when_clean": [conversion_by_flag.get(0, np.nan)],
        }
    )


def top_k_accuracy(y_true, proba: np.ndarray, classes: np.ndarray, k: int = 3) -> float:
    """Fraction of rows where the true label is within the top-k predicted classes."""
    y_true = np.asarray(y_true)
    top_k_labels = classes[np.argsort(-proba, axis=1)[:, :k]]
    return float(np.mean([y_true[i] in top_k_labels[i] for i in range(len(y_true))]))


def personalization_lift(y_true, proba: np.ndarray, classes: np.ndarray, baseline_class: str = NDF_LABEL) -> pd.DataFrame:
    """Model top-1/top-3 accuracy vs. a naive 'always predict the majority class' baseline."""
    y_true = np.asarray(y_true)
    return pd.DataFrame(
        {
            "metric": ["baseline_top1 (always NDF)", "model_top1", "model_top3"],
            "accuracy": [
                float((y_true == baseline_class).mean()),
                top_k_accuracy(y_true, proba, classes, k=1),
                top_k_accuracy(y_true, proba, classes, k=3),
            ],
        }
    )


def confidence_coverage(
    y_true,
    proba: np.ndarray,
    classes: np.ndarray,
    thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> pd.DataFrame:
    """At each confidence threshold: the share of users a model is confident
    enough to personalize for (coverage), and how accurate those calls are.
    """
    y_true = np.asarray(y_true)
    max_proba = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    rows = []
    for t in thresholds:
        mask = max_proba >= t
        rows.append(
            {
                "threshold": t,
                "coverage": float(mask.mean()),
                "precision_within_covered": float((pred[mask] == y_true[mask]).mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def diversity_at_1(pred, n_possible: int | None = None) -> pd.DataFrame:
    """Guardrail for the 'where?' stage: how spread-out are the top-1
    recommendations across the user base?

    A model can score decent top-3 while recommending the same popular
    destination to everyone — that is accurate-but-not-personalised. This
    reports how many distinct destinations actually get recommended, the share
    funneled to the single most-recommended one (higher = more homogeneous),
    and the normalised entropy of the recommendation mix (1.0 = perfectly even,
    0.0 = everyone sent to one place).
    """
    pred = np.asarray(pred)
    counts = pd.Series(pred).value_counts()
    p = counts / counts.sum()
    k = n_possible if n_possible is not None else len(counts)
    norm_entropy = float(-(p * np.log(p)).sum() / np.log(k)) if k > 1 else 0.0
    return pd.DataFrame(
        {
            "metric": ["distinct_recommended", "top_dest_share", "diversity_entropy"],
            "value": [int(counts.size), float(p.iloc[0]), norm_entropy],
        }
    )


def booking_metrics(y_booked, proba_booked, threshold: float = 0.5) -> pd.DataFrame:
    """Stage-1 (book vs NDF) metrics. `y_booked` is 1 when the user booked
    (destination != NDF); `proba_booked` is the model's P(book).

    Leads with PR-AUC because it is threshold-free and honest under the ~52/48
    book/NDF split, then reports precision/recall at a chosen operating point
    (the threshold is the personalize-vs-show-generic gate — tune it to the
    business cost, don't leave it at 0.5 by default without justifying it).
    """
    from sklearn.metrics import average_precision_score, precision_score, recall_score

    y = np.asarray(y_booked)
    proba_booked = np.asarray(proba_booked)
    pred = (proba_booked >= threshold).astype(int)
    return pd.DataFrame(
        {
            "metric": ["pr_auc", f"precision@{threshold}", f"recall@{threshold}"],
            "value": [
                float(average_precision_score(y, proba_booked)),
                float(precision_score(y, pred, zero_division=0)),
                float(recall_score(y, pred, zero_division=0)),
            ],
        }
    )
