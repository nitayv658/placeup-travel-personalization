"""Loading, cleaning, and shared preprocessing pipeline for the PlaceUp travel dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "country_destination"
ID_COL = "user_id"
NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = [
    "gender",
    "signup_method",
    "signup_flow",
    "affiliate_channel",
    "first_device_type",
]
ENGINEERED_FEATURES = ["age_missing_or_invalid", "gender_missing"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES

# gender is never null, but ~15% of rows carry the placeholder "UNKNOWN" — missingness
# already string-filled upstream. Treat it consistently with age: flag the missingness
# as its own feature, while keeping UNKNOWN as a category so the model can still learn a
# "prefers not to say" signal separately from the flag.
GENDER_MISSING_TOKEN = "UNKNOWN"

# Plausible adult signup age range; values outside this are treated as data-entry
# errors (e.g. birth years like 2014 typed into an age field) rather than real ages.
MIN_PLAUSIBLE_AGE = 13
MAX_PLAUSIBLE_AGE = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "PlaceUp_travel_users_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """Flag missing/implausible ages and null out the implausible ones.

    The flag is kept as a feature rather than discarded, since missingness
    itself may correlate with booking behavior.
    """
    df = df.copy()
    implausible = df["age"].notna() & (
        (df["age"] < MIN_PLAUSIBLE_AGE) | (df["age"] > MAX_PLAUSIBLE_AGE)
    )
    df["age_missing_or_invalid"] = (df["age"].isna() | implausible).astype(int)
    df.loc[implausible, "age"] = np.nan
    return df


def flag_gender_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Flag the 'UNKNOWN' gender placeholder as missingness, mirroring clean_age.

    The raw gender column is retained (incl. UNKNOWN) so one-hot encoding can still
    treat it as a distinct level; the 0/1 flag makes the missingness explicit and
    consistent with age_missing_or_invalid.
    """
    df = df.copy()
    df["gender_missing"] = (df["gender"] == GENDER_MISSING_TOKEN).astype(int)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pass applied once, before any train/val/test split."""
    df = clean_age(df)
    df = flag_gender_missing(df)
    # signup_flow is a small set of discrete codes (0/3/12/25), not a
    # meaningful magnitude, so treat it as categorical rather than numeric.
    df["signup_flow"] = df["signup_flow"].astype(str)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Preprocessing shared by every model: median-impute+scale age, one-hot
    encode categoricals, pass the engineered missingness flag through as-is.
    Fit only on training folds (e.g. inside a CV loop) to avoid leakage.
    """
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("passthrough", "passthrough", ENGINEERED_FEATURES),
        ]
    )


def make_splits(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """Stratified 70/15/15 train/val/test split by country_destination."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[TARGET], random_state=random_state
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: Path = PROCESSED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)


def load_splits(out_dir: Path = PROCESSED_DIR):
    dtype = {"signup_flow": str}
    train = pd.read_csv(out_dir / "train.csv", dtype=dtype)
    val = pd.read_csv(out_dir / "val.csv", dtype=dtype)
    test = pd.read_csv(out_dir / "test.csv", dtype=dtype)
    return train, val, test


def get_X_y(df: pd.DataFrame):
    return df[FEATURE_COLS], df[TARGET]
