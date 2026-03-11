"""
ml_model.py
-----------
Machine-learning pipeline for F1 lap-time prediction.
Uses scikit-learn RandomForestRegressor with a simple preprocessing step.

Pipeline
--------
1. Extract lap-level features from FastF1 session
2. Encode categorical features (tire compound)
3. Train RandomForestRegressor
4. Evaluate (MAE, RMSE, R²)
5. Persist model with joblib
6. Provide predict_lap_time() inference function
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress minor sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lap_time_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature definition ────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["avg_speed", "max_speed", "min_speed", "avg_throttle",
                    "avg_brake", "avg_gear", "LapNumber"]
CATEGORICAL_FEATURES = ["Compound"]
TARGET = "LapTime_s"

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing target or key features; fill other NaNs."""
    df = df.copy()

    # Standardise compound strings
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].fillna("UNKNOWN").str.upper()
        df.loc[~df["Compound"].isin(COMPOUNDS), "Compound"] = "UNKNOWN"

    df = df.dropna(subset=[TARGET])
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())

    # Remove pit laps and safety car laps (outlier filter: > 5 s slower than median)
    median_lt = df[TARGET].median()
    df = df[df[TARGET] <= median_lt * 1.15]
    df = df[df[TARGET] >= median_lt * 0.92]

    return df.reset_index(drop=True)


def _build_pipeline() -> Pipeline:
    """Construct the sklearn preprocessing + model pipeline."""
    cat_enc = OrdinalEncoder(
        categories=[COMPOUNDS],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", cat_enc, CATEGORICAL_FEATURES),
    ])
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", pre), ("regressor", model)])


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(lap_df: pd.DataFrame) -> dict:
    """
    Train the lap-time prediction model.

    Args:
        lap_df: DataFrame produced by analytics.build_lap_feature_dataset().
                Must contain NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET.

    Returns:
        Dict with keys 'pipeline', 'metrics', 'feature_importance'.
    """
    df = _clean_dataset(lap_df)

    if len(df) < 20:
        raise ValueError(
            f"Too few clean laps for training ({len(df)}). "
            "Load a full race or qualifying session."
        )

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[all_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    # Feature importances from the RF regressor
    rf = pipe.named_steps["regressor"]
    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "pipeline": pipe,
        "metrics": {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)},
        "feature_importance": importance_df,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    """Serialise the trained pipeline to disk using joblib."""
    joblib.dump(pipeline, path)


def load_model(path: str = MODEL_PATH) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Train the model first via the dashboard."
        )
    return joblib.load(path)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_lap_time(
    features: dict,
    pipeline: Pipeline | None = None,
) -> float:
    """
    Predict lap time in seconds for a single set of features.

    Args:
        features: Dict with keys matching NUMERIC_FEATURES + CATEGORICAL_FEATURES.
                  Missing numeric values are filled with 0.
        pipeline: Optional pre-loaded pipeline (avoids disk I/O in hot path).

    Returns:
        Predicted lap time in seconds.
    """
    if pipeline is None:
        pipeline = load_model()

    row = {f: features.get(f, 0.0) for f in NUMERIC_FEATURES}
    row["Compound"] = str(features.get("Compound", "UNKNOWN")).upper()

    df = pd.DataFrame([row])
    return float(pipeline.predict(df)[0])


# ── Convenience: format seconds → mm:ss.ms ───────────────────────────────────

def format_lap_time(seconds: float) -> str:
    """Convert a float number of seconds to 'M:SS.mmm' string."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:06.3f}"
