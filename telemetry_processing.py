"""
telemetry_processing.py
-----------------------
Preprocessing and alignment utilities for F1 telemetry time-series data.
Provides distance normalisation, multi-lap alignment, and segment decomposition.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Safe numeric column detection ────────────────────────────────────────────

def _safe_numeric_cols(telemetry: pd.DataFrame) -> list[str]:
    """
    Return column names that can be safely cast to float64 for interpolation.

    Explicitly excludes timedelta64 (FastF1 'Time', 'SessionTime') and any
    other non-float-castable dtype, which would otherwise trigger:
        "Cannot cast array data from dtype('<m8[ns]') to dtype('float64')"
    """
    safe = []
    for col in telemetry.columns:
        if col == "Distance":
            continue
        dtype = telemetry[col].dtype
        # Skip timedelta, datetime, object, and category columns
        if pd.api.types.is_timedelta64_dtype(dtype):
            continue
        if pd.api.types.is_datetime64_any_dtype(dtype):
            continue
        if dtype == object or str(dtype) == "category":
            continue
        # Only keep columns that numpy can interpolate (int / float / bool)
        try:
            telemetry[col].values.astype(np.float64)
            safe.append(col)
        except (ValueError, TypeError):
            continue
    return safe


# ── Distance normalisation ────────────────────────────────────────────────────

def normalise_distance(telemetry: pd.DataFrame, n_points: int = 1000) -> pd.DataFrame:
    """
    Resample telemetry onto a uniform distance grid.

    Args:
        telemetry: Raw telemetry DataFrame (must contain 'Distance' column).
        n_points:  Number of evenly-spaced distance samples.

    Returns:
        New DataFrame indexed 0..n_points-1 with all safely numeric channels
        interpolated onto the common grid.
    """
    if telemetry.empty or "Distance" not in telemetry.columns:
        return telemetry

    dist_min = float(telemetry["Distance"].min())
    dist_max = float(telemetry["Distance"].max())
    grid = np.linspace(dist_min, dist_max, n_points)

    # Use only columns that can be safely cast to float64
    interp_cols = _safe_numeric_cols(telemetry)

    resampled = {"Distance": grid}
    dist_src = telemetry["Distance"].values.astype(np.float64)
    for col in interp_cols:
        values = telemetry[col].values.astype(np.float64)
        resampled[col] = np.interp(grid, dist_src, values)

    return pd.DataFrame(resampled)


# ── Telemetry alignment ───────────────────────────────────────────────────────

def align_telemetry(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    n_points: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two telemetry DataFrames onto the same distance grid so that
    channel-wise comparisons are valid.

    Args:
        tel_a:    Telemetry for driver / lap A.
        tel_b:    Telemetry for driver / lap B.
        n_points: Resampling resolution.

    Returns:
        Tuple (tel_a_aligned, tel_b_aligned) on matching distance vectors.
    """
    tel_a_norm = normalise_distance(tel_a, n_points)
    tel_b_norm = normalise_distance(tel_b, n_points)

    # Use shorter lap's distance range to avoid extrapolation artifacts
    dist_max = min(tel_a_norm["Distance"].max(), tel_b_norm["Distance"].max())
    mask_a = tel_a_norm["Distance"] <= dist_max
    mask_b = tel_b_norm["Distance"] <= dist_max

    return tel_a_norm[mask_a].reset_index(drop=True), tel_b_norm[mask_b].reset_index(drop=True)


# ── Segment decomposition ─────────────────────────────────────────────────────

def segment_telemetry(
    telemetry: pd.DataFrame,
    segment_length_m: float = 200.0,
) -> pd.DataFrame:
    """
    Decompose a lap's telemetry into fixed-distance segments and compute
    mean performance metrics per segment.

    Args:
        telemetry:         Normalised telemetry DataFrame.
        segment_length_m:  Bin width in metres (default 200 m).

    Returns:
        DataFrame with one row per segment containing:
            segment_id, dist_start, dist_end,
            avg_speed, avg_throttle, avg_brake, avg_gear.
    """
    if telemetry.empty:
        return pd.DataFrame()

    dist_max = telemetry["Distance"].max()
    bins = np.arange(0, dist_max + segment_length_m, segment_length_m)

    records = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (telemetry["Distance"] >= lo) & (telemetry["Distance"] < hi)
        seg = telemetry[mask]
        if seg.empty:
            continue

        row = {
            "segment_id": i,
            "dist_start": lo,
            "dist_end": hi,
            "avg_speed": seg["Speed"].mean() if "Speed" in seg else np.nan,
            "avg_throttle": seg["Throttle"].mean() if "Throttle" in seg else np.nan,
            "avg_brake": seg["Brake"].mean() if "Brake" in seg else np.nan,
            "avg_gear": seg["nGear"].mean() if "nGear" in seg else np.nan,
        }
        records.append(row)

    return pd.DataFrame(records)


# ── Lap-time delta ────────────────────────────────────────────────────────────

def compute_lap_delta(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate cumulative lap-time delta between two aligned telemetry frames.
    Positive delta means driver A is ahead (faster) at that point on track.

    Args:
        tel_a: Aligned telemetry for driver A (reference).
        tel_b: Aligned telemetry for driver B.

    Returns:
        DataFrame with 'Distance' and 'Delta' (seconds) columns.
    """
    if tel_a.empty or tel_b.empty:
        return pd.DataFrame({"Distance": [], "Delta": []})

    # Compute time-at-distance for both via distance/speed integration
    def time_at_distance(tel: pd.DataFrame) -> np.ndarray:
        speed_ms = np.maximum(tel["Speed"].values, 1.0) / 3.6  # avoid division by zero
        dist = tel["Distance"].values
        dd = np.diff(dist, prepend=dist[0])
        dt = dd / speed_ms  # seconds per sample
        return np.cumsum(dt)

    t_a = time_at_distance(tel_a)
    t_b = time_at_distance(tel_b)

    # Use shorter array length
    n = min(len(t_a), len(t_b))
    delta = t_b[:n] - t_a[:n]

    return pd.DataFrame({
        "Distance": tel_a["Distance"].values[:n],
        "Delta": delta,
    })


# ── Convenience: extract scalar lap features ─────────────────────────────────

def extract_lap_features(telemetry: pd.DataFrame) -> dict:
    """
    Compute scalar summary statistics from a lap's telemetry.
    Used as input features for the ML model.

    Returns:
        Dict with keys: avg_speed, max_speed, avg_throttle, avg_brake,
                        avg_gear, min_speed (cornering proxy).
    """
    feats = {}
    if telemetry.empty:
        return feats

    feats["avg_speed"] = telemetry["Speed"].mean()
    feats["max_speed"] = telemetry["Speed"].max()
    feats["min_speed"] = telemetry["Speed"].min()
    feats["avg_throttle"] = telemetry["Throttle"].mean() if "Throttle" in telemetry else np.nan
    feats["avg_brake"] = telemetry["Brake"].mean() if "Brake" in telemetry else 0.0
    feats["avg_gear"] = telemetry["nGear"].mean() if "nGear" in telemetry else np.nan
    return feats
