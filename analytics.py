"""
analytics.py
------------
Race-engineering analytics: driver comparison, lap deltas, sector analysis,
tire strategy, and segment-level performance decomposition.
"""

import numpy as np
import pandas as pd
import fastf1
import fastf1.utils

from data_loader import get_driver_laps, get_fastest_lap, extract_telemetry
from telemetry_processing import (
    align_telemetry,
    segment_telemetry,
    compute_lap_delta,
    extract_lap_features,
)


# ── 1. Driver telemetry comparison ────────────────────────────────────────────

def compare_driver_telemetry(
    session: fastf1.core.Session,
    driver_a: str,
    driver_b: str,
    n_points: int = 1000,
) -> dict:
    """
    Retrieve and align fastest-lap telemetry for two drivers.

    Returns:
        Dict with keys 'tel_a', 'tel_b', 'driver_a', 'driver_b'.
    """
    lap_a = get_fastest_lap(session, driver_a)
    lap_b = get_fastest_lap(session, driver_b)

    raw_a = extract_telemetry(lap_a)
    raw_b = extract_telemetry(lap_b)

    tel_a, tel_b = align_telemetry(raw_a, raw_b, n_points)

    return {
        "tel_a": tel_a,
        "tel_b": tel_b,
        "driver_a": driver_a,
        "driver_b": driver_b,
        "lap_a": lap_a,
        "lap_b": lap_b,
    }


# ── 2. Lap delta analysis ─────────────────────────────────────────────────────

def lap_delta_analysis(
    session: fastf1.core.Session,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """
    Compute the cumulative time delta between two drivers' fastest laps.

    Returns:
        DataFrame with 'Distance' and 'Delta' columns.
        Positive delta → driver B is slower (driver A has an advantage).
    """
    lap_a = get_fastest_lap(session, driver_a)
    lap_b = get_fastest_lap(session, driver_b)

    raw_a = extract_telemetry(lap_a)
    raw_b = extract_telemetry(lap_b)

    tel_a, tel_b = align_telemetry(raw_a, raw_b)
    return compute_lap_delta(tel_a, tel_b)


# ── 3. Sector performance comparison ─────────────────────────────────────────

def sector_comparison(
    session: fastf1.core.Session,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """
    Compare sector times for both drivers' fastest laps.

    Returns:
        DataFrame with columns: Sector, DriverA_time, DriverB_time, Delta (s).
    """
    lap_a = get_fastest_lap(session, driver_a)
    lap_b = get_fastest_lap(session, driver_b)

    sectors = ["Sector1Time", "Sector2Time", "Sector3Time"]
    rows = []
    for s in sectors:
        t_a = lap_a[s].total_seconds() if pd.notna(lap_a[s]) else np.nan
        t_b = lap_b[s].total_seconds() if pd.notna(lap_b[s]) else np.nan
        rows.append({
            "Sector": s.replace("Time", ""),
            f"{driver_a}_time": t_a,
            f"{driver_b}_time": t_b,
            "Delta_s": t_b - t_a,
        })

    return pd.DataFrame(rows)


# ── 4. Tire strategy analysis ─────────────────────────────────────────────────

def tire_strategy_analysis(
    session: fastf1.core.Session,
    driver: str,
) -> pd.DataFrame:
    """
    Build a per-lap tire strategy table for a driver.

    Returns:
        DataFrame with LapNumber, Compound, TyreLife, LapTime_s columns.
    """
    laps = get_driver_laps(session, driver)

    cols = ["LapNumber", "Compound", "TyreLife", "LapTime"]
    available = [c for c in cols if c in laps.columns]
    df = laps[available].copy()

    if "LapTime" in df.columns:
        df["LapTime_s"] = df["LapTime"].dt.total_seconds()
        df = df.drop(columns=["LapTime"])

    return df.dropna(subset=["LapNumber"]).reset_index(drop=True)


def tire_strategy_both_drivers(
    session: fastf1.core.Session,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """
    Combine tire strategy for two drivers (for overlay plots).
    Adds a 'Driver' column.
    """
    df_a = tire_strategy_analysis(session, driver_a)
    df_b = tire_strategy_analysis(session, driver_b)
    df_a["Driver"] = driver_a
    df_b["Driver"] = driver_b
    return pd.concat([df_a, df_b], ignore_index=True)


# ── 5. Segment-level performance decomposition ────────────────────────────────

def segment_performance_comparison(
    session: fastf1.core.Session,
    driver_a: str,
    driver_b: str,
    segment_length_m: float = 200.0,
) -> pd.DataFrame:
    """
    Compare average speed per track segment between two drivers.

    Returns:
        DataFrame with segment_id, dist_start, dist_end,
        avg_speed_A, avg_speed_B, speed_diff (A − B).
    """
    lap_a = get_fastest_lap(session, driver_a)
    lap_b = get_fastest_lap(session, driver_b)

    raw_a = extract_telemetry(lap_a)
    raw_b = extract_telemetry(lap_b)

    tel_a, tel_b = align_telemetry(raw_a, raw_b)

    seg_a = segment_telemetry(tel_a, segment_length_m).set_index("segment_id")
    seg_b = segment_telemetry(tel_b, segment_length_m).set_index("segment_id")

    merged = seg_a[["dist_start", "dist_end", "avg_speed"]].join(
        seg_b[["avg_speed"]], lsuffix="_A", rsuffix="_B", how="inner"
    )
    merged["speed_diff"] = merged["avg_speed_A"] - merged["avg_speed_B"]
    return merged.reset_index()


# ── Lap feature dataset builder (for ML) ──────────────────────────────────────
from typing import Optional, List
def build_lap_feature_dataset(
    session: fastf1.core.Session,
    drivers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Iterate over all laps in the session and extract ML-ready features.

    Args:
        session: Loaded FastF1 Session.
        drivers: Optional list of driver codes to include (None = all).

    Returns:
        DataFrame suitable for training the lap-time prediction model.
    """
    if drivers is None:
        drivers = session.laps["Driver"].unique().tolist()

    records = []
    for drv in drivers:
        laps = get_driver_laps(session, drv)
        for _, lap in laps.iterrows():
            if pd.isna(lap.get("LapTime")):
                continue
            try:
                tel = extract_telemetry(lap)  # type: ignore[arg-type]
            except Exception:
                continue

            feats = extract_lap_features(tel)
            feats["LapTime_s"] = lap["LapTime"].total_seconds()
            feats["LapNumber"] = lap.get("LapNumber", np.nan)
            feats["Compound"] = lap.get("Compound", "UNKNOWN")
            feats["Driver"] = drv
            records.append(feats)

    return pd.DataFrame(records)
