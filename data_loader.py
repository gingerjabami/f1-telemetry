"""
data_loader.py
--------------
Responsible for retrieving and caching Formula 1 session data via FastF1.
Provides clean interfaces for session loading, lap extraction, and telemetry access.
"""

import os
import fastf1
import pandas as pd
import numpy as np
from typing import Optional

# ── Cache configuration ──────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── Session loading ───────────────────────────────────────────────────────────

def load_session(year: int, race: str, session_type: str) -> fastf1.core.Session:
    """
    Load a FastF1 session with telemetry data.

    Args:
        year:         Championship year (e.g. 2023)
        race:         Race name or round number (e.g. 'Bahrain', 1)
        session_type: 'FP1','FP2','FP3','Q','SQ','R','SS'

    Returns:
        Loaded FastF1 Session object.
    """
    session = fastf1.get_session(year, race, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    return session


# ── Driver helpers ────────────────────────────────────────────────────────────

def get_driver_laps(session: fastf1.core.Session, driver: str) -> fastf1.core.Laps:
    """
    Return all recorded laps for a driver in the session.

    Args:
        session: Loaded FastF1 Session.
        driver:  Three-letter driver code (e.g. 'VER', 'HAM').

    Returns:
        FastF1 Laps object filtered to that driver.
    """
    laps = session.laps.pick_driver(driver)
    return laps


def get_fastest_lap(session: fastf1.core.Session, driver: str) -> fastf1.core.Lap:
    """
    Return the single fastest valid lap for a driver.

    Args:
        session: Loaded FastF1 Session.
        driver:  Three-letter driver code.

    Returns:
        FastF1 Lap (a single-row Laps subset).
    """
    laps = get_driver_laps(session, driver)
    fastest = laps.pick_fastest()
    return fastest


def extract_telemetry(lap: fastf1.core.Lap) -> pd.DataFrame:
    """
    Extract raw telemetry for a single lap and add a cumulative Distance column.

    Sanitisation applied here:
    - Brake column cast to float (FastF1 may return bool or int8)
    - Timedelta columns (Time, SessionTime) are kept for reference but will be
      excluded from numeric interpolation by telemetry_processing._safe_numeric_cols

    Args:
        lap: FastF1 Lap object.

    Returns:
        DataFrame with columns: Time, Speed, Throttle, Brake, nGear,
        RPM, DRS, X, Y, Z, Distance.
    """
    tel = lap.get_telemetry()

    # Ensure Distance column exists (FastF1 usually provides it)
    if "Distance" not in tel.columns:
        # Approximate distance from speed × dt
        dt = tel["Time"].diff().dt.total_seconds().fillna(0)
        tel["Distance"] = (tel["Speed"] * dt / 3.6).cumsum()

    # Normalise Brake to float 0/1 regardless of incoming dtype
    # (FastF1 can return bool, int8, or float depending on version)
    if "Brake" in tel.columns:
        tel["Brake"] = tel["Brake"].astype(np.float64)

    # Ensure Speed and Throttle are plain float64
    for col in ("Speed", "Throttle", "RPM"):
        if col in tel.columns:
            tel[col] = pd.to_numeric(tel[col], errors="coerce").astype(np.float64)

    # Ensure Distance is float64
    tel["Distance"] = tel["Distance"].astype(np.float64)

    return tel.reset_index(drop=True)


# ── Session metadata helpers ──────────────────────────────────────────────────

def get_available_drivers(session: fastf1.core.Session) -> list[str]:
    """Return sorted list of driver codes present in the session."""
    return sorted(session.laps["Driver"].unique().tolist())


def get_compound_map(session: fastf1.core.Session, driver: str) -> pd.DataFrame:
    """
    Return per-lap compound information for a driver.

    Returns DataFrame with columns: LapNumber, Compound, TyreLife, FreshTyre.
    """
    laps = get_driver_laps(session, driver)
    cols = ["LapNumber", "Compound", "TyreLife", "FreshTyre", "LapTime"]
    available = [c for c in cols if c in laps.columns]
    return laps[available].copy()
