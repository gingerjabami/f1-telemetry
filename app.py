"""
app.py
------
Motorsport Telemetry Analytics Platform – Streamlit Dashboard.

Run locally:
    streamlit run app.py

Docker:
    docker build -t motorsport-analytics .
    docker run -p 8501:8501 motorsport-analytics
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_session, get_available_drivers
from analytics import (
    compare_driver_telemetry,
    lap_delta_analysis,
    sector_comparison,
    tire_strategy_both_drivers,
    segment_performance_comparison,
    build_lap_feature_dataset,
)
from visualizations import (
    speed_trace,
    throttle_trace,
    brake_trace,
    lap_delta_plot,
    track_map,
    tire_strategy_chart,
    segment_bar_chart,
    telemetry_overview,
)
from ml_model import (
    train_model,
    save_model,
    load_model,
    predict_lap_time,
    format_lap_time,
    MODEL_PATH,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Telemetry Analytics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom dark-themed CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d0d0d; color: #f0f0f0; }
    [data-testid="stSidebar"]          { background: #1a1a1a; }
    h1, h2, h3 { color: #FFD700; }
    .metric-card {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stButton > button {
        background: #E8002D;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover { background: #c0001f; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏎️  F1 Telemetry Analytics")
st.markdown("*Race engineering insights powered by FastF1; engineered by Aditi*")
st.divider()

# ── Sidebar: session selection ────────────────────────────────────────────────
with st.sidebar:
    st.header("📡 Session Selection")

    year = st.selectbox("Year", list(range(2024, 2017, -1)), index=0)

    # Pre-populated race list (user can type any valid FastF1 race name)
    races_2024 = [
        "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
        "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
        "Austria", "Great Britain", "Hungary", "Belgium",
        "Netherlands", "Italy", "Azerbaijan", "Singapore",
        "United States", "Mexico City", "São Paulo",
        "Las Vegas", "Qatar", "Abu Dhabi",
    ]
    race = st.selectbox("Race", races_2024, index=0)

    session_type = st.selectbox(
        "Session Type",
        ["FP1", "FP2", "FP3", "Q", "R"],
        index=3,  # default to Qualifying
    )

    load_btn = st.button("🔄  Load Session", use_container_width=True)

    st.divider()
    st.header("👤 Drivers")
    driver_a = st.text_input("Driver A", value="VER").upper()
    driver_b = st.text_input("Driver B", value="LEC").upper()

    st.divider()
    st.header("🗺️ Track Map")
    colour_by = st.selectbox("Colour by", ["Speed", "nGear", "Throttle"], index=0)

    st.divider()
    st.header("🤖 Machine Learning")
    train_ml_btn = st.button("Train Lap Time Model", use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session" not in st.session_state:
    st.session_state.session = None
if "tel_data" not in st.session_state:
    st.session_state.tel_data = None

# ── Load session ──────────────────────────────────────────────────────────────
if load_btn:
    with st.spinner(f"Loading {year} {race} – {session_type} …"):
        try:
            session = load_session(year, race, session_type)
            st.session_state.session = session
            st.session_state.tel_data = None   # reset on new session
            st.success(f"✅  Loaded {session.event['EventName']} {session_type}")
        except Exception as e:
            st.error(f"Failed to load session: {e}")
            st.stop()

# ── Guard: require loaded session ─────────────────────────────────────────────
if st.session_state.session is None:
    st.info("👈  Select a session in the sidebar and click **Load Session** to begin.")
    st.stop()

session = st.session_state.session

# ── Validate driver codes ─────────────────────────────────────────────────────
available_drivers = get_available_drivers(session)
if driver_a not in available_drivers or driver_b not in available_drivers:
    st.warning(
        f"Drivers available in this session: {', '.join(available_drivers)}\n\n"
        f"You entered: **{driver_a}** and **{driver_b}**. Please update the sidebar."
    )
    st.stop()

# ── Load / cache telemetry data ───────────────────────────────────────────────
@st.cache_data(show_spinner="Processing telemetry …")
def get_tel_data(session_hash: str, drv_a: str, drv_b: str):
    """Cache-wrapped telemetry comparison data."""
    return compare_driver_telemetry(session, drv_a, drv_b)

session_hash = f"{year}_{race}_{session_type}"
try:
    tel_data = get_tel_data(session_hash, driver_a, driver_b)
except Exception as e:
    st.error(f"Telemetry error: {e}")
    st.stop()

tel_a = tel_data["tel_a"]
tel_b = tel_data["tel_b"]
lap_a = tel_data["lap_a"]
lap_b = tel_data["lap_b"]

# ── Section 1: Lap time summary ───────────────────────────────────────────────
st.subheader("⏱️  Fastest Lap Summary")
c1, c2, c3 = st.columns(3)

with c1:
    lt_a = lap_a["LapTime"].total_seconds() if pd.notna(lap_a["LapTime"]) else float("nan")
    st.metric(f"🔴 {driver_a}", format_lap_time(lt_a))

with c2:
    lt_b = lap_b["LapTime"].total_seconds() if pd.notna(lap_b["LapTime"]) else float("nan")
    st.metric(f"🟢 {driver_b}", format_lap_time(lt_b))

with c3:
    delta_total = lt_b - lt_a
    st.metric("Delta", f"{delta_total:+.3f} s",
              delta=f"{driver_a} {'faster' if delta_total > 0 else 'slower'}")

st.divider()

# ── Section 2: Full telemetry overview ───────────────────────────────────────
st.subheader("📊  Telemetry Overview")
fig_overview = telemetry_overview(tel_a, tel_b, driver_a, driver_b)
st.plotly_chart(fig_overview, use_container_width=True)

# ── Section 3: Individual traces ─────────────────────────────────────────────
st.subheader("🔍  Detailed Traces")
col_speed, col_throttle = st.columns(2)
with col_speed:
    st.plotly_chart(speed_trace(tel_a, tel_b, driver_a, driver_b), use_container_width=True)
with col_throttle:
    st.plotly_chart(throttle_trace(tel_a, tel_b, driver_a, driver_b), use_container_width=True)

st.plotly_chart(brake_trace(tel_a, tel_b, driver_a, driver_b), use_container_width=True)

st.divider()

# ── Section 4: Lap delta ──────────────────────────────────────────────────────
st.subheader("⚡  Lap Delta Analysis")
with st.spinner("Computing lap delta …"):
    try:
        delta_df = lap_delta_analysis(session, driver_a, driver_b)
        st.plotly_chart(lap_delta_plot(delta_df, driver_a, driver_b), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute lap delta: {e}")

# ── Section 5: Sector comparison ─────────────────────────────────────────────
st.subheader("🏁  Sector Times")
with st.spinner("Comparing sectors …"):
    try:
        sector_df = sector_comparison(session, driver_a, driver_b)
        st.dataframe(
            sector_df.style.applymap(
                lambda v: "color: #00D2BE" if isinstance(v, float) and v < 0
                else ("color: #E8002D" if isinstance(v, float) and v > 0 else ""),
                subset=["Delta_s"]
            ),
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Sector data unavailable: {e}")

st.divider()

# ── Section 6: Track map ──────────────────────────────────────────────────────
st.subheader("🗺️  Track Telemetry Map")
col_map_a, col_map_b = st.columns(2)
with col_map_a:
    st.plotly_chart(track_map(tel_a, colour_by, driver_a), use_container_width=True)
with col_map_b:
    st.plotly_chart(track_map(tel_b, colour_by, driver_b), use_container_width=True)

st.divider()

# ── Section 7: Tire strategy (race sessions) ──────────────────────────────────
st.subheader("🔄  Tire Strategy & Race Pace")
with st.spinner("Loading tire strategy …"):
    try:
        strategy_df = tire_strategy_both_drivers(session, driver_a, driver_b)
        if not strategy_df.empty and "LapTime_s" in strategy_df.columns:
            st.plotly_chart(
                tire_strategy_chart(strategy_df, driver_a, driver_b),
                use_container_width=True,
            )
        else:
            st.info("Tire strategy data is most informative for Race sessions.")
    except Exception as e:
        st.warning(f"Tire strategy error: {e}")

st.divider()

# ── Section 8: Segment performance ───────────────────────────────────────────
st.subheader("📐  Track Segment Performance")
seg_length = st.slider("Segment length (m)", 100, 500, 200, 50)
with st.spinner("Computing segment performance …"):
    try:
        seg_df = segment_performance_comparison(session, driver_a, driver_b, seg_length)
        st.plotly_chart(segment_bar_chart(seg_df, driver_a, driver_b), use_container_width=True)

        # Top/bottom segments table
        if not seg_df.empty:
            seg_df["speed_diff"] = seg_df["avg_speed_A"] - seg_df["avg_speed_B"]
            top = seg_df.nlargest(5, "speed_diff")[["dist_start", "dist_end", "avg_speed_A", "avg_speed_B", "speed_diff"]]
            bot = seg_df.nsmallest(5, "speed_diff")[["dist_start", "dist_end", "avg_speed_A", "avg_speed_B", "speed_diff"]]

            c_top, c_bot = st.columns(2)
            with c_top:
                st.markdown(f"**{driver_a} fastest segments**")
                st.dataframe(top.round(1), use_container_width=True)
            with c_bot:
                st.markdown(f"**{driver_b} fastest segments**")
                st.dataframe(bot.round(1), use_container_width=True)
    except Exception as e:
        st.warning(f"Segment analysis error: {e}")

st.divider()

# ── Section 9: ML lap time prediction ────────────────────────────────────────
st.subheader("🤖  Lap Time Prediction (ML)")

# Training
if train_ml_btn:
    with st.spinner("Building feature dataset and training model … (this may take ~1–2 min)"):
        try:
            lap_df = build_lap_feature_dataset(session)
            results = train_model(lap_df)
            save_model(results["pipeline"])

            m = results["metrics"]
            st.success("Model trained and saved ✅")

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("MAE", f"{m['MAE']:.3f} s")
            mc2.metric("RMSE", f"{m['RMSE']:.3f} s")
            mc3.metric("R²", f"{m['R2']:.4f}")
            mc4.metric("Training laps", results["n_train"])

            st.bar_chart(
                results["feature_importance"].set_index("feature")["importance"],
            )
        except Exception as e:
            st.error(f"Training failed: {e}")

# Inference UI
st.markdown("#### 🔮 Predict a Lap Time")
with st.expander("Enter lap features"):
    pi_col1, pi_col2, pi_col3 = st.columns(3)
    with pi_col1:
        pi_avg_speed   = st.number_input("Avg Speed (km/h)", 100.0, 300.0, 210.0, 1.0)
        pi_max_speed   = st.number_input("Max Speed (km/h)", 200.0, 400.0, 330.0, 1.0)
        pi_min_speed   = st.number_input("Min Speed (km/h)", 50.0,  200.0,  80.0, 1.0)
    with pi_col2:
        pi_avg_throttle = st.number_input("Avg Throttle (%)", 0.0, 100.0, 65.0, 1.0)
        pi_avg_brake    = st.number_input("Avg Brake (0-1)",  0.0,   1.0,  0.15, 0.01)
        pi_avg_gear     = st.number_input("Avg Gear",         1.0,   8.0,  5.5,  0.1)
    with pi_col3:
        pi_lap_number = st.number_input("Lap Number", 1, 80, 5, 1)
        pi_compound   = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])

    predict_btn = st.button("🔮 Predict Lap Time")

    if predict_btn:
        feat = {
            "avg_speed":    pi_avg_speed,
            "max_speed":    pi_max_speed,
            "min_speed":    pi_min_speed,
            "avg_throttle": pi_avg_throttle,
            "avg_brake":    pi_avg_brake,
            "avg_gear":     pi_avg_gear,
            "LapNumber":    pi_lap_number,
            "Compound":     pi_compound,
        }
        try:
            predicted_s = predict_lap_time(feat)
            st.success(f"Predicted Lap Time: **{format_lap_time(predicted_s)}** ({predicted_s:.3f} s)")
        except FileNotFoundError:
            st.warning("No trained model found. Click **Train Lap Time Model** in the sidebar first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8rem;'>"
    "Motorsport Telemetry Analytics Platform &nbsp;|&nbsp; "
    "Data: FastF1 &nbsp;|&nbsp; Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
