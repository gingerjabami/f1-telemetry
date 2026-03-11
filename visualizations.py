"""
visualizations.py
-----------------
Reusable Plotly chart factories for the motorsport analytics dashboard.
All functions return go.Figure objects for Streamlit rendering.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Colour palette ────────────────────────────────────────────────────────────
_DRIVER_COLOURS = {
    "A": "#E8002D",   # red
    "B": "#00D2BE",   # teal
}
_COMPOUND_COLOURS = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
    "UNKNOWN": "#888888",
}


def _driver_color(driver_label: str) -> str:
    """Map 'A'/'B' or arbitrary string to a hex colour."""
    return _DRIVER_COLOURS.get(driver_label, "#AAAAAA")


# ── 1. Speed trace ────────────────────────────────────────────────────────────

def speed_trace(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure:
    """Distance vs Speed comparison for two drivers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tel_a["Distance"], y=tel_a["Speed"],
        mode="lines", name=driver_a,
        line=dict(color="#E8002D", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=tel_b["Distance"], y=tel_b["Speed"],
        mode="lines", name=driver_b,
        line=dict(color="#00D2BE", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Speed Trace",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ── 2. Throttle trace ─────────────────────────────────────────────────────────

def throttle_trace(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure:
    """Distance vs Throttle (%) comparison."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tel_a["Distance"], y=tel_a["Throttle"],
        mode="lines", name=driver_a,
        line=dict(color="#E8002D", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=tel_b["Distance"], y=tel_b["Throttle"],
        mode="lines", name=driver_b,
        line=dict(color="#00D2BE", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Throttle Application",
        xaxis_title="Distance (m)",
        yaxis_title="Throttle (%)",
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ── 3. Brake trace ────────────────────────────────────────────────────────────

def brake_trace(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure:
    """Distance vs Brake (0/1) comparison."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tel_a["Distance"], y=tel_a["Brake"],
        mode="lines", name=driver_a, fill="tozeroy",
        line=dict(color="#E8002D", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=tel_b["Distance"], y=tel_b["Brake"],
        mode="lines", name=driver_b, fill="tozeroy",
        line=dict(color="#00D2BE", width=1),
        opacity=0.6,
    ))
    fig.update_layout(
        title="Brake Application",
        xaxis_title="Distance (m)",
        yaxis_title="Brake",
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ── 4. Lap delta plot ─────────────────────────────────────────────────────────

def lap_delta_plot(
    delta_df: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure:
    """
    Distance vs cumulative time delta.
    Above zero-line → driver A is ahead.
    """
    fig = go.Figure()

    # Shaded region
    fig.add_trace(go.Scatter(
        x=delta_df["Distance"], y=delta_df["Delta"],
        mode="lines", name=f"{driver_a} vs {driver_b}",
        line=dict(color="#FFD700", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,215,0,0.15)",
    ))
    fig.add_hline(y=0, line_color="white", line_dash="dot", line_width=1)

    fig.update_layout(
        title=f"Lap Delta: {driver_a} (ref) vs {driver_b}",
        xaxis_title="Distance (m)",
        yaxis_title="Δ Time (s)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ── 5. Track map telemetry ────────────────────────────────────────────────────

def track_map(
    telemetry: pd.DataFrame,
    colour_by: str = "Speed",
    driver: str = "",
) -> go.Figure:
    """
    Plot the track layout (X/Y coordinates) coloured by a telemetry channel.

    Args:
        telemetry:  Normalised telemetry with X, Y columns.
        colour_by:  Column name to use for colour scale ('Speed', 'nGear', etc.)
        driver:     Label for the chart title.
    """
    if "X" not in telemetry.columns or "Y" not in telemetry.columns:
        fig = go.Figure()
        fig.add_annotation(text="Track coordinates unavailable", showarrow=False)
        return fig

    z = telemetry[colour_by] if colour_by in telemetry.columns else telemetry["Speed"]

    fig = go.Figure(go.Scatter(
        x=telemetry["X"],
        y=telemetry["Y"],
        mode="markers",
        marker=dict(
            color=z,
            colorscale="RdYlGn",
            size=4,
            colorbar=dict(title=colour_by),
            showscale=True,
        ),
        hovertemplate=f"Distance: %{{text}}<br>{colour_by}: %{{marker.color:.1f}}<extra></extra>",
        text=telemetry["Distance"].round(0).astype(int).astype(str),
    ))
    fig.update_layout(
        title=f"Track Map – {driver} ({colour_by})",
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ── 6. Tire strategy ──────────────────────────────────────────────────────────

def tire_strategy_chart(
    strategy_df: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure:
    """
    Lap-by-lap pace coloured by compound for two drivers.

    Args:
        strategy_df: DataFrame with columns Driver, LapNumber, LapTime_s, Compound.
    """
    if strategy_df.empty:
        return go.Figure()

    fig = go.Figure()

    compounds = strategy_df["Compound"].dropna().unique() if "Compound" in strategy_df else []

    for driver in [driver_a, driver_b]:
        df = strategy_df[strategy_df["Driver"] == driver]
        if df.empty:
            continue

        # Plot each compound stint as separate trace for colour coding
        for compound in df["Compound"].unique() if "Compound" in df else []:
            mask = df["Compound"] == compound
            colour = _COMPOUND_COLOURS.get(str(compound).upper(), "#888")
            fig.add_trace(go.Scatter(
                x=df[mask]["LapNumber"],
                y=df[mask]["LapTime_s"],
                mode="markers+lines",
                name=f"{driver} – {compound}",
                marker=dict(color=colour, size=8,
                            symbol="circle" if driver == driver_a else "diamond"),
                line=dict(color=colour, width=2,
                          dash="solid" if driver == driver_a else "dash"),
            ))

    fig.update_layout(
        title="Race Pace & Tire Strategy",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


# ── 7. Segment performance bar chart ─────────────────────────────────────────

def segment_bar_chart(
    segment_df: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure:
    """
    Bar chart of average speed per track segment for two drivers.

    Args:
        segment_df: Output of analytics.segment_performance_comparison().
    """
    if segment_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=segment_df["dist_start"],
        y=segment_df["avg_speed_A"],
        name=driver_a,
        marker_color="#E8002D",
        width=segment_df["dist_end"].iloc[0] - segment_df["dist_start"].iloc[0] if len(segment_df) else 200,
    ))
    fig.add_trace(go.Bar(
        x=segment_df["dist_start"],
        y=segment_df["avg_speed_B"],
        name=driver_b,
        marker_color="#00D2BE",
        width=segment_df["dist_end"].iloc[0] - segment_df["dist_start"].iloc[0] if len(segment_df) else 200,
        opacity=0.8,
    ))
    fig.update_layout(
        title="Average Speed per Track Segment",
        xaxis_title="Distance from start (m)",
        yaxis_title="Avg Speed (km/h)",
        barmode="overlay",
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ── 8. Combined telemetry overview (speed + throttle + brake stacked) ─────────

def telemetry_overview(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure:
    """Three-panel stacked plot: Speed / Throttle / Brake."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Speed (km/h)", "Throttle (%)", "Brake"),
        row_heights=[0.45, 0.3, 0.25],
        vertical_spacing=0.08,
    )

    # Speed
    fig.add_trace(go.Scatter(x=tel_a["Distance"], y=tel_a["Speed"],
                             name=driver_a, line=dict(color="#E8002D")), row=1, col=1)
    fig.add_trace(go.Scatter(x=tel_b["Distance"], y=tel_b["Speed"],
                             name=driver_b, line=dict(color="#00D2BE", dash="dash")), row=1, col=1)

    # Throttle
    fig.add_trace(go.Scatter(x=tel_a["Distance"], y=tel_a["Throttle"],
                             name=driver_a, line=dict(color="#E8002D"),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=tel_b["Distance"], y=tel_b["Throttle"],
                             name=driver_b, line=dict(color="#00D2BE", dash="dash"),
                             showlegend=False), row=2, col=1)

    # Brake
    fig.add_trace(go.Scatter(x=tel_a["Distance"], y=tel_a["Brake"],
                             name=driver_a, line=dict(color="#E8002D"),
                             showlegend=False, fill="tozeroy"), row=3, col=1)
    fig.add_trace(go.Scatter(x=tel_b["Distance"], y=tel_b["Brake"],
                             name=driver_b, line=dict(color="#00D2BE", dash="dash"),
                             showlegend=False, fill="tozeroy", opacity=0.6), row=3, col=1)

    fig.update_xaxes(title_text="Distance (m)", row=3, col=1)
    fig.update_layout(
        title=f"Telemetry Overview: {driver_a} vs {driver_b}",
        template="plotly_dark",
        height=700,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=80, b=40),
    )
    return fig
