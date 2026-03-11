# 🏎️ F1 Telemetry Analytics Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastF1](https://img.shields.io/badge/FastF1-3.3.9-FF1801?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.22-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade Formula 1 telemetry analytics platform that simulates real-world race engineering workflows.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Docker](#-docker) • [ML Model](#-machine-learning-model) • [API Reference](#-module-api-reference) • [Screenshots](#-dashboard-preview)

</div>

---

## 📖 Overview

This platform replicates the kind of **data-driven tooling used by Formula 1 race engineers** to analyse driver performance, compare telemetry, evaluate tire strategy, and forecast lap times. It ingests historical F1 data through the [FastF1](https://theoehrly.github.io/Fast-F1/) library and surfaces insights through an interactive Streamlit dashboard with Plotly visualisations.

Whether you're a motorsport enthusiast, a data engineer, or a machine learning practitioner, this project demonstrates how to build a real end-to-end analytics product from raw time-series sensor data through to a trained predictive model.

---

## ✨ Features

### 🔬 Telemetry Analysis
- **Speed, Throttle & Brake traces** — compare two drivers across the full lap distance
- **3-panel stacked overview** — speed / throttle / brake aligned on a single distance axis
- **Distance normalisation** — telemetry resampled to a common 1 000-point grid for valid comparisons

### ⚡ Race Engineering Insights
- **Lap delta analysis** — cumulative time gap between two drivers mapped across the track
- **Sector time comparison** — colour-coded sector breakdown with split deltas
- **200 m track segment decomposition** — identify where each driver gains or loses time

### 🗺️ Track Visualisation
- **Track map** — XY coordinate layout coloured by Speed, Gear, or Throttle
- **Side-by-side driver maps** — instantly spot where each driver is on the limit

### 🔄 Strategy Analysis
- **Tire strategy chart** — race pace plotted lap-by-lap, colour-coded by compound (SOFT / MEDIUM / HARD / WET / INTER)
- **Stint identification** — visual stints overlaid for both drivers simultaneously

### 🤖 Machine Learning
- **Lap time prediction** — RandomForestRegressor trained on all laps in a session
- **Feature importance** — understand which inputs drive lap time most
- **Interactive inference UI** — enter custom lap parameters, get a predicted time instantly
- **Model persistence** — trained model saved to `models/lap_time_model.pkl` via joblib

---

## 🏗️ Architecture

```
motorsport-analytics/
│
├── app.py                    ← Streamlit dashboard (entry point)
├── data_loader.py            ← FastF1 session & telemetry retrieval
├── telemetry_processing.py   ← Signal processing & alignment
├── analytics.py              ← Race engineering analytics engine
├── visualizations.py         ← Plotly chart factory (8 chart types)
├── ml_model.py               ← scikit-learn ML pipeline
│
├── models/
│   └── lap_time_model.pkl    ← Serialised model (generated at runtime)
│
├── cache/                    ← FastF1 data cache (auto-managed)
│
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

### Data Flow

```
User selects Year / Race / Session / Drivers
             │
             ▼
    FastF1 loads session data
    (cached to disk on first load)
             │
             ▼
  Telemetry extracted per lap
  Distance-normalised → 1 000-point grid
             │
        ┌────┴────┐
        ▼         ▼
  Analytics     ML Model
  • Speed traces   • Feature extraction
  • Lap delta      • RandomForest training
  • Sectors        • Lap time prediction
  • Tire strategy
  • Segments
        │         │
        └────┬────┘
             ▼
     Plotly interactive charts
             │
             ▼
    Streamlit renders dashboard
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip
- Internet connection (FastF1 fetches from the F1 timing API on first load)

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/motorsport-analytics.git
cd motorsport-analytics
```

### 2 — Create a virtual environment

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Launch the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker

### Build the image

```bash
docker build -t motorsport-analytics .
```

### Run the container

```bash
docker run -p 8501:8501 \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/models:/app/models \
  motorsport-analytics
```

> Mounting `cache/` and `models/` as volumes means FastF1 data and trained models **persist between container restarts**, avoiding repeated downloads.


---

## 🎮 Using the Dashboard

### Step 1 — Load a session

Use the **sidebar** on the left:

| Control | Options |
|---|---|
| Year | 2018 – 2024 |
| Race | Bahrain, Monaco, Silverstone… (any FastF1-supported round) |
| Session Type | FP1, FP2, FP3, Q (Qualifying), R (Race) |
| Driver A | Three-letter code e.g. `VER` |
| Driver B | Three-letter code e.g. `LEC` |

Click **🔄 Load Session**. FastF1 will download and cache the data (first load: ~20–60 s depending on session size; subsequent loads: instant from cache).

### Step 2 — Explore the dashboard sections

| Section | What it shows |
|---|---|
| ⏱️ **Fastest Lap Summary** | Lap times for both drivers + absolute delta |
| 📊 **Telemetry Overview** | Stacked Speed / Throttle / Brake for the full lap |
| 🔍 **Detailed Traces** | Individual speed, throttle and brake charts |
| ⚡ **Lap Delta** | Cumulative time gap — positive = Driver A is ahead |
| 🏁 **Sector Times** | Sector 1/2/3 breakdown with colour-coded deltas |
| 🗺️ **Track Map** | XY layout coloured by Speed, Gear, or Throttle |
| 🔄 **Tire Strategy** | Lap-by-lap pace coloured by tire compound |
| 📐 **Segment Performance** | 200 m segment average speed + best/worst tables |
| 🤖 **ML Prediction** | Train model, view metrics, predict custom lap times |

### Step 3 — Train the ML model

1. Load any session (Race sessions give the most laps and the best model).
2. Click **Train Lap Time Model** in the sidebar.
3. Training runs in ~30–90 seconds.
4. MAE, RMSE, R² metrics and feature importances are displayed inline.
5. The model is saved to `models/lap_time_model.pkl`.

### Step 4 — Predict a lap time

1. Scroll to **🔮 Predict a Lap Time** at the bottom of the dashboard.
2. Expand the panel and enter your parameters.
3. Click **🔮 Predict Lap Time** — the predicted time appears in `M:SS.mmm` format.

---

## 🧠 Machine Learning Model

### Problem statement

> Given aggregated telemetry statistics for a single lap, predict the lap time in seconds.

### Feature engineering

Each lap in the session becomes one training sample. Features are derived from the raw telemetry channel data:

| Feature | Description |
|---|---|
| `avg_speed` | Mean speed across the lap (km/h) |
| `max_speed` | Top speed achieved (km/h) |
| `min_speed` | Lowest speed — proxy for cornering pace (km/h) |
| `avg_throttle` | Mean throttle application (%) |
| `avg_brake` | Mean brake application (0–1) |
| `avg_gear` | Mean gear — proxy for high-speed vs technical circuit |
| `LapNumber` | Lap count (captures track evolution and fuel load) |
| `Compound` | Tire compound — ordinal-encoded (SOFT / MEDIUM / HARD / …) |

### Pipeline

```
Raw laps
   │
   ▼
extract_lap_features()  ← telemetry_processing.py
   │
   ▼
Outlier filtering       ← remove pit laps, safety car laps (±15% of median)
   │
   ▼
ColumnTransformer
  ├─ Numeric features  → passthrough
  └─ Compound          → OrdinalEncoder
   │
   ▼
RandomForestRegressor(n_estimators=200, max_depth=12)
   │
   ▼
Evaluation: MAE / RMSE / R²
   │
   ▼
joblib.dump → models/lap_time_model.pkl
```

### Typical performance (2023 Bahrain Race)

| Metric | Value |
|---|---|
| MAE | ~0.35 s |
| RMSE | ~0.52 s |
| R² | ~0.91 |

---



## 🗂️ Cache & Storage

| Path | Contents | Notes |
|---|---|---|
| `cache/` | FastF1 session data | Auto-managed; can grow to several GB for a full season |
| `models/lap_time_model.pkl` | Trained RandomForest pipeline | Generated by clicking "Train Model" in the dashboard |

To clear the cache:
```bash
rm -rf cache/*
```

To delete the trained model:
```bash
rm models/lap_time_model.pkl
```

---

## ⚠️ Known Limitations

- **Internet required on first load** — FastF1 fetches from the official F1 timing API. Subsequent loads use the local disk cache.
- **Tire strategy is richest in Race sessions** — FP and Qualifying sessions have fewer laps and less compound variety.
- **Sector times may be missing** for some older seasons (pre-2018) or certain session types.
- **Track coordinates (X/Y)** are session-dependent; if unavailable, the track map shows a placeholder message.
- **ML accuracy scales with data volume** — a full 70-lap race session trains a significantly better model than a short FP session.

---


## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

F1 data is sourced from the official Formula 1 timing feed via the [FastF1](https://github.com/theOehrly/Fast-F1) library. This project is not affiliated with or endorsed by Formula 1, the FIA, or any F1 team.

---


<div align="center">
  <sub>Wasn't it simply lovely?❤️  </sub>
</div>
