import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="F1 2026 Championship Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CALIBRATED_FILE = DATA_DIR / "monte_carlo_2026_calibrated_results.csv"
MONTE_CARLO_FILE = DATA_DIR / "monte_carlo_2026_results.csv"
RACE_PREDICTION_FILE = DATA_DIR / "predicted_race_2026_round_16.csv"
LOO_SUMMARY_FILE = DATA_DIR / "loo_calibration_summary.csv"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .metric-note {
            color: #777;
            font-size: 0.82rem;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 0.7rem;
        }

        .method-box {
            padding: 1rem 1.2rem;
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 10px;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


championship = load_csv(CALIBRATED_FILE)
raw_monte_carlo = load_csv(MONTE_CARLO_FILE)
race_prediction = load_csv(RACE_PREDICTION_FILE)
loo_summary = load_csv(LOO_SUMMARY_FILE)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏎️ F1 2026 Championship Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Machine-learning race prediction + Monte Carlo simulation + "
    "historical probability calibration"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# DATA CHECK
# ============================================================

missing_files = []

for path in [
    CALIBRATED_FILE,
    RACE_PREDICTION_FILE,
    LOO_SUMMARY_FILE,
]:
    if not path.exists():
        missing_files.append(path.name)

if championship is None:
    st.error(
        "The calibrated championship results file was not found. "
        "Run the model pipeline first."
    )
    st.stop()

if missing_files:
    st.warning(
        "Some optional project files are missing: "
        + ", ".join(missing_files)
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard")

st.sidebar.caption(
    "Current model state: after Round 15, "
    "with 8 races remaining."
)

st.sidebar.markdown("---")

show_raw = st.sidebar.checkbox(
    "Show raw Monte Carlo probabilities",
    value=False,
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Probabilities are model estimates based on "
    "historical F1 data and simulated future outcomes."
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

probability_col = (
    "calibrated_championship_probability"
    if "calibrated_championship_probability" in championship.columns
    else "championship_probability"
)

if "current_points" not in championship.columns:
    championship["current_points"] = 0.0

if "expected_final_points" not in championship.columns:
    championship["expected_final_points"] = None

if "expected_final_position" not in championship.columns:
    championship["expected_final_position"] = None

if "most_likely_final_position" not in championship.columns:
    championship["most_likely_final_position"] = None

championship = championship.sort_values(
    probability_col,
    ascending=False,
).reset_index(drop=True)


# ============================================================
# OVERVIEW
# ============================================================

leader = championship.iloc[0]

leader_name = leader["driver"]
leader_probability = float(leader[probability_col])
leader_points = float(leader["current_points"])

remaining_races = 8

st.markdown(
    '<div class="section-title">Championship Overview</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Model-estimated leader",
        leader_name,
    )

with c2:
    st.metric(
        "Current points",
        f"{leader_points:.0f}",
    )

with c3:
    st.metric(
        "Championship probability",
        f"{leader_probability:.2f}%",
    )

with c4:
    st.metric(
        "Races remaining",
        remaining_races,
    )


# ============================================================
# PROBABILITY CHART
# ============================================================

st.markdown(
    '<div class="section-title">Championship Probability</div>',
    unsafe_allow_html=True,
)

chart_df = championship[
    [
        "driver",
        probability_col,
    ]
].copy()

chart_df = chart_df.rename(
    columns={
        probability_col: "Probability"
    }
)

chart_df["Probability"] = chart_df["Probability"].astype(float)

fig = px.bar(
    chart_df,
    x="Probability",
    y="driver",
    orientation="h",
    text="Probability",
    labels={
        "driver": "Driver",
        "Probability": "Championship probability (%)",
    },
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
)

fig.update_layout(
    height=max(450, len(chart_df) * 26),
    yaxis={"categoryorder": "total ascending"},
    margin=dict(l=10, r=80, t=20, b=20),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# TOP CONTENDERS
# ============================================================

st.markdown(
    '<div class="section-title">Championship Contenders</div>',
    unsafe_allow_html=True,
)

top_n = min(8, len(championship))

display_df = championship.head(top_n).copy()

display_df["Championship probability"] = (
    display_df[probability_col].map(
        lambda x: f"{float(x):.2f}%"
    )
)

display_df["Current points"] = display_df[
    "current_points"
].map(
    lambda x: f"{float(x):.0f}"
)

if display_df["expected_final_points"].notna().any():
    display_df["Expected final points"] = display_df[
        "expected_final_points"
    ].map(
        lambda x: (
            f"{float(x):.1f}"
            if pd.notna(x)
            else "—"
        )
    )

if display_df["expected_final_position"].notna().any():
    display_df["Expected final position"] = display_df[
        "expected_final_position"
    ].map(
        lambda x: (
            f"{float(x):.2f}"
            if pd.notna(x)
            else "—"
        )
    )

if display_df["most_likely_final_position"].notna().any():
    display_df["Most likely position"] = display_df[
        "most_likely_final_position"
    ].map(
        lambda x: (
            f"{int(x)}"
            if pd.notna(x)
            else "—"
        )
    )

columns_to_show = [
    "driver",
    "constructor",
    "Current points",
]

for col in [
    "Expected final points",
    "Expected final position",
    "Most likely position",
    "Championship probability",
]:
    if col in display_df.columns:
        columns_to_show.append(col)

st.dataframe(
    display_df[columns_to_show].rename(
        columns={
            "driver": "Driver",
            "constructor": "Constructor",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# ROUND 16
# ============================================================

st.markdown(
    '<div class="section-title">Round 16 — Race Prediction</div>',
    unsafe_allow_html=True,
)

if race_prediction is not None:

    race_df = race_prediction.copy()

    preferred_columns = [
        "predicted_position",
        "driver",
        "constructor",
        "raw_predicted_position",
        "predicted_race_points",
        "prediction_uncertainty",
    ]

    available = [
        c for c in preferred_columns
        if c in race_df.columns
    ]

    race_df = race_df[available]

    rename_map = {
        "predicted_position": "Predicted position",
        "driver": "Driver",
        "constructor": "Constructor",
        "raw_predicted_position": "Raw predicted position",
        "predicted_race_points": "Predicted points",
        "prediction_uncertainty": "Prediction uncertainty",
    }

    race_df = race_df.rename(
        columns=rename_map
    )

    for col in [
        "Raw predicted position",
        "Predicted points",
        "Prediction uncertainty",
    ]:
        if col in race_df.columns:
            race_df[col] = race_df[col].round(2)

    st.dataframe(
        race_df,
        use_container_width=True,
        hide_index=True,
    )

    if "predicted_race_points" in race_prediction.columns:
        total_points = race_prediction[
            "predicted_race_points"
        ].sum()

        st.caption(
            f"Total predicted Round 16 race points: "
            f"{total_points:.0f}"
        )

else:
    st.info(
        "Round 16 prediction file is not available."
    )


# ============================================================
# RAW VS CALIBRATED
# ============================================================

if show_raw and raw_monte_carlo is not None:

    st.markdown(
        '<div class="section-title">'
        "Raw vs Calibrated Probabilities"
        "</div>",
        unsafe_allow_html=True,
    )

    if "championship_probability" in raw_monte_carlo.columns:

        raw_df = raw_monte_carlo[
            [
                "driver",
                "championship_probability",
            ]
        ].copy()

        calibrated_lookup = championship[
            [
                "driver",
                probability_col,
            ]
        ].copy()

        comparison = raw_df.merge(
            calibrated_lookup,
            on="driver",
            how="left",
        )

        comparison = comparison.rename(
            columns={
                "championship_probability":
                    "Raw probability (%)",
                probability_col:
                    "Calibrated probability (%)",
            }
        )

        st.dataframe(
            comparison.sort_values(
                "Calibrated probability (%)",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">Model Performance & Calibration</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "2025 Test MAE",
        "3.853",
    )

with m2:
    st.metric(
        "2025 Test RMSE",
        "4.758",
    )

with m3:
    st.metric(
        "Raw LOO Brier",
        "0.6975",
    )

with m4:
    st.metric(
        "Calibrated LOO Brier",
        "0.4554",
    )


if loo_summary is not None:

    st.markdown("**Leave-one-season-out calibration**")

    summary_display = loo_summary.copy()

    rename_summary = {
        "evaluation": "Evaluation",
        "brier_score": "Brier score",
        "log_loss": "Log loss",
        "champion_accuracy": "Champion accuracy",
        "temperature": "Temperature",
    }

    summary_display = summary_display.rename(
        columns={
            k: v
            for k, v in rename_summary.items()
            if k in summary_display.columns
        }
    )

    if "Champion accuracy" in summary_display.columns:
        summary_display[
            "Champion accuracy"
        ] = summary_display[
            "Champion accuracy"
        ].map(
            lambda x: (
                f"{float(x) * 100:.1f}%"
                if pd.notna(x)
                else "—"
            )
        )

    st.dataframe(
        summary_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# METHODOLOGY
# ============================================================

st.markdown(
    '<div class="section-title">How the Model Works</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="method-box">
    <b>1. Historical F1 data</b><br>
    Race results, qualifying, sprint results, driver performance,
    constructor performance, and championship features are used to
    construct the modelling dataset.
    </div>

    <div class="method-box">
    <b>2. Pre-qualifying Random Forest</b><br>
    The model estimates each driver's finishing position before the
    current race's grid and qualifying result are known.
    </div>

    <div class="method-box">
    <b>3. Monte Carlo simulation</b><br>
    The remaining 2026 races are simulated 10,000 times, including
    prediction uncertainty and driver DNF behaviour.
    </div>

    <div class="method-box">
    <b>4. Championship probabilities</b><br>
    Each simulation produces a final championship outcome. The
    proportion of simulations won by each driver becomes the raw
    championship probability.
    </div>

    <div class="method-box">
    <b>5. Probability calibration</b><br>
    Historical leave-one-season-out evaluation is used to reduce
    overconfidence in the raw Monte Carlo probabilities.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "F1 2026 Championship Predictor • "
    "Probabilities are model estimates, not guarantees."
)
