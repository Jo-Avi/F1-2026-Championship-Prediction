"""
F1 Championship Probability Calibration

Uses historical Monte Carlo backtest probabilities to compare:
1. Raw probabilities
2. Temperature scaling
3. Uniform shrinkage

Then applies the selected calibration method to the existing 2026
Monte Carlo results.

Expected input files:
    data/monte_carlo_backtest_driver_results.csv
    data/monte_carlo_2026_results.csv

Outputs:
    data/calibrated_historical_probabilities.csv
    data/probability_calibration_summary.csv
    data/monte_carlo_2026_calibrated_results.csv
"""

import os
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

HISTORICAL_FILE = os.path.join(
    DATA_DIR, "monte_carlo_backtest_driver_results.csv"
)

CURRENT_2026_FILE = os.path.join(
    DATA_DIR, "monte_carlo_2026_results.csv"
)

HISTORICAL_OUTPUT = os.path.join(
    DATA_DIR, "calibrated_historical_probabilities.csv"
)

SUMMARY_OUTPUT = os.path.join(
    DATA_DIR, "probability_calibration_summary.csv"
)

CURRENT_OUTPUT = os.path.join(
    DATA_DIR, "monte_carlo_2026_calibrated_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_probabilities(values):
    """Convert values to a valid probability vector."""
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 1e-12, None)
    total = values.sum()

    if total <= 0:
        return np.full(len(values), 1.0 / len(values))

    return values / total


def multiclass_brier(probabilities, actual_indices):
    """
    Proper multiclass Brier score.

    For each season:
        sum_i (p_i - y_i)^2

    Lower is better.
    """
    scores = []

    for p, actual_index in zip(probabilities, actual_indices):
        y = np.zeros(len(p))
        y[actual_index] = 1.0
        scores.append(np.sum((p - y) ** 2))

    return float(np.mean(scores))


def multiclass_log_loss(probabilities, actual_indices):
    """Multiclass log loss using the probability assigned to the actual champion."""
    values = []

    for p, actual_index in zip(probabilities, actual_indices):
        actual_probability = np.clip(p[actual_index], 1e-12, 1.0)
        values.append(-np.log(actual_probability))

    return float(np.mean(values))


def temperature_scale(probabilities, temperature):
    """
    Temperature scaling.

    T > 1 reduces confidence.
    T < 1 increases confidence.
    """
    probabilities = normalize_probabilities(probabilities)

    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    scaled = logits / temperature

    scaled -= np.max(scaled)

    exp_values = np.exp(scaled)
    return normalize_probabilities(exp_values)


def shrink_to_uniform(probabilities, alpha):
    """
    Pull probabilities toward a uniform distribution.

    alpha = 0 -> unchanged
    alpha = 1 -> completely uniform
    """
    probabilities = normalize_probabilities(probabilities)
    uniform = np.full(
        len(probabilities),
        1.0 / len(probabilities)
    )

    return (
        (1.0 - alpha) * probabilities
        + alpha * uniform
    )


# ============================================================
# LOAD HISTORICAL RESULTS
# ============================================================

print("=" * 72)
print("F1 CHAMPIONSHIP PROBABILITY CALIBRATION")
print("=" * 72)

print("\nLoading historical Monte Carlo results...")

if not os.path.exists(HISTORICAL_FILE):
    raise FileNotFoundError(
        f"Missing historical results:\n{HISTORICAL_FILE}"
    )

historical = pd.read_csv(HISTORICAL_FILE)

required_columns = [
    "season",
    "driver",
    "actual_champion",
    "championship_probability",
]

missing = [
    c for c in required_columns
    if c not in historical.columns
]

if missing:
    raise ValueError(
        f"Historical results are missing columns: {missing}"
    )

print(f"Historical rows: {len(historical):,}")


# ============================================================
# BUILD ONE PROBABILITY VECTOR PER SEASON
# ============================================================

season_vectors = []
actual_indices = []
season_numbers = []
driver_orders = []

for season in sorted(historical["season"].unique()):

    season_data = historical[
        historical["season"] == season
    ].copy()

    season_data = season_data.sort_values(
        "championship_probability",
        ascending=False
    ).reset_index(drop=True)

    probabilities = normalize_probabilities(
        season_data["championship_probability"].values
    )

    actual_rows = season_data[
        season_data["actual_champion"] == True
    ]

    if len(actual_rows) != 1:
        raise ValueError(
            f"Expected exactly one actual champion for "
            f"{season}, found {len(actual_rows)}."
        )

    actual_driver = actual_rows.iloc[0]["driver"]

    actual_index = int(
        season_data.index[
            season_data["driver"] == actual_driver
        ][0]
    )

    season_vectors.append(probabilities)
    actual_indices.append(actual_index)
    season_numbers.append(int(season))
    driver_orders.append(
        season_data["driver"].tolist()
    )


# ============================================================
# RAW METRICS
# ============================================================

raw_brier = multiclass_brier(
    season_vectors,
    actual_indices
)

raw_log_loss = multiclass_log_loss(
    season_vectors,
    actual_indices
)


# ============================================================
# TEMPERATURE SEARCH
# ============================================================

best_temperature = None
best_temperature_brier = float("inf")
best_temperature_log_loss = None

temperature_results = []

temperature_grid = np.linspace(
    0.50,
    10.00,
    191
)

for temperature in temperature_grid:

    calibrated_vectors = [
        temperature_scale(p, temperature)
        for p in season_vectors
    ]

    brier = multiclass_brier(
        calibrated_vectors,
        actual_indices
    )

    log_loss = multiclass_log_loss(
        calibrated_vectors,
        actual_indices
    )

    temperature_results.append({
        "temperature": temperature,
        "brier_score": brier,
        "log_loss": log_loss,
    })

    if brier < best_temperature_brier:
        best_temperature_brier = brier
        best_temperature = temperature
        best_temperature_log_loss = log_loss


# ============================================================
# SHRINKAGE SEARCH
# ============================================================

best_alpha = None
best_shrinkage_brier = float("inf")
best_shrinkage_log_loss = None

shrinkage_results = []

alpha_grid = np.linspace(
    0.00,
    1.00,
    101
)

for alpha in alpha_grid:

    calibrated_vectors = [
        shrink_to_uniform(p, alpha)
        for p in season_vectors
    ]

    brier = multiclass_brier(
        calibrated_vectors,
        actual_indices
    )

    log_loss = multiclass_log_loss(
        calibrated_vectors,
        actual_indices
    )

    shrinkage_results.append({
        "alpha": alpha,
        "brier_score": brier,
        "log_loss": log_loss,
    })

    if brier < best_shrinkage_brier:
        best_shrinkage_brier = brier
        best_alpha = alpha
        best_shrinkage_log_loss = log_loss


# ============================================================
# SELECT METHOD
# ============================================================

methods = {
    "raw": raw_brier,
    "temperature_scaling": best_temperature_brier,
    "uniform_shrinkage": best_shrinkage_brier,
}

selected_method = min(
    methods,
    key=methods.get
)


if selected_method == "temperature_scaling":
    selected_parameter = best_temperature
elif selected_method == "uniform_shrinkage":
    selected_parameter = best_alpha
else:
    selected_parameter = 0.0


# ============================================================
# CALIBRATE HISTORICAL RESULTS
# ============================================================

calibrated_rows = []

for season, probabilities, drivers in zip(
    season_numbers,
    season_vectors,
    driver_orders
):

    if selected_method == "temperature_scaling":
        calibrated = temperature_scale(
            probabilities,
            best_temperature
        )

    elif selected_method == "uniform_shrinkage":
        calibrated = shrink_to_uniform(
            probabilities,
            best_alpha
        )

    else:
        calibrated = probabilities.copy()

    for driver, raw_probability, calibrated_probability in zip(
        drivers,
        probabilities,
        calibrated
    ):
        calibrated_rows.append({
            "season": season,
            "driver": driver,
            "raw_probability": raw_probability * 100,
            "calibrated_probability": calibrated_probability * 100,
        })


calibrated_historical = pd.DataFrame(
    calibrated_rows
)


# ============================================================
# APPLY TO 2026
# ============================================================

if not os.path.exists(CURRENT_2026_FILE):
    print(
        "\nWARNING: 2026 Monte Carlo results were not found:"
    )
    print(CURRENT_2026_FILE)
    print(
        "\nHistorical calibration was completed, but "
        "2026 calibration was skipped."
    )

    calibrated_historical.to_csv(
        HISTORICAL_OUTPUT,
        index=False
    )

else:

    current_2026 = pd.read_csv(
        CURRENT_2026_FILE
    )

    probability_column = "championship_probability"

    if probability_column not in current_2026.columns:
        raise ValueError(
            f"2026 results are missing '{probability_column}'."
        )

    raw_2026 = normalize_probabilities(
        current_2026[probability_column].values
    )

    if selected_method == "temperature_scaling":

        calibrated_2026 = temperature_scale(
            raw_2026,
            best_temperature
        )

    elif selected_method == "uniform_shrinkage":

        calibrated_2026 = shrink_to_uniform(
            raw_2026,
            best_alpha
        )

    else:

        calibrated_2026 = raw_2026.copy()

    current_2026[
        "raw_championship_probability"
    ] = raw_2026 * 100

    current_2026[
        "calibrated_championship_probability"
    ] = calibrated_2026 * 100

    current_2026 = current_2026.sort_values(
        "calibrated_championship_probability",
        ascending=False
    ).reset_index(drop=True)

    current_2026[
        "calibrated_championship_probability"
    ] = current_2026[
        "calibrated_championship_probability"
    ].round(2)

    current_2026[
        "raw_championship_probability"
    ] = current_2026[
        "raw_championship_probability"
    ].round(2)

    current_2026.to_csv(
        CURRENT_OUTPUT,
        index=False
    )


# ============================================================
# SAVE HISTORICAL CALIBRATION
# ============================================================

calibrated_historical[
    "raw_probability"
] = calibrated_historical[
    "raw_probability"
].round(4)

calibrated_historical[
    "calibrated_probability"
] = calibrated_historical[
    "calibrated_probability"
].round(4)

calibrated_historical.to_csv(
    HISTORICAL_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "method": "raw",
        "parameter": 0.0,
        "brier_score": raw_brier,
        "log_loss": raw_log_loss,
    },
    {
        "method": "temperature_scaling",
        "parameter": best_temperature,
        "brier_score": best_temperature_brier,
        "log_loss": best_temperature_log_loss,
    },
    {
        "method": "uniform_shrinkage",
        "parameter": best_alpha,
        "brier_score": best_shrinkage_brier,
        "log_loss": best_shrinkage_log_loss,
    },
])

summary[
    "brier_score"
] = summary[
    "brier_score"
].round(6)

summary[
    "log_loss"
] = summary[
    "log_loss"
].round(6)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 72)
print("HISTORICAL CALIBRATION METRICS")
print("=" * 72)

print(
    f"\nRaw Brier score: "
    f"{raw_brier:.6f}"
)

print(
    f"Temperature scaling Brier score: "
    f"{best_temperature_brier:.6f}"
)

print(
    f"Uniform shrinkage Brier score: "
    f"{best_shrinkage_brier:.6f}"
)

print(
    f"\nRaw log loss: "
    f"{raw_log_loss:.6f}"
)

print(
    f"Temperature scaling log loss: "
    f"{best_temperature_log_loss:.6f}"
)

print(
    f"Uniform shrinkage log loss: "
    f"{best_shrinkage_log_loss:.6f}"
)

print(
    f"\nLearned temperature: "
    f"{best_temperature:.3f}"
)

print(
    f"Learned shrinkage alpha: "
    f"{best_alpha:.3f}"
)

print(
    f"\nSelected method by Brier: "
    f"{selected_method}"
)

print(
    f"Selected parameter: "
    f"{selected_parameter:.3f}"
)

print("\nCalibration summary:")
print(summary.to_string(index=False))

if os.path.exists(CURRENT_OUTPUT):

    print(
        "\n2026 calibrated results saved to:"
    )
    print(CURRENT_OUTPUT)

    display_columns = [
        c for c in [
            "driver",
            "constructor",
            "current_points",
            "raw_championship_probability",
            "calibrated_championship_probability",
        ]
        if c in current_2026.columns
    ]

    print(
        "\n2026 calibrated championship probabilities:\n"
    )

    print(
        current_2026[
            display_columns
        ].to_string(index=False)
    )

print(
    "\nHistorical calibrated results saved to:"
)
print(HISTORICAL_OUTPUT)

print(
    "\nCalibration summary saved to:"
)
print(SUMMARY_OUTPUT)

print("\n" + "=" * 72)
print("CALIBRATION COMPLETE")
print("=" * 72)
