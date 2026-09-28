"""
F1 Championship Probability Calibration — Leave-One-Season-Out

Purpose:
    Evaluate temperature scaling without evaluating calibration on the same
    seasons used to fit the temperature.

Historical input:
    data/monte_carlo_backtest_driver_results.csv

Optional 2026 input:
    data/monte_carlo_2026_results.csv

Outputs:
    data/loo_calibration_results.csv
    data/loo_calibration_summary.csv
    data/monte_carlo_2026_calibrated_results.csv

Important:
    Only five historical seasons (2021–2025) are available. Results should
    therefore be treated as a small-sample calibration analysis.
"""

import os
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

HISTORICAL_FILE = os.path.join(
    DATA_DIR,
    "monte_carlo_backtest_driver_results.csv"
)

CURRENT_2026_FILE = os.path.join(
    DATA_DIR,
    "monte_carlo_2026_results.csv"
)

LOO_RESULTS_FILE = os.path.join(
    DATA_DIR,
    "loo_calibration_results.csv"
)

LOO_SUMMARY_FILE = os.path.join(
    DATA_DIR,
    "loo_calibration_summary.csv"
)

CURRENT_OUTPUT = os.path.join(
    DATA_DIR,
    "monte_carlo_2026_calibrated_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_probabilities(values):
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 1e-12, None)
    total = values.sum()

    if total <= 0:
        return np.full(len(values), 1.0 / len(values))

    return values / total


def temperature_scale(probabilities, temperature):
    """
    Temperature scaling.

    T > 1 reduces confidence.
    T < 1 increases confidence.
    """
    probabilities = normalize_probabilities(probabilities)

    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    scaled_logits = logits / temperature

    scaled_logits -= np.max(scaled_logits)

    exp_values = np.exp(scaled_logits)

    return normalize_probabilities(exp_values)


def multiclass_brier(probabilities, actual_indices):
    """
    Proper multiclass Brier score:
        mean over seasons of sum_i (p_i - y_i)^2
    """
    scores = []

    for p, actual_index in zip(
        probabilities,
        actual_indices
    ):
        y = np.zeros(len(p))
        y[actual_index] = 1.0

        scores.append(
            np.sum((p - y) ** 2)
        )

    return float(np.mean(scores))


def multiclass_log_loss(probabilities, actual_indices):
    values = []

    for p, actual_index in zip(
        probabilities,
        actual_indices
    ):
        actual_probability = np.clip(
            p[actual_index],
            1e-12,
            1.0
        )

        values.append(
            -np.log(actual_probability)
        )

    return float(np.mean(values))


def fit_temperature(
    training_vectors,
    training_actual_indices
):
    """
    Fit temperature using only the supplied training seasons.

    Temperature is selected by minimizing multiclass Brier score.
    """
    best_temperature = None
    best_brier = float("inf")
    best_log_loss = None

    # Wide enough to handle the strong overconfidence observed
    # in the original Monte Carlo probabilities.
    temperature_grid = np.linspace(
        0.50,
        10.00,
        381
    )

    for temperature in temperature_grid:

        calibrated = [
            temperature_scale(
                p,
                temperature
            )
            for p in training_vectors
        ]

        brier = multiclass_brier(
            calibrated,
            training_actual_indices
        )

        if brier < best_brier:
            best_brier = brier
            best_temperature = temperature
            best_log_loss = multiclass_log_loss(
                calibrated,
                training_actual_indices
            )

    return (
        float(best_temperature),
        float(best_brier),
        float(best_log_loss)
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("F1 LEAVE-ONE-SEASON-OUT PROBABILITY CALIBRATION")
print("=" * 72)

print("\nLoading historical Monte Carlo results...")

if not os.path.exists(HISTORICAL_FILE):
    raise FileNotFoundError(
        f"Missing historical results:\n{HISTORICAL_FILE}"
    )

historical = pd.read_csv(
    HISTORICAL_FILE
)

required_columns = [
    "season",
    "driver",
    "actual_champion",
    "championship_probability",
]

missing = [
    c
    for c in required_columns
    if c not in historical.columns
]

if missing:
    raise ValueError(
        f"Historical results are missing: {missing}"
    )


# ============================================================
# BUILD SEASON VECTORS
# ============================================================

season_data_map = {}
season_vectors = {}
season_actual_indices = {}

for season in sorted(
    historical["season"].unique()
):

    season_data = historical[
        historical["season"] == season
    ].copy()

    season_data = season_data.sort_values(
        "championship_probability",
        ascending=False
    ).reset_index(drop=True)

    probabilities = normalize_probabilities(
        season_data[
            "championship_probability"
        ].values
    )

    actual_rows = season_data[
        season_data["actual_champion"] == True
    ]

    if len(actual_rows) != 1:
        raise ValueError(
            f"Expected exactly one actual champion "
            f"for {season}; found {len(actual_rows)}."
        )

    actual_driver = actual_rows.iloc[0]["driver"]

    matching = season_data.index[
        season_data["driver"] == actual_driver
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Could not uniquely locate actual champion "
            f"{actual_driver} in {season}."
        )

    actual_index = int(matching[0])

    season_data_map[int(season)] = season_data
    season_vectors[int(season)] = probabilities
    season_actual_indices[int(season)] = actual_index


seasons = sorted(season_vectors)

print(
    f"Historical seasons: {seasons}"
)

if len(seasons) < 3:
    raise ValueError(
        "At least three seasons are required for "
        "leave-one-season-out calibration."
    )


# ============================================================
# RAW BASELINE
# ============================================================

raw_vectors = [
    season_vectors[season]
    for season in seasons
]

raw_actual_indices = [
    season_actual_indices[season]
    for season in seasons
]

raw_brier = multiclass_brier(
    raw_vectors,
    raw_actual_indices
)

raw_log_loss = multiclass_log_loss(
    raw_vectors,
    raw_actual_indices
)


# ============================================================
# LEAVE-ONE-SEASON-OUT CALIBRATION
# ============================================================

print(
    "\nRunning leave-one-season-out calibration..."
)

loo_rows = []

loo_probabilities = []
loo_actual_indices = []

for held_out_season in seasons:

    training_seasons = [
        season
        for season in seasons
        if season != held_out_season
    ]

    training_vectors = [
        season_vectors[season]
        for season in training_seasons
    ]

    training_actual_indices = [
        season_actual_indices[season]
        for season in training_seasons
    ]

    test_vector = season_vectors[
        held_out_season
    ]

    test_actual_index = season_actual_indices[
        held_out_season
    ]

    # Fit temperature ONLY on the other seasons.
    temperature, training_brier, training_log_loss = fit_temperature(
        training_vectors,
        training_actual_indices
    )

    # Evaluate ONLY on the held-out season.
    calibrated_test = temperature_scale(
        test_vector,
        temperature
    )

    raw_test_brier = multiclass_brier(
        [test_vector],
        [test_actual_index]
    )

    calibrated_test_brier = multiclass_brier(
        [calibrated_test],
        [test_actual_index]
    )

    raw_test_log_loss = multiclass_log_loss(
        [test_vector],
        [test_actual_index]
    )

    calibrated_test_log_loss = multiclass_log_loss(
        [calibrated_test],
        [test_actual_index]
    )

    actual_driver = season_data_map[
        held_out_season
    ].loc[
        test_actual_index,
        "driver"
    ]

    raw_predicted_index = int(
        np.argmax(test_vector)
    )

    calibrated_predicted_index = int(
        np.argmax(calibrated_test)
    )

    raw_predicted_driver = season_data_map[
        held_out_season
    ].loc[
        raw_predicted_index,
        "driver"
    ]

    calibrated_predicted_driver = season_data_map[
        held_out_season
    ].loc[
        calibrated_predicted_index,
        "driver"
    ]

    raw_actual_probability = (
        test_vector[test_actual_index] * 100
    )

    calibrated_actual_probability = (
        calibrated_test[test_actual_index] * 100
    )

    print(
        f"\nHeld-out season: {held_out_season}"
    )
    print(
        f"Training seasons: {training_seasons}"
    )
    print(
        f"Learned temperature: {temperature:.2f}"
    )
    print(
        f"Actual champion: {actual_driver}"
    )
    print(
        f"Raw actual-champion probability: "
        f"{raw_actual_probability:.2f}%"
    )
    print(
        f"Calibrated actual-champion probability: "
        f"{calibrated_actual_probability:.2f}%"
    )

    loo_rows.append({
        "held_out_season": held_out_season,
        "training_seasons": ",".join(
            map(str, training_seasons)
        ),
        "temperature": temperature,
        "actual_champion": actual_driver,
        "raw_predicted_champion": raw_predicted_driver,
        "calibrated_predicted_champion": calibrated_predicted_driver,
        "raw_actual_champion_probability": raw_actual_probability,
        "calibrated_actual_champion_probability": calibrated_actual_probability,
        "raw_brier_score": raw_test_brier,
        "calibrated_brier_score": calibrated_test_brier,
        "raw_log_loss": raw_test_log_loss,
        "calibrated_log_loss": calibrated_test_log_loss,
        "raw_correct": (
            raw_predicted_driver == actual_driver
        ),
        "calibrated_correct": (
            calibrated_predicted_driver == actual_driver
        ),
    })

    loo_probabilities.append(
        calibrated_test
    )

    loo_actual_indices.append(
        test_actual_index
    )


loo_results = pd.DataFrame(
    loo_rows
)


# ============================================================
# AGGREGATE OUT-OF-SAMPLE METRICS
# ============================================================

loo_raw_brier = float(
    loo_results["raw_brier_score"].mean()
)

loo_calibrated_brier = float(
    loo_results["calibrated_brier_score"].mean()
)

loo_raw_log_loss = float(
    loo_results["raw_log_loss"].mean()
)

loo_calibrated_log_loss = float(
    loo_results["calibrated_log_loss"].mean()
)

raw_accuracy = float(
    loo_results["raw_correct"].mean()
)

calibrated_accuracy = float(
    loo_results["calibrated_correct"].mean()
)


# ============================================================
# FIT FINAL TEMPERATURE ON ALL HISTORICAL SEASONS
# ============================================================

final_temperature, final_training_brier, final_training_log_loss = (
    fit_temperature(
        raw_vectors,
        raw_actual_indices
    )
)


# ============================================================
# APPLY FINAL CALIBRATION TO 2026
# ============================================================

if not os.path.exists(CURRENT_2026_FILE):

    print(
        "\nWARNING: 2026 Monte Carlo results not found."
    )

    current_2026 = None

else:

    current_2026 = pd.read_csv(
        CURRENT_2026_FILE
    )

    if "championship_probability" not in current_2026.columns:
        raise ValueError(
            "2026 results are missing "
            "'championship_probability'."
        )

    raw_2026 = normalize_probabilities(
        current_2026[
            "championship_probability"
        ].values
    )

    calibrated_2026 = temperature_scale(
        raw_2026,
        final_temperature
    )

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
        "raw_championship_probability"
    ] = current_2026[
        "raw_championship_probability"
    ].round(2)

    current_2026[
        "calibrated_championship_probability"
    ] = current_2026[
        "calibrated_championship_probability"
    ].round(2)

    current_2026.to_csv(
        CURRENT_OUTPUT,
        index=False
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

loo_results[
    "temperature"
] = loo_results[
    "temperature"
].round(4)

loo_results[
    "raw_actual_champion_probability"
] = loo_results[
    "raw_actual_champion_probability"
].round(4)

loo_results[
    "calibrated_actual_champion_probability"
] = loo_results[
    "calibrated_actual_champion_probability"
].round(4)

loo_results[
    "raw_brier_score"
] = loo_results[
    "raw_brier_score"
].round(6)

loo_results[
    "calibrated_brier_score"
] = loo_results[
    "calibrated_brier_score"
].round(6)

loo_results[
    "raw_log_loss"
] = loo_results[
    "raw_log_loss"
].round(6)

loo_results[
    "calibrated_log_loss"
] = loo_results[
    "calibrated_log_loss"
].round(6)

loo_results.to_csv(
    LOO_RESULTS_FILE,
    index=False
)


summary = pd.DataFrame([
    {
        "evaluation": "LOO raw",
        "brier_score": loo_raw_brier,
        "log_loss": loo_raw_log_loss,
        "champion_accuracy": raw_accuracy,
        "temperature": np.nan,
    },
    {
        "evaluation": "LOO temperature scaling",
        "brier_score": loo_calibrated_brier,
        "log_loss": loo_calibrated_log_loss,
        "champion_accuracy": calibrated_accuracy,
        "temperature": np.nan,
    },
    {
        "evaluation": "Final all-season temperature",
        "brier_score": final_training_brier,
        "log_loss": final_training_log_loss,
        "champion_accuracy": np.nan,
        "temperature": final_temperature,
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

summary[
    "champion_accuracy"
] = summary[
    "champion_accuracy"
].round(4)

summary[
    "temperature"
] = summary[
    "temperature"
].round(4)

summary.to_csv(
    LOO_SUMMARY_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 72)
print("LEAVE-ONE-SEASON-OUT CALIBRATION COMPLETE")
print("=" * 72)

print("\nOut-of-sample metrics:")
print(
    f"Raw LOO Brier score: "
    f"{loo_raw_brier:.6f}"
)
print(
    f"Calibrated LOO Brier score: "
    f"{loo_calibrated_brier:.6f}"
)
print(
    f"Raw LOO log loss: "
    f"{loo_raw_log_loss:.6f}"
)
print(
    f"Calibrated LOO log loss: "
    f"{loo_calibrated_log_loss:.6f}"
)
print(
    f"Raw LOO champion accuracy: "
    f"{raw_accuracy * 100:.1f}%"
)
print(
    f"Calibrated LOO champion accuracy: "
    f"{calibrated_accuracy * 100:.1f}%"
)

print(
    f"\nFinal temperature fitted on all "
    f"historical seasons: {final_temperature:.3f}"
)

print("\nSeason-by-season LOO results:\n")
print(
    loo_results.to_string(
        index=False
    )
)

if current_2026 is not None:

    print(
        "\n2026 probabilities using the final "
        "all-season temperature:\n"
    )

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
        current_2026[
            display_columns
        ].to_string(index=False)
    )

    print(
        "\n2026 calibrated results saved to:"
    )
    print(CURRENT_OUTPUT)

print(
    "\nLOO results saved to:"
)
print(LOO_RESULTS_FILE)

print(
    "\nLOO summary saved to:"
)
print(LOO_SUMMARY_FILE)

print("\n" + "=" * 72)
