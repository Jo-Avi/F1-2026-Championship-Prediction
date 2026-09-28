import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "data/upcoming_race_features_2026.csv"

MODEL_FILE = "models/random_forest_prequalifying.pkl"

OUTPUT_FILE = "data/predicted_race_2026_round_16.csv"


# ============================================================
# F1 RACE POINTS
# ============================================================

# Standard full-race points for positions 1–10.
#
# The model predicts finishing order. We convert the projected
# positions into championship points.

RACE_POINTS = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("2026 F1 ROUND 16 RACE PREDICTION")
print("=" * 70)

print("\nLoading pre-qualifying Random Forest...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading upcoming race features...")

df = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Feature dataset shape: {df.shape}"
)


# ============================================================
# CHECK ROUND
# ============================================================

if "round" not in df.columns:

    raise ValueError(
        "The feature dataset does not contain a 'round' column."
    )


prediction_round = int(
    df["round"].iloc[0]
)

print(
    f"\nPrediction round: {prediction_round}"
)


# ============================================================
# LOAD FEATURE CONFIGURATION
# ============================================================

FEATURE_CONFIG_FILE = (
    "models/prequalifying_features.pkl"
)

feature_config = joblib.load(
    FEATURE_CONFIG_FILE
)

numeric_features = (
    feature_config["numeric_features"]
)

categorical_features = (
    feature_config["categorical_features"]
)

all_features = (
    feature_config["all_features"]
)


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing_features = [
    feature
    for feature in all_features
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing prediction features:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

X = df[
    all_features
].copy()


# ============================================================
# PREDICT FINISHING POSITION
# ============================================================

print("\nGenerating race predictions...")

raw_predictions = model.predict(
    X
)


df["raw_predicted_position"] = (
    raw_predictions
)


# ============================================================
# CONVERT REGRESSION OUTPUT INTO A VALID RANKING
# ============================================================

# Random Forest regression can produce duplicate predicted
# positions such as:
#
# Driver A -> 5.7
# Driver B -> 5.9
#
# A real race cannot have two drivers finishing P6.
#
# Therefore we sort drivers by predicted position and assign
# unique projected finishing positions.

df = df.sort_values(
    "raw_predicted_position"
).reset_index(
    drop=True
)

df["predicted_position"] = (
    np.arange(
        1,
        len(df) + 1
    )
)


# ============================================================
# ASSIGN RACE POINTS
# ============================================================

df["predicted_race_points"] = (
    df["predicted_position"]
    .map(RACE_POINTS)
    .fillna(0)
)


# ============================================================
# PREDICTION UNCERTAINTY
# ============================================================

# Calculate the standard deviation of predictions from all
# individual trees in the Random Forest.
#
# This gives us a rough indication of how much the trees
# disagree about a driver's finishing position.

try:

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    rf_model = (
        model.named_steps[
            "model"
        ]
    )

    X_transformed = (
        preprocessor.transform(X)
    )

    tree_predictions = np.column_stack(
        [
            estimator.predict(
                X_transformed
            )
            for estimator in rf_model.estimators_
        ]
    )

    prediction_std = (
        tree_predictions.std(
            axis=1
        )
    )

    df["prediction_uncertainty"] = (
        prediction_std
    )

except Exception as error:

    print(
        "\nWarning: Could not calculate "
        "tree-level prediction uncertainty."
    )

    print(
        f"Reason: {error}"
    )

    df["prediction_uncertainty"] = np.nan


# ============================================================
# FINAL RESULT COLUMNS
# ============================================================

result_columns = [
    "season",
    "round",
    "driver",
    "constructor",
    "raw_predicted_position",
    "predicted_position",
    "predicted_race_points",
    "prediction_uncertainty"
]

results = df[
    result_columns
].copy()


results["raw_predicted_position"] = (
    results["raw_predicted_position"]
    .round(2)
)

results["prediction_uncertainty"] = (
    results["prediction_uncertainty"]
    .round(2)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print(
    f"2026 ROUND {prediction_round} PREDICTION"
)
print("=" * 70)

display_columns = [
    "predicted_position",
    "driver",
    "constructor",
    "raw_predicted_position",
    "predicted_race_points",
    "prediction_uncertainty"
]

print(
    results[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# TOTAL PREDICTED POINTS
# ============================================================

total_points = (
    results["predicted_race_points"]
    .sum()
)

print(
    f"\nTotal predicted race points: "
    f"{total_points:.0f}"
)


# ============================================================
# OUTPUT
# ============================================================

print("\nSaved prediction file:")

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("RACE PREDICTION COMPLETE")
print("=" * 70)