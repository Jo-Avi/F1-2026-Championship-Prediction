import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/ml_dataset.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "random_forest_prequalifying.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("F1 PRE-QUALIFYING RACE PREDICTION MODEL")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

target = "position"


# ============================================================
# FEATURES
# ============================================================

# IMPORTANT:
#
# grid and qualifying_position are intentionally excluded.
#
# They are unknown when predicting a race before qualifying.
#
# Historical qualifying performance is still allowed because
# those rolling features only use PREVIOUS races.

numeric_features = [

    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",

    "driver_avg_points_last_3",
    "driver_avg_points_last_5",

    "driver_avg_qualifying_last_3",
    "driver_avg_qualifying_last_5",

    "driver_dnf_rate",

    "constructor_avg_finish_last_3",
    "constructor_avg_points_last_3",
    "constructor_avg_qualifying_last_3",

    "driver_championship_position_before",
    "driver_championship_points_before",
    "driver_wins_before",

    "constructor_championship_position_before",
    "constructor_championship_points_before",
    "constructor_wins_before",
]


categorical_features = [
    "driver",
    "constructor",
]


all_features = (
    numeric_features
    + categorical_features
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = (
    all_features
    + [target, "season"]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# TIME-BASED SPLIT
# ============================================================

train_df = df[
    df["season"] <= 2023
].copy()

validation_df = df[
    df["season"] == 2024
].copy()

test_df = df[
    df["season"] == 2025
].copy()


print("\nTime-based split:")

print(
    f"Training   : {train_df.shape}"
)

print(
    f"Validation : {validation_df.shape}"
)

print(
    f"Test       : {test_df.shape}"
)


# ============================================================
# X / y
# ============================================================

X_train = train_df[
    all_features
]

y_train = train_df[
    target
]


X_validation = validation_df[
    all_features
]

y_validation = validation_df[
    target
]


X_test = test_df[
    all_features
]

y_test = test_df[
    target
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining pre-qualifying Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# PREDICTIONS
# ============================================================

train_predictions = pipeline.predict(
    X_train
)

validation_predictions = pipeline.predict(
    X_validation
)

test_predictions = pipeline.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    return mae, rmse


train_mae, train_rmse = calculate_metrics(
    y_train,
    train_predictions
)

validation_mae, validation_rmse = calculate_metrics(
    y_validation,
    validation_predictions
)

test_mae, test_rmse = calculate_metrics(
    y_test,
    test_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("PRE-QUALIFYING MODEL RESULTS")
print("=" * 65)


print("\nTraining")

print(
    f"MAE  : {train_mae:.3f}"
)

print(
    f"RMSE : {train_rmse:.3f}"
)


print("\nValidation - 2024")

print(
    f"MAE  : {validation_mae:.3f}"
)

print(
    f"RMSE : {validation_rmse:.3f}"
)


print("\nTest - 2025")

print(
    f"MAE  : {test_mae:.3f}"
)

print(
    f"RMSE : {test_rmse:.3f}"
)


# ============================================================
# SAMPLE TEST PREDICTIONS
# ============================================================

results = test_df[
    [
        "season",
        "round",
        "driver",
        "constructor",
        "position"
    ]
].copy()


results["predicted_position"] = (
    test_predictions
)

results["prediction_error"] = (
    results["predicted_position"]
    - results["position"]
)


results["predicted_position"] = (
    results["predicted_position"]
    .round(2)
)


print("\nSample 2025 predictions:")

print(
    results
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    pipeline,
    MODEL_FILE
)


print("\nModel saved:")

print(
    MODEL_FILE
)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_info = {
    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    "all_features":
        all_features,

    "prediction_type":
        "pre_qualifying"
}


feature_file = os.path.join(
    MODEL_DIR,
    "prequalifying_features.pkl"
)


joblib.dump(
    feature_info,
    feature_file
)


print("\nFeature configuration saved:")

print(
    feature_file
)


print("\n" + "=" * 65)
print("PRE-QUALIFYING MODEL COMPLETE")
print("=" * 65)