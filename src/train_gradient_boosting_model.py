import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/ml_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("F1 FINISHING POSITION - GRADIENT BOOSTING MODEL")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

target = "position"


# ============================================================
# FEATURES
# ============================================================

numeric_features = [
    "grid",
    "qualifying_position",

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


# ============================================================
# TIME-BASED SPLIT
# ============================================================

train_df = df[df["season"] <= 2023].copy()
validation_df = df[df["season"] == 2024].copy()
test_df = df[df["season"] == 2025].copy()

print("\nTime-based split:")
print(f"Training   : {train_df.shape}")
print(f"Validation : {validation_df.shape}")
print(f"Test       : {test_df.shape}")


# ============================================================
# X / y
# ============================================================

X_train = train_df[numeric_features + categorical_features]
y_train = train_df[target]

X_validation = validation_df[numeric_features + categorical_features]
y_validation = validation_df[target]

X_test = test_df[numeric_features + categorical_features]
y_test = test_df[target]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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
# GRADIENT BOOSTING MODEL
# ============================================================

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    loss="squared_error"
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

print("\nTraining Gradient Boosting" \
"...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# Training predictions
train_predictions = pipeline.predict(X_train)

train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))


# ============================================================
# VALIDATION
# ============================================================

validation_predictions = pipeline.predict(
    X_validation
)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)


# ============================================================
# TEST
# ============================================================

test_predictions = pipeline.predict(
    X_test
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print("\nValidation - 2024")
print(f"MAE  : {validation_mae:.3f}")
print(f"RMSE : {validation_rmse:.3f}")

print("\nTest - 2025")
print(f"MAE  : {test_mae:.3f}")
print(f"RMSE : {test_rmse:.3f}")

print("=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print("\nTraining")
print(f"MAE  : {train_mae:.3f}")
print(f"RMSE : {train_rmse:.3f}")

print("\nValidation - 2024")
print(f"MAE  : {validation_mae:.3f}")
print(f"RMSE : {validation_rmse:.3f}")

print("\nTest - 2025")
print(f"MAE  : {test_mae:.3f}")
print(f"RMSE : {test_rmse:.3f}")

# ============================================================
# SAMPLE PREDICTIONS
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

results["predicted_position"] = test_predictions

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
    results.head(20).to_string(
        index=False
    )
)


print("\n" + "=" * 60)
print("GRADIENT BOOSTING MODEL COMPLETE")
print("=" * 60)