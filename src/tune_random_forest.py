import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


print("=" * 60)
print("F1 RANDOM FOREST - HYPERPARAMETER TUNING")
print("=" * 60)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/ml_dataset.csv")

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# 2. Features
# --------------------------------------------------

features = [
    "grid",

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
    "constructor_wins_before"
]

target = "position"


# --------------------------------------------------
# 3. Time-based split
# --------------------------------------------------

train_df = df[df["season"] <= 2023]
validation_df = df[df["season"] == 2024]
test_df = df[df["season"] == 2025]

X_train = train_df[features]
y_train = train_df[target]

X_val = validation_df[features]
y_val = validation_df[target]

X_test = test_df[features]
y_test = test_df[target]


print("\nTime-based split:")
print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)


# --------------------------------------------------
# 4. Hyperparameter combinations
# --------------------------------------------------

configs = [
    {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "max_features": 0.8
    },
    {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "max_features": 0.8
    },
    {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 3,
        "max_features": 0.8
    },
    {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 3,
        "max_features": 0.7
    },
    {
        "n_estimators": 400,
        "max_depth": 15,
        "min_samples_leaf": 3,
        "max_features": 0.8
    },
    {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_leaf": 4,
        "max_features": 0.7
    }
]


# --------------------------------------------------
# 5. Tune using validation set
# --------------------------------------------------

results = []

print("\nStarting hyperparameter tuning...\n")


for i, config in enumerate(configs, start=1):

    print(f"Testing configuration {i}/{len(configs)}...")
    print(config)

    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    mae = mean_absolute_error(
        y_val,
        val_predictions
    )

    rmse = mean_squared_error(
        y_val,
        val_predictions
    ) ** 0.5

    results.append({
        **config,
        "validation_mae": mae,
        "validation_rmse": rmse
    })

    print(f"Validation MAE  : {mae:.3f}")
    print(f"Validation RMSE : {rmse:.3f}")
    print()


# --------------------------------------------------
# 6. Compare configurations
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="validation_mae"
)


print("=" * 60)
print("TUNING RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# 7. Select best configuration
# --------------------------------------------------

best = results_df.iloc[0]

print("\n" + "=" * 60)
print("BEST CONFIGURATION")
print("=" * 60)

print("n_estimators    :", int(best["n_estimators"]))
print("max_depth       :", int(best["max_depth"]))
print("min_samples_leaf:", int(best["min_samples_leaf"]))
print("max_features    :", best["max_features"])

print("\nValidation MAE :", round(best["validation_mae"], 3))
print("Validation RMSE:", round(best["validation_rmse"], 3))


# --------------------------------------------------
# 8. Test best configuration
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST - 2025")
print("=" * 60)

best_model = RandomForestRegressor(
    n_estimators=int(best["n_estimators"]),
    max_depth=int(best["max_depth"]),
    min_samples_leaf=int(best["min_samples_leaf"]),
    max_features=best["max_features"],
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

print("2025 MAE :", round(test_mae, 3))
print("2025 RMSE:", round(test_rmse, 3))


# --------------------------------------------------
# 9. Save tuning results
# --------------------------------------------------

results_df.to_csv(
    "data/random_forest_tuning_results.csv",
    index=False
)

print("\nSaved:")
print("data/random_forest_tuning_results.csv")

print("\n" + "=" * 60)
print("TUNING COMPLETE")
print("=" * 60)