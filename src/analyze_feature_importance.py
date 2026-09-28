import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


print("=" * 60)
print("F1 MODEL - FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/ml_dataset.csv")

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

target = "position"

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


X = df[features]
y = df[target]


# --------------------------------------------------
# 3. Time-based split
# --------------------------------------------------

train_df = df[df["season"] <= 2023]
validation_df = df[df["season"] == 2024]
test_df = df[df["season"] == 2025]

X_train = train_df[features]
y_train = train_df[target]


print("\nTraining rows:", len(X_train))


# --------------------------------------------------
# 4. Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete.")


# --------------------------------------------------
# 5. Calculate feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.to_string(index=False)
)


# --------------------------------------------------
# 7. Plot feature importance
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    importance["feature"],
    importance["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "data/random_forest_feature_importance.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

importance.to_csv(
    "data/random_forest_feature_importance.csv",
    index=False
)

print("\nSaved:")
print("data/random_forest_feature_importance.csv")
print("data/random_forest_feature_importance.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)