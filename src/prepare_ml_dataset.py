import pandas as pd
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/feature_dataset_v3.csv"
OUTPUT_FILE = "data/ml_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PREPARING ML DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal dataset shape: {df.shape}")


# ============================================================
# DEFINE FEATURES
# ============================================================

feature_columns = [
    # Race / qualifying information
    "grid",
    "qualifying_position",

    # Driver historical performance
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_avg_points_last_3",
    "driver_avg_points_last_5",
    "driver_avg_qualifying_last_3",
    "driver_avg_qualifying_last_5",
    "driver_dnf_rate",

    # Constructor historical performance
    "constructor_avg_finish_last_3",
    "constructor_avg_points_last_3",
    "constructor_avg_qualifying_last_3",

    # Championship position before race
    "driver_championship_position_before",
    "driver_championship_points_before",
    "driver_wins_before",

    "constructor_championship_position_before",
    "constructor_championship_points_before",
    "constructor_wins_before",
]


target_column = "position"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = feature_columns + [
    target_column,
    "season",
    "round",
    "driver",
    "constructor",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    for column in missing_columns:
        print(f" - {column}")
    raise SystemExit(1)


# ============================================================
# CREATE MODELING DATASET
# ============================================================

ml_columns = [
    "season",
    "round",
    "driver",
    "constructor",
] + feature_columns + [target_column]

ml_df = df[ml_columns].copy()


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = feature_columns + [target_column]

for column in numeric_columns:
    ml_df[column] = pd.to_numeric(
        ml_df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE ROWS WITHOUT TARGET
# ============================================================

before_target_drop = len(ml_df)

ml_df = ml_df.dropna(
    subset=[target_column]
)

after_target_drop = len(ml_df)

print(
    f"\nRows removed because target was missing: "
    f"{before_target_drop - after_target_drop}"
)


# ============================================================
# MISSING VALUE REPORT
# ============================================================

print("\nMissing values before imputation:")

missing_report = (
    ml_df[feature_columns]
    .isnull()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing_report[
        missing_report > 0
    ]
)


# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs("data", exist_ok=True)

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("ML DATASET CREATED")
print("=" * 60)

print(f"\nShape: {ml_df.shape}")

print(f"\nSeasons:")
print(sorted(ml_df["season"].unique()))

print(f"\nNumber of drivers: {ml_df['driver'].nunique()}")

print(
    f"Number of constructors: "
    f"{ml_df['constructor'].nunique()}"
)

print(
    f"Number of races: "
    f"{ml_df[['season', 'round']].drop_duplicates().shape[0]}"
)

print("\nTarget statistics:")
print(ml_df[target_column].describe())

print("\nFeature columns:")

for column in feature_columns:
    print(f" - {column}")

print(f"\nSaved to: {OUTPUT_FILE}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)