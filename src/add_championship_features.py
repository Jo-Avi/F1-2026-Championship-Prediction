import pandas as pd
import os

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv(
    "data/feature_dataset_v2.csv"
)

driver_standings = pd.read_csv(
    "data/driver_standings_2018_2025.csv"
)

constructor_standings = pd.read_csv(
    "data/constructor_standings_2018_2025.csv"
)


# ==========================================================
# 2. PREPARE DRIVER STANDINGS
# ==========================================================

# The standings for Round N represent the championship
# AFTER Round N.
#
# Therefore, standings from Round N-1 represent the
# championship state BEFORE Round N.

driver_standings["target_round"] = (
    driver_standings["round"] + 1
)

driver_features = driver_standings[
    [
        "season",
        "target_round",
        "driver",
        "position",
        "points",
        "wins"
    ]
].copy()

driver_features.rename(
    columns={
        "target_round":
            "round",

        "position":
            "driver_championship_position_before",

        "points":
            "driver_championship_points_before",

        "wins":
            "driver_wins_before"
    },
    inplace=True
)


# ==========================================================
# 3. MERGE DRIVER CHAMPIONSHIP FEATURES
# ==========================================================

df = df.merge(
    driver_features,

    on=[
        "season",
        "round",
        "driver"
    ],

    how="left"
)


# ==========================================================
# 4. PREPARE CONSTRUCTOR STANDINGS
# ==========================================================

constructor_standings["target_round"] = (
    constructor_standings["round"] + 1
)

constructor_features = constructor_standings[
    [
        "season",
        "target_round",
        "constructor",
        "position",
        "points",
        "wins"
    ]
].copy()

constructor_features.rename(
    columns={
        "target_round":
            "round",

        "position":
            "constructor_championship_position_before",

        "points":
            "constructor_championship_points_before",

        "wins":
            "constructor_wins_before"
    },
    inplace=True
)


# ==========================================================
# 5. MERGE CONSTRUCTOR FEATURES
# ==========================================================

df = df.merge(
    constructor_features,

    on=[
        "season",
        "round",
        "constructor"
    ],

    how="left"
)


# ==========================================================
# 6. SAVE V3 DATASET
# ==========================================================

os.makedirs(
    "data",
    exist_ok=True
)

df.to_csv(
    "data/feature_dataset_v3.csv",
    index=False
)


# ==========================================================
# 7. DISPLAY RESULTS
# ==========================================================

print("\n==========================================")
print("Championship Features Added!")
print("==========================================")

print("\nDataset shape:")
print(df.shape)


features = [
    "driver_championship_position_before",
    "driver_championship_points_before",
    "driver_wins_before",
    "constructor_championship_position_before",
    "constructor_championship_points_before",
    "constructor_wins_before"
]


print("\nChampionship features:")

for feature in features:
    print("-", feature)


print("\nSample:")

print(
    df[
        [
            "season",
            "round",
            "driver",
            "constructor",
            "driver_championship_position_before",
            "driver_championship_points_before",
            "constructor_championship_position_before",
            "constructor_championship_points_before",
            "position"
        ]
    ].head(15)
)


print("\nMissing values:")

print(
    df[features].isnull().sum()
)


print("\nSaved to:")
print("data/feature_dataset_v3.csv")