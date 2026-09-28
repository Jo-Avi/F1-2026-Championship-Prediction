import pandas as pd
import numpy as np
import os

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv(
    "data/master_race_dataset.csv"
)

# Sort chronologically
df = df.sort_values(
    ["season", "round", "driver_id"]
).reset_index(drop=True)


# ==========================================================
# 2. DRIVER FEATURES
# ==========================================================

def add_driver_features(group):

    group = group.sort_values(
        ["season", "round"]
    ).copy()

    # ------------------------------------------------------
    # Previous race finishing position
    # ------------------------------------------------------

    group["driver_avg_finish_last_3"] = (
        group["position"]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )

    group["driver_avg_finish_last_5"] = (
        group["position"]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )

    # ------------------------------------------------------
    # Previous race points
    # ------------------------------------------------------

    group["driver_avg_points_last_3"] = (
        group["points"]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )

    group["driver_avg_points_last_5"] = (
        group["points"]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )

    # ------------------------------------------------------
    # Qualifying performance
    # ------------------------------------------------------

    group["driver_avg_qualifying_last_3"] = (
        group["qualifying_position"]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )

    group["driver_avg_qualifying_last_5"] = (
        group["qualifying_position"]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )

    # ------------------------------------------------------
    # DNF rate
    # ------------------------------------------------------

    finished_statuses = [
        "finished",
        "+1 lap",
        "+2 laps",
        "+3 laps",
        "+4 laps",
        "+5 laps"
    ]

    group["is_dnf"] = (
        ~group["status"]
        .astype(str)
        .str.lower()
        .isin(finished_statuses)
    )

    group["driver_dnf_rate"] = (
        group["is_dnf"]
        .shift(1)
        .rolling(10, min_periods=1)
        .mean()
    )

    return group


df = (
    df.groupby(
        "driver_id",
        group_keys=False
    )
    .apply(add_driver_features)
    .reset_index(drop=True)
)


# ==========================================================
# 3. CONSTRUCTOR / TEAM FEATURES
# ==========================================================

# First create one row per constructor per race.
#
# This prevents the two drivers of the same team from
# being counted as separate constructor observations.

constructor_race = (
    df.groupby(
        [
            "season",
            "round",
            "constructor"
        ],
        as_index=False
    )
    .agg(
        constructor_avg_finish=(
            "position",
            "mean"
        ),

        constructor_total_points=(
            "points",
            "sum"
        ),

        constructor_avg_qualifying=(
            "qualifying_position",
            "mean"
        )
    )
)


# ==========================================================
# Sort constructor data chronologically
# ==========================================================

constructor_race = constructor_race.sort_values(
    [
        "constructor",
        "season",
        "round"
    ]
).reset_index(drop=True)


# ==========================================================
# Previous constructor performance
# ==========================================================

constructor_race[
    "constructor_avg_finish_last_3"
] = (
    constructor_race
    .groupby("constructor")["constructor_avg_finish"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )
)


constructor_race[
    "constructor_avg_points_last_3"
] = (
    constructor_race
    .groupby("constructor")["constructor_total_points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )
)


constructor_race[
    "constructor_avg_qualifying_last_3"
] = (
    constructor_race
    .groupby("constructor")["constructor_avg_qualifying"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )
)


# ==========================================================
# 4. MERGE CONSTRUCTOR FEATURES
# ==========================================================

df = df.merge(
    constructor_race[
        [
            "season",
            "round",
            "constructor",

            "constructor_avg_finish_last_3",
            "constructor_avg_points_last_3",
            "constructor_avg_qualifying_last_3"
        ]
    ],

    on=[
        "season",
        "round",
        "constructor"
    ],

    how="left"
)


# ==========================================================
# 5. REMOVE TEMPORARY COLUMN
# ==========================================================

if "is_dnf" in df.columns:

    df.drop(
        columns=["is_dnf"],
        inplace=True
    )


# ==========================================================
# 6. SAVE DATASET
# ==========================================================

os.makedirs(
    "data",
    exist_ok=True
)

df.to_csv(
    "data/feature_dataset_v2.csv",
    index=False
)


# ==========================================================
# 7. DISPLAY INFORMATION
# ==========================================================

print("\n==========================================")
print("Feature Engineering V2 Complete!")
print("==========================================")

print("\nDataset shape:")
print(df.shape)

print("\nFeature columns:")

features = [
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_avg_points_last_3",
    "driver_avg_points_last_5",
    "driver_avg_qualifying_last_3",
    "driver_avg_qualifying_last_5",
    "driver_dnf_rate",

    "constructor_avg_finish_last_3",
    "constructor_avg_points_last_3",
    "constructor_avg_qualifying_last_3"
]

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
            "driver_avg_finish_last_3",
            "driver_avg_points_last_3",
            "driver_avg_qualifying_last_3",
            "constructor_avg_finish_last_3",
            "constructor_avg_points_last_3",
            "position"
        ]
    ].head(10)
)


print("\nMissing values in new features:")

print(
    df[features].isnull().sum()
)


print("\nSaved to:")
print("data/feature_dataset_v2.csv")