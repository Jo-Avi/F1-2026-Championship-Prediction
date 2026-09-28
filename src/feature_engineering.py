import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# Load master dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/master_race_dataset.csv"
)

# Sort chronologically
df = df.sort_values(
    ["season", "round"]
).reset_index(drop=True)


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def calculate_driver_features(group):

    group = group.sort_values(
        ["season", "round"]
    ).copy()

    # ----------------------------------------------
    # Previous race finish positions
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Previous race points
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Previous qualifying performance
    # ----------------------------------------------

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

    # ----------------------------------------------
    # DNF rate
    # ----------------------------------------------

    group["is_dnf"] = (
        ~group["status"].str.lower().isin(
            [
                "finished",
                "+1 lap",
                "+2 laps",
                "+3 laps",
                "+4 laps",
                "+5 laps"
            ]
        )
    )

    group["driver_dnf_rate"] = (
        group["is_dnf"]
        .shift(1)
        .rolling(10, min_periods=1)
        .mean()
    )

    return group


# --------------------------------------------------
# Apply driver features
# --------------------------------------------------

df = (
    df.groupby("driver_id", group_keys=False)
      .apply(calculate_driver_features)
      .reset_index(drop=True)
)


# --------------------------------------------------
# Remove helper column
# --------------------------------------------------

df.drop(
    columns=["is_dnf"],
    inplace=True
)


# --------------------------------------------------
# Save
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/feature_dataset_v1.csv",
    index=False
)

print("\nFeature engineering complete!")

print("\nDataset shape:")
print(df.shape)

print("\nNew features:")

features = [
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_avg_points_last_3",
    "driver_avg_points_last_5",
    "driver_avg_qualifying_last_3",
    "driver_avg_qualifying_last_5",
    "driver_dnf_rate"
]

print(features)

print("\nFirst 10 rows:")
print(
    df[
        [
            "season",
            "round",
            "driver",
            "driver_avg_finish_last_3",
            "driver_avg_points_last_3",
            "driver_avg_qualifying_last_3",
            "driver_dnf_rate",
            "position"
        ]
    ].head(10)
)