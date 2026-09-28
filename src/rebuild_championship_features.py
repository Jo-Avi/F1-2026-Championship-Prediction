import pandas as pd
import numpy as np


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/feature_dataset_v2.csv"
RACE_FILE = "data/race_results_2018_2025.csv"
SPRINT_FILE = "data/sprint_results_2018_2025.csv"

OUTPUT_FILE = "data/feature_dataset_v3.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)
race = pd.read_csv(RACE_FILE)
sprint = pd.read_csv(SPRINT_FILE)


print("=" * 60)
print("REBUILDING CHAMPIONSHIP FEATURES")
print("=" * 60)

print(f"\nFeature dataset: {df.shape}")
print(f"Race results:    {race.shape}")
print(f"Sprint results:  {sprint.shape}")


# ============================================================
# CLEAN TYPES
# ============================================================

for data in [df, race, sprint]:
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    data["round"] = pd.to_numeric(data["round"], errors="coerce")
    data["points"] = pd.to_numeric(data["points"], errors="coerce").fillna(0)


# ============================================================
# DRIVER CHAMPIONSHIP HISTORY
# ============================================================

print("\nCalculating driver championship history...")


# Race points by driver and round
driver_race_points = (
    race.groupby(
        ["season", "round", "driver_id"],
        as_index=False
    )["points"]
    .sum()
    .rename(columns={"points": "race_points"})
)


# Sprint points by driver and round
if not sprint.empty:

    sprint["points"] = pd.to_numeric(
        sprint["points"],
        errors="coerce"
    ).fillna(0)

    driver_sprint_points = (
        sprint.groupby(
            ["season", "round", "driver_id"],
            as_index=False
        )["points"]
        .sum()
        .rename(columns={"points": "sprint_points"})
    )

else:
    driver_sprint_points = pd.DataFrame(
        columns=["season", "round", "driver_id", "sprint_points"]
    )


# Combine race + sprint points
driver_points = driver_race_points.merge(
    driver_sprint_points,
    on=["season", "round", "driver_id"],
    how="left"
)

driver_points["sprint_points"] = (
    driver_points["sprint_points"].fillna(0)
)

driver_points["total_round_points"] = (
    driver_points["race_points"]
    + driver_points["sprint_points"]
)


# ------------------------------------------------------------
# Driver wins
# ------------------------------------------------------------

# A win is a Grand Prix race finishing position of 1.
driver_wins = (
    race.assign(
        race_win=(
            pd.to_numeric(
                race["position"],
                errors="coerce"
            ) == 1
        ).astype(int)
    )
    .groupby(
        ["season", "round", "driver_id"],
        as_index=False
    )["race_win"]
    .sum()
)


driver_points = driver_points.merge(
    driver_wins,
    on=["season", "round", "driver_id"],
    how="left"
)

driver_points["race_win"] = (
    driver_points["race_win"].fillna(0)
)


# ============================================================
# CALCULATE PREVIOUS CHAMPIONSHIP STATE
# ============================================================

driver_points = driver_points.sort_values(
    ["season", "driver_id", "round"]
)


# Cumulative points including current round
driver_points["cumulative_points"] = (
    driver_points
    .groupby(["season", "driver_id"])["total_round_points"]
    .cumsum()
)


# Cumulative wins including current round
driver_points["cumulative_wins"] = (
    driver_points
    .groupby(["season", "driver_id"])["race_win"]
    .cumsum()
)


# Shift so current race result is NOT included
driver_points["driver_championship_points_before"] = (
    driver_points
    .groupby(["season", "driver_id"])["cumulative_points"]
    .shift(1)
)


driver_points["driver_wins_before"] = (
    driver_points
    .groupby(["season", "driver_id"])["cumulative_wins"]
    .shift(1)
)


# ============================================================
# CALCULATE CHAMPIONSHIP POSITION
# ============================================================

# We need the championship table BEFORE each round.
#
# First calculate championship position using cumulative points,
# then shift it so the current round is excluded.

driver_points["championship_position"] = (
    driver_points
    .groupby(["season", "round"])["cumulative_points"]
    .rank(
        method="min",
        ascending=False
    )
)


driver_points["driver_championship_position_before"] = (
    driver_points
    .groupby(["season", "driver_id"])["championship_position"]
    .shift(1)
)


# ============================================================
# CONSTRUCTOR CHAMPIONSHIP HISTORY
# ============================================================

print("Calculating constructor championship history...")


# Add constructor information to race results
race_constructor = race[
    [
        "season",
        "round",
        "constructor",
        "points",
        "position"
    ]
].copy()


race_constructor["position"] = pd.to_numeric(
    race_constructor["position"],
    errors="coerce"
)


# Race points per constructor per round
constructor_race_points = (
    race_constructor
    .groupby(
        ["season", "round", "constructor"],
        as_index=False
    )["points"]
    .sum()
    .rename(columns={"points": "race_points"})
)


# Constructor wins
constructor_wins = (
    race_constructor.assign(
        race_win=(
            race_constructor["position"] == 1
        ).astype(int)
    )
    .groupby(
        ["season", "round", "constructor"],
        as_index=False
    )["race_win"]
    .sum()
)


constructor_points = constructor_race_points.merge(
    constructor_wins,
    on=["season", "round", "constructor"],
    how="left"
)


constructor_points["race_win"] = (
    constructor_points["race_win"].fillna(0)
)


# ------------------------------------------------------------
# Constructor sprint points
# ------------------------------------------------------------

if not sprint.empty:

    sprint_constructor = sprint[
        [
            "season",
            "round",
            "constructor",
            "points"
        ]
    ].copy()

    sprint_constructor["points"] = pd.to_numeric(
        sprint_constructor["points"],
        errors="coerce"
    ).fillna(0)

    constructor_sprint_points = (
        sprint_constructor
        .groupby(
            ["season", "round", "constructor"],
            as_index=False
        )["points"]
        .sum()
        .rename(columns={"points": "sprint_points"})
    )

    constructor_points = constructor_points.merge(
        constructor_sprint_points,
        on=["season", "round", "constructor"],
        how="left"
    )

else:

    constructor_points["sprint_points"] = 0


constructor_points["sprint_points"] = (
    constructor_points["sprint_points"].fillna(0)
)


constructor_points["total_round_points"] = (
    constructor_points["race_points"]
    + constructor_points["sprint_points"]
)


# ============================================================
# CONSTRUCTOR CUMULATIVE HISTORY
# ============================================================

constructor_points = constructor_points.sort_values(
    ["season", "constructor", "round"]
)


constructor_points["cumulative_points"] = (
    constructor_points
    .groupby(
        ["season", "constructor"]
    )["total_round_points"]
    .cumsum()
)


constructor_points["cumulative_wins"] = (
    constructor_points
    .groupby(
        ["season", "constructor"]
    )["race_win"]
    .cumsum()
)


# Shift previous state
constructor_points["constructor_championship_points_before"] = (
    constructor_points
    .groupby(
        ["season", "constructor"]
    )["cumulative_points"]
    .shift(1)
)


constructor_points["constructor_wins_before"] = (
    constructor_points
    .groupby(
        ["season", "constructor"]
    )["cumulative_wins"]
    .shift(1)
)


# Championship position
constructor_points["championship_position"] = (
    constructor_points
    .groupby(["season", "round"])["cumulative_points"]
    .rank(
        method="min",
        ascending=False
    )
)


constructor_points["constructor_championship_position_before"] = (
    constructor_points
    .groupby(
        ["season", "constructor"]
    )["championship_position"]
    .shift(1)
)


# ============================================================
# MERGE DRIVER FEATURES
# ============================================================

driver_features = driver_points[
    [
        "season",
        "round",
        "driver_id",
        "driver_championship_position_before",
        "driver_championship_points_before",
        "driver_wins_before"
    ]
].copy()


# V2 does not currently contain driver_id,
# so we use driver name from the original data.

driver_lookup = race[
    [
        "season",
        "round",
        "driver_id",
        "driver"
    ]
].drop_duplicates()


driver_features = driver_features.merge(
    driver_lookup,
    on=["season", "round", "driver_id"],
    how="left"
)


df = df.merge(
    driver_features[
        [
            "season",
            "round",
            "driver",
            "driver_championship_position_before",
            "driver_championship_points_before",
            "driver_wins_before"
        ]
    ],
    on=["season", "round", "driver"],
    how="left"
)


# ============================================================
# MERGE CONSTRUCTOR FEATURES
# ============================================================

constructor_features = constructor_points[
    [
        "season",
        "round",
        "constructor",
        "constructor_championship_position_before",
        "constructor_championship_points_before",
        "constructor_wins_before"
    ]
].copy()


df = df.merge(
    constructor_features,
    on=["season", "round", "constructor"],
    how="left"
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CHAMPIONSHIP FEATURES REBUILT")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

championship_features = [
    "driver_championship_position_before",
    "driver_championship_points_before",
    "driver_wins_before",
    "constructor_championship_position_before",
    "constructor_championship_points_before",
    "constructor_wins_before"
]

print("\nMissing values:")
print(df[championship_features].isna().sum())


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
            "driver_wins_before",
            "constructor_championship_position_before",
            "constructor_championship_points_before",
            "constructor_wins_before",
            "position"
        ]
    ].head(20)
)


print(f"\nSaved to: {OUTPUT_FILE}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)