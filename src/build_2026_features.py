import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

HISTORICAL_RACE_FILE = os.path.join(DATA_DIR, "race_results_2018_2025.csv")
HISTORICAL_QUALIFYING_FILE = os.path.join(
    DATA_DIR, "qualifying_results_2018_2025.csv"
)
HISTORICAL_STANDINGS_FILE = os.path.join(
    DATA_DIR, "driver_standings_2018_2025.csv"
)
HISTORICAL_CONSTRUCTOR_STANDINGS_FILE = os.path.join(
    DATA_DIR, "constructor_standings_2018_2025.csv"
)

RACE_2026_FILE = os.path.join(DATA_DIR, "race_results_2026.csv")
QUALIFYING_2026_FILE = os.path.join(DATA_DIR, "qualifying_results_2026.csv")
DRIVER_STANDINGS_2026_FILE = os.path.join(
    DATA_DIR, "driver_standings_2026.csv"
)
CONSTRUCTOR_STANDINGS_2026_FILE = os.path.join(
    DATA_DIR, "constructor_standings_2026.csv"
)

OUTPUT_2026_FEATURES = os.path.join(
    DATA_DIR, "feature_dataset_2026.csv"
)

OUTPUT_UPCOMING_FEATURES = os.path.join(
    DATA_DIR, "upcoming_race_features_2026.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(series):
    """
    Convert a pandas Series to numeric values.
    Invalid values become NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def calculate_dnf(status):
    """
    Approximate DNF indicator.

    This follows the same general idea used in the existing
    baseline feature engineering: statuses other than a
    classified finish are treated as non-finish events.

    Note:
    Some F1 statuses such as '+1 Lap' or '+2 Laps' are
    classified finishes and should NOT count as DNFs.
    """

    if pd.isna(status):
        return 1

    status = str(status).strip().lower()

    classified_finish_statuses = {
        "finished",
        "+1 lap",
        "+2 laps",
        "+3 laps",
        "+4 laps",
        "+5 laps",
        "+6 laps",
        "+7 laps",
        "+8 laps",
        "+9 laps",
        "+10 laps",
    }

    if status in classified_finish_statuses:
        return 0

    return 1


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING LEAKAGE-SAFE 2026 FEATURE DATASET")
print("=" * 70)


print("\nLoading historical race data...")
historical_races = pd.read_csv(HISTORICAL_RACE_FILE)

print("Loading historical qualifying data...")
historical_qualifying = pd.read_csv(HISTORICAL_QUALIFYING_FILE)

print("Loading historical driver standings...")
historical_driver_standings = pd.read_csv(
    HISTORICAL_STANDINGS_FILE
)

print("Loading historical constructor standings...")
historical_constructor_standings = pd.read_csv(
    HISTORICAL_CONSTRUCTOR_STANDINGS_FILE
)

print("\nLoading 2026 race data...")
races_2026 = pd.read_csv(RACE_2026_FILE)

print("Loading 2026 qualifying data...")
qualifying_2026 = pd.read_csv(QUALIFYING_2026_FILE)

print("Loading 2026 driver standings...")
driver_standings_2026 = pd.read_csv(
    DRIVER_STANDINGS_2026_FILE
)

print("Loading 2026 constructor standings...")
constructor_standings_2026 = pd.read_csv(
    CONSTRUCTOR_STANDINGS_2026_FILE
)


# ============================================================
# STANDARDIZE DATA TYPES
# ============================================================

race_columns = [
    historical_races,
    races_2026
]

for df in race_columns:
    df["season"] = safe_numeric(df["season"]).astype(int)
    df["round"] = safe_numeric(df["round"]).astype(int)
    df["grid"] = safe_numeric(df["grid"])
    df["position"] = safe_numeric(df["position"])
    df["points"] = safe_numeric(df["points"])
    df["laps"] = safe_numeric(df["laps"])


qualifying_columns = [
    historical_qualifying,
    qualifying_2026
]

for df in qualifying_columns:
    df["season"] = safe_numeric(df["season"]).astype(int)
    df["round"] = safe_numeric(df["round"]).astype(int)
    df["qualifying_position"] = safe_numeric(
        df["qualifying_position"]
    )


# ============================================================
# COMBINE HISTORICAL + 2026 RACE DATA
# ============================================================

print("\nCombining historical and 2026 race data...")

all_races = pd.concat(
    [
        historical_races,
        races_2026
    ],
    ignore_index=True
)

all_races = all_races.sort_values(
    [
        "driver_id",
        "season",
        "round"
    ]
).reset_index(drop=True)

print("Combined race rows:", len(all_races))

print(
    "Historical seasons:",
    sorted(historical_races["season"].unique())
)

print(
    "2026 rounds:",
    sorted(races_2026["round"].unique())
)


# ============================================================
# COMBINE HISTORICAL + 2026 QUALIFYING
# ============================================================

print("\nCombining qualifying data...")

all_qualifying = pd.concat(
    [
        historical_qualifying,
        qualifying_2026
    ],
    ignore_index=True
)

all_qualifying = all_qualifying[
    [
        "season",
        "round",
        "driver_id",
        "qualifying_position"
    ]
].drop_duplicates(
    subset=[
        "season",
        "round",
        "driver_id"
    ]
)

all_qualifying = all_qualifying.sort_values(
    [
        "season",
        "round",
        "driver_id"
    ]
)

print("Combined qualifying rows:", len(all_qualifying))


# ============================================================
# MERGE QUALIFYING INTO RACE DATA
# ============================================================

print("\nMerging qualifying positions...")

all_races = all_races.merge(
    all_qualifying,
    on=[
        "season",
        "round",
        "driver_id"
    ],
    how="left",
    suffixes=("", "_qualifying")
)

# If the original race dataset already contained a
# qualifying_position column, use the merged value where needed.
if "qualifying_position_qualifying" in all_races.columns:

    all_races["qualifying_position"] = all_races[
        "qualifying_position_qualifying"
    ].combine_first(
        all_races.get("qualifying_position")
    )

    all_races.drop(
        columns=["qualifying_position_qualifying"],
        inplace=True
    )


# ============================================================
# DRIVER DNF FLAG
# ============================================================

print("\nCalculating DNF indicator...")

all_races["is_dnf"] = all_races["status"].apply(
    calculate_dnf
)


# ============================================================
# DRIVER ROLLING FEATURES
# ============================================================

print("\nCalculating driver rolling features...")

driver_group = all_races.groupby("driver_id", group_keys=False)


# ------------------------------------------------------------
# IMPORTANT:
# shift(1) means the current race is NOT included.
#
# Example:
#
# Round 5 feature
# = performance from rounds 1-4
#
# This prevents target leakage.
# ------------------------------------------------------------

all_races["driver_avg_finish_last_3"] = (
    driver_group["position"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

all_races["driver_avg_finish_last_5"] = (
    driver_group["position"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)

all_races["driver_avg_points_last_3"] = (
    driver_group["points"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

all_races["driver_avg_points_last_5"] = (
    driver_group["points"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=1
        ).mean()
    )

)

all_races["driver_avg_qualifying_last_3"] = (
    driver_group["qualifying_position"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

all_races["driver_avg_qualifying_last_5"] = (
    driver_group["qualifying_position"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)

all_races["driver_dnf_rate"] = (
    driver_group["is_dnf"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)


# ============================================================
# CONSTRUCTOR-LEVEL RACE DATA
# ============================================================

print("\nCalculating constructor race-level performance...")

constructor_race = (
    all_races
    .groupby(
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

constructor_race = constructor_race.sort_values(
    [
        "constructor",
        "season",
        "round"
    ]
).reset_index(drop=True)


# ============================================================
# CONSTRUCTOR ROLLING FEATURES
# ============================================================

print("\nCalculating constructor rolling features...")

constructor_group = constructor_race.groupby(
    "constructor",
    group_keys=False
)

constructor_race["constructor_avg_finish_last_3"] = (
    constructor_group["constructor_avg_finish"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

constructor_race["constructor_avg_points_last_3"] = (
    constructor_group["constructor_total_points"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

constructor_race["constructor_avg_qualifying_last_3"] = (
    constructor_group["constructor_avg_qualifying"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)


# ============================================================
# MERGE CONSTRUCTOR FEATURES BACK
# ============================================================

constructor_feature_columns = [
    "season",
    "round",
    "constructor",
    "constructor_avg_finish_last_3",
    "constructor_avg_points_last_3",
    "constructor_avg_qualifying_last_3"
]

all_races = all_races.merge(
    constructor_race[
        constructor_feature_columns
    ],
    on=[
        "season",
        "round",
        "constructor"
    ],
    how="left"
)


# ============================================================
# CHAMPIONSHIP STANDINGS
# ============================================================

print("\nPreparing championship standings...")

all_driver_standings = pd.concat(
    [
        historical_driver_standings,
        driver_standings_2026
    ],
    ignore_index=True
)

all_constructor_standings = pd.concat(
    [
        historical_constructor_standings,
        constructor_standings_2026
    ],
    ignore_index=True
)

all_driver_standings["season"] = safe_numeric(
    all_driver_standings["season"]
).astype(int)

all_driver_standings["round"] = safe_numeric(
    all_driver_standings["round"]
).astype(int)

all_constructor_standings["season"] = safe_numeric(
    all_constructor_standings["season"]
).astype(int)

all_constructor_standings["round"] = safe_numeric(
    all_constructor_standings["round"]
).astype(int)


# ============================================================
# DRIVER CHAMPIONSHIP FEATURES
# ============================================================

print("\nBuilding driver championship history...")

driver_standings_features = all_driver_standings[
    [
        "season",
        "round",
        "driver_id",
        "championship_position",
        "championship_points",
        "wins"
    ]
].copy()

driver_standings_features = driver_standings_features.rename(
    columns={
        "championship_position":
            "driver_championship_position_before",
        "championship_points":
            "driver_championship_points_before",
        "wins":
            "driver_wins_before"
    }
)

# The standings after Round N describe the championship AFTER
# Round N. Therefore, for Round N+1, those values are valid.
#
# Shift by one round within each driver/season.

driver_standings_features = driver_standings_features.sort_values(
    [
        "driver_id",
        "season",
        "round"
    ]
)

driver_standings_features[
    [
        "driver_championship_position_before",
        "driver_championship_points_before",
        "driver_wins_before"
    ]
] = (
    driver_standings_features
    .groupby(
        [
            "driver_id",
            "season"
        ]
    )[
        [
            "driver_championship_position_before",
            "driver_championship_points_before",
            "driver_wins_before"
        ]
    ]
    .shift(1)
)


# ============================================================
# CONSTRUCTOR CHAMPIONSHIP FEATURES
# ============================================================

print("\nBuilding constructor championship history...")

constructor_standings_features = all_constructor_standings[
    [
        "season",
        "round",
        "constructor",
        "championship_position",
        "championship_points",
        "wins"
    ]
].copy()

constructor_standings_features = (
    constructor_standings_features.rename(
        columns={
            "championship_position":
                "constructor_championship_position_before",
            "championship_points":
                "constructor_championship_points_before",
            "wins":
                "constructor_wins_before"
        }
    )
)

constructor_standings_features = (
    constructor_standings_features.sort_values(
        [
            "constructor",
            "season",
            "round"
        ]
    )
)

constructor_standings_features[
    [
        "constructor_championship_position_before",
        "constructor_championship_points_before",
        "constructor_wins_before"
    ]
] = (
    constructor_standings_features
    .groupby(
        [
            "constructor",
            "season"
        ]
    )[
        [
            "constructor_championship_position_before",
            "constructor_championship_points_before",
            "constructor_wins_before"
        ]
    ]
    .shift(1)
)


# ============================================================
# MERGE CHAMPIONSHIP FEATURES
# ============================================================

all_races = all_races.merge(
    driver_standings_features[
        [
            "season",
            "round",
            "driver_id",
            "driver_championship_position_before",
            "driver_championship_points_before",
            "driver_wins_before"
        ]
    ],
    on=[
        "season",
        "round",
        "driver_id"
    ],
    how="left"
)

all_races = all_races.merge(
    constructor_standings_features[
        [
            "season",
            "round",
            "constructor",
            "constructor_championship_position_before",
            "constructor_championship_points_before",
            "constructor_wins_before"
        ]
    ],
    on=[
        "season",
        "round",
        "constructor"
    ],
    how="left"
)


# ============================================================
# SELECT FINAL FEATURE COLUMNS
# ============================================================

feature_columns = [
    "season",
    "round",
    "race",
    "circuit",

    "driver_id",
    "driver",
    "constructor",

    "grid",
    "position",
    "points",
    "status",
    "laps",
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
    "constructor_wins_before"
]

# Keep only columns that actually exist.
feature_columns = [
    col for col in feature_columns
    if col in all_races.columns
]

feature_dataset_2026 = all_races[
    all_races["season"] == 2026
][feature_columns].copy()

feature_dataset_2026 = feature_dataset_2026.sort_values(
    [
        "round",
        "driver_id"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE COMPLETED 2026 FEATURE DATASET
# ============================================================

feature_dataset_2026.to_csv(
    OUTPUT_2026_FEATURES,
    index=False
)


# ============================================================
# IDENTIFY NEXT RACE
# ============================================================

completed_rounds = sorted(
    races_2026["round"].dropna().astype(int).unique()
)

if len(completed_rounds) == 0:

    print("\nNo completed 2026 race rounds found.")
    print("Cannot create upcoming race feature rows.")

else:

    latest_completed_round = max(completed_rounds)

    next_round = latest_completed_round + 1

    print("\nLatest completed 2026 round:", latest_completed_round)
    print("Next predicted round:", next_round)


    # --------------------------------------------------------
    # Get drivers/constructors from the latest completed race.
    # These are the drivers we currently know about.
    # --------------------------------------------------------

    latest_race = races_2026[
        races_2026["round"] == latest_completed_round
    ].copy()

    upcoming_drivers = latest_race[
        [
            "driver_id",
            "driver",
            "constructor"
        ]
    ].drop_duplicates(
        subset=["driver_id"]
    )


    # --------------------------------------------------------
    # Get previous-round driver championship standings.
    #
    # These are the standings available before the next race.
    # --------------------------------------------------------

    previous_driver_standings = driver_standings_2026[
        driver_standings_2026["round"]
        == latest_completed_round
    ].copy()

    previous_driver_standings = (
        previous_driver_standings[
            [
                "driver_id",
                "championship_position",
                "championship_points",
                "wins"
            ]
        ]
        .rename(
            columns={
                "championship_position":
                    "driver_championship_position_before",
                "championship_points":
                    "driver_championship_points_before",
                "wins":
                    "driver_wins_before"
            }
        )
    )


    # --------------------------------------------------------
    # Previous constructor standings
    # --------------------------------------------------------

    previous_constructor_standings = (
        constructor_standings_2026[
            constructor_standings_2026["round"]
            == latest_completed_round
        ]
        [
            [
                "constructor",
                "championship_position",
                "championship_points",
                "wins"
            ]
        ]
        .rename(
            columns={
                "championship_position":
                    "constructor_championship_position_before",
                "championship_points":
                    "constructor_championship_points_before",
                "wins":
                    "constructor_wins_before"
            }
        )
    )


    # --------------------------------------------------------
    # Get the latest historical feature values for each driver.
    #
    # These already use shift(1), so they represent information
    # available before the corresponding completed race.
    #
    # For the next race, however, we need the rolling statistics
    # INCLUDING the latest completed race.
    # --------------------------------------------------------

    driver_history = all_races[
        all_races["season"] == 2026
    ].copy()

    driver_history = driver_history[
        driver_history["round"] <= latest_completed_round
    ]

    driver_history = driver_history.sort_values(
        [
            "driver_id",
            "season",
            "round"
        ]
    )


    # --------------------------------------------------------
    # Calculate next-race driver features directly from all
    # completed 2026 races.
    #
    # Historical 2018–2025 performance is already represented
    # by the combined dataset, but for drivers with 2026 history
    # we use their most recent completed races.
    # --------------------------------------------------------

    def latest_driver_feature(group):

        group = group.sort_values(
            ["season", "round"]
        )

        return pd.Series(
            {
                "driver_avg_finish_last_3":
                    group["position"].tail(3).mean(),

                "driver_avg_finish_last_5":
                    group["position"].tail(5).mean(),

                "driver_avg_points_last_3":
                    group["points"].tail(3).mean(),

                "driver_avg_points_last_5":
                    group["points"].tail(5).mean(),

                "driver_avg_qualifying_last_3":
                    group["qualifying_position"]
                    .tail(3)
                    .mean(),

                "driver_avg_qualifying_last_5":
                    group["qualifying_position"]
                    .tail(5)
                    .mean(),

                "driver_dnf_rate":
                    group["is_dnf"]
                    .tail(5)
                    .mean()
            }
        )


    next_driver_features = (
        driver_history
        .groupby("driver_id")
        .apply(
            latest_driver_feature,
            include_groups=False
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Constructor features for next race
    # --------------------------------------------------------

    constructor_history = (
        all_races[
            all_races["season"] == 2026
        ]
        .copy()
    )

    constructor_history = constructor_history[
        constructor_history["round"]
        <= latest_completed_round
    ]

    constructor_history = (
        constructor_history
        .groupby(
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


    def latest_constructor_feature(group):

        group = group.sort_values(
            ["season", "round"]
        )

        return pd.Series(
            {
                "constructor_avg_finish_last_3":
                    group["constructor_avg_finish"]
                    .tail(3)
                    .mean(),

                "constructor_avg_points_last_3":
                    group["constructor_total_points"]
                    .tail(3)
                    .mean(),

                "constructor_avg_qualifying_last_3":
                    group["constructor_avg_qualifying"]
                    .tail(3)
                    .mean()
            }
        )


    next_constructor_features = (
        constructor_history
        .groupby("constructor")
        .apply(
            latest_constructor_feature,
            include_groups=False
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Merge everything
    # --------------------------------------------------------

    upcoming_features = upcoming_drivers.merge(
        next_driver_features,
        on="driver_id",
        how="left"
    )

    upcoming_features = upcoming_features.merge(
        next_constructor_features,
        on="constructor",
        how="left"
    )

    upcoming_features = upcoming_features.merge(
        previous_driver_standings,
        on="driver_id",
        how="left"
    )

    upcoming_features = upcoming_features.merge(
        previous_constructor_standings,
        on="constructor",
        how="left"
    )


    # --------------------------------------------------------
    # Add prediction metadata
    # --------------------------------------------------------

    upcoming_features.insert(
        0,
        "season",
        2026
    )

    upcoming_features.insert(
        1,
        "round",
        next_round
    )


    # --------------------------------------------------------
    # Grid is intentionally UNKNOWN here.
    #
    # We do NOT copy a future grid value because that would be
    # future-data leakage.
    #
    # If we later decide to predict AFTER qualifying, this column
    # can be filled with the actual qualifying/grid data.
    # --------------------------------------------------------

    upcoming_features["grid"] = np.nan

    upcoming_features = upcoming_features[
        [
            "season",
            "round",
            "driver_id",
            "driver",
            "constructor",
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
    ]


    upcoming_features = upcoming_features.sort_values(
        "driver_championship_position_before"
    ).reset_index(drop=True)


    upcoming_features.to_csv(
        OUTPUT_UPCOMING_FEATURES,
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("2026 FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("\nSaved:")
print(
    f"  {OUTPUT_2026_FEATURES}"
)

if len(completed_rounds) > 0:
    print(
        f"  {OUTPUT_UPCOMING_FEATURES}"
    )

print("\n2026 feature dataset shape:")
print(
    feature_dataset_2026.shape
)

print("\n2026 rounds represented:")
print(
    sorted(
        feature_dataset_2026["round"]
        .unique()
    )
)

print("\nFeature columns:")
for column in feature_dataset_2026.columns:
    print(" -", column)

print("\nImportant:")
print(
    "The upcoming-race grid is intentionally left as NaN "
    "because future grid position is unknown."
)

print("\nDONE")