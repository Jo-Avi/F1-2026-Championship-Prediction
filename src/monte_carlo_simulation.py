import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SIMULATIONS = 10_000

CURRENT_ROUND = 15
REMAINING_ROUNDS = list(range(16, 24))

# Singapore is the only remaining Sprint weekend.
SPRINT_ROUNDS = {17}

RANDOM_SEED = 42

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "monte_carlo_2026_results.csv"
)

# Historical out-of-sample performance of the
# pre-qualifying model on 2025:
#
# MAE  ~= 3.85 positions
# RMSE ~= 4.76 positions
#
# We use MAE as the baseline race-error scale instead
# of treating Random Forest tree spread as true uncertainty.
HISTORICAL_ERROR_SCALE = 3.85

# Small multiplier used for calibration experiments.
# Keep at 1.0 for the initial evidence-based version.
RACE_ERROR_SCALE = 1.0

# Minimum uncertainty prevents deterministic outcomes.
MIN_RACE_UNCERTAINTY = 2.0


# ============================================================
# POINTS SYSTEM
# ============================================================

RACE_POINTS = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}

SPRINT_POINTS = {
    1: 8,
    2: 7,
    3: 6,
    4: 5,
    5: 4,
    6: 3,
    7: 2,
    8: 1,
}

np.random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, possible_names, required=True):
    for name in possible_names:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            f"Could not find any of these columns: {possible_names}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def update_rolling_feature(old_value, new_value, window):
    """
    Approximate the update of a rolling feature.

    This is intentionally a lightweight approximation rather
    than rebuilding the entire historical feature pipeline
    10,000 times.
    """
    if pd.isna(old_value):
        return new_value

    return (
        old_value * (window - 1) + new_value
    ) / window


def get_model_uncertainty(model, X):
    """
    Calculate Random Forest tree-to-tree prediction spread.

    This is only used as a secondary uncertainty component.
    Historical out-of-sample error remains the dominant
    uncertainty estimate.
    """
    try:
        preprocessor = model.named_steps["preprocessor"]
        rf_model = model.named_steps["model"]

        transformed = preprocessor.transform(X)

        tree_predictions = np.array([
            tree.predict(transformed)
            for tree in rf_model.estimators_
        ])

        uncertainty = np.std(
            tree_predictions,
            axis=0
        )

        return np.maximum(uncertainty, 0.0)

    except Exception:
        return np.zeros(len(X))


def most_common_integer(values):
    counts = np.bincount(
        values.astype(int),
        minlength=N_DRIVERS + 1
    )

    counts[0] = 0

    return int(np.argmax(counts))


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("F1 2026 MONTE CARLO CHAMPIONSHIP SIMULATION")
print("=" * 70)

print("\nLoading data...")

race_results = pd.read_csv(
    os.path.join(DATA_DIR, "race_results_2026.csv")
)

qualifying_results = pd.read_csv(
    os.path.join(DATA_DIR, "qualifying_results_2026.csv")
)

driver_standings = pd.read_csv(
    os.path.join(DATA_DIR, "driver_standings_2026.csv")
)

feature_dataset = pd.read_csv(
    os.path.join(DATA_DIR, "feature_dataset_2026.csv")
)

upcoming = pd.read_csv(
    os.path.join(DATA_DIR, "upcoming_race_features_2026.csv")
)

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "random_forest_prequalifying.pkl"
    )
)

feature_config = joblib.load(
    os.path.join(
        MODEL_DIR,
        "prequalifying_features.pkl"
    )
)

FEATURE_COLUMNS = feature_config["all_features"]

print(f"Loaded {SIMULATIONS:,} simulations.")
print(f"Remaining races: {len(REMAINING_ROUNDS)}")


# ============================================================
# IDENTIFY DATASET COLUMNS
# ============================================================

driver_id_col = find_column(
    race_results,
    ["driver_id", "driverId"]
)

driver_name_col = find_column(
    race_results,
    ["driver", "driver_name"]
)

constructor_col = find_column(
    race_results,
    ["constructor", "constructor_name"]
)

round_col = find_column(
    race_results,
    ["round"]
)

position_col = find_column(
    race_results,
    ["position"]
)

points_col = find_column(
    race_results,
    ["points"]
)

standings_driver_id_col = find_column(
    driver_standings,
    ["driver_id", "driverId"]
)

standings_points_col = find_column(
    driver_standings,
    ["championship_points", "points"]
)

standings_position_col = find_column(
    driver_standings,
    ["championship_position", "position"]
)

standings_wins_col = find_column(
    driver_standings,
    ["wins"],
    required=False
)


# ============================================================
# CURRENT DRIVER LIST
# ============================================================

latest_race = race_results[
    race_results[round_col] == CURRENT_ROUND
].copy()

drivers = latest_race[
    [
        driver_id_col,
        driver_name_col,
        constructor_col,
    ]
].drop_duplicates(
    subset=[driver_id_col]
).reset_index(drop=True)

drivers = drivers.rename(
    columns={
        driver_id_col: "driver_id",
        driver_name_col: "driver",
        constructor_col: "constructor",
    }
)

N_DRIVERS = len(drivers)

driver_ids = drivers["driver_id"].tolist()
driver_names = drivers["driver"].tolist()
driver_constructors = drivers["constructor"].tolist()

print(f"Drivers: {N_DRIVERS}")


# ============================================================
# CURRENT DRIVER CHAMPIONSHIP
# ============================================================

latest_standings = driver_standings[
    driver_standings["round"] == CURRENT_ROUND
].copy()

latest_standings = latest_standings.rename(
    columns={
        standings_driver_id_col: "driver_id"
    }
)

current_points = {}
current_positions = {}
current_wins = {}

for _, row in latest_standings.iterrows():

    driver_id = row["driver_id"]

    current_points[driver_id] = float(
        pd.to_numeric(
            row[standings_points_col],
            errors="coerce"
        )
    )

    current_positions[driver_id] = float(
        pd.to_numeric(
            row[standings_position_col],
            errors="coerce"
        )
    )

    if standings_wins_col:
        current_wins[driver_id] = float(
            pd.to_numeric(
                row[standings_wins_col],
                errors="coerce"
            )
        )
    else:
        current_wins[driver_id] = 0.0


# ============================================================
# CURRENT CONSTRUCTOR CHAMPIONSHIP
# ============================================================

race_numeric = race_results.copy()

race_numeric["position_num"] = pd.to_numeric(
    race_numeric[position_col],
    errors="coerce"
)

race_numeric["points_num"] = pd.to_numeric(
    race_numeric[points_col],
    errors="coerce"
).fillna(0)

constructor_races = race_numeric[
    race_numeric[round_col] <= CURRENT_ROUND
].copy()

constructor_summary = (
    constructor_races
    .groupby(constructor_col)
    .agg(
        points=("points_num", "sum"),
        wins=(
            "position_num",
            lambda x: int((x == 1).sum())
        ),
    )
    .reset_index()
)

constructor_summary = constructor_summary.sort_values(
    ["points", "wins"],
    ascending=[False, False]
).reset_index(drop=True)

constructor_summary["position"] = (
    np.arange(len(constructor_summary)) + 1
)

constructor_points = dict(
    zip(
        constructor_summary[constructor_col],
        constructor_summary["points"]
    )
)

constructor_positions = dict(
    zip(
        constructor_summary[constructor_col],
        constructor_summary["position"]
    )
)

constructor_wins = dict(
    zip(
        constructor_summary[constructor_col],
        constructor_summary["wins"]
    )
)


# ============================================================
# CURRENT FEATURE STATE
# ============================================================

print("Preparing starting feature state...")

latest_features = feature_dataset[
    feature_dataset["round"] == CURRENT_ROUND
].copy()

round16_features = upcoming[
    upcoming["round"] == CURRENT_ROUND + 1
].copy()

# Start from the actual Round 16 feature dataset.
base_features = drivers.merge(
    round16_features,
    on=["driver_id", "driver", "constructor"],
    how="left",
    suffixes=("", "_upcoming")
)

feature_names = [
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

latest_feature_lookup = latest_features.set_index(
    "driver_id"
)

for feature in feature_names:

    if feature not in base_features.columns:
        base_features[feature] = np.nan

    if feature in latest_feature_lookup.columns:

        fallback = base_features[
            "driver_id"
        ].map(
            latest_feature_lookup[feature]
        )

        base_features[feature] = (
            base_features[feature]
            .fillna(fallback)
        )


# Use actual current championship values.
base_features[
    "driver_championship_position_before"
] = base_features[
    "driver_id"
].map(current_positions)

base_features[
    "driver_championship_points_before"
] = base_features[
    "driver_id"
].map(current_points)

base_features[
    "driver_wins_before"
] = base_features[
    "driver_id"
].map(current_wins)

base_features[
    "constructor_championship_position_before"
] = base_features[
    "constructor"
].map(constructor_positions)

base_features[
    "constructor_championship_points_before"
] = base_features[
    "constructor"
].map(constructor_points)

base_features[
    "constructor_wins_before"
] = base_features[
    "constructor"
].map(constructor_wins)


# ============================================================
# MODEL FEATURE CHECK
# ============================================================

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in base_features.columns
]

if missing_features:

    print("\nMissing model features:")

    for feature in missing_features:
        print(f"  - {feature}")

    raise ValueError(
        "Could not construct all features required "
        "by the trained model."
    )


# ============================================================
# SIMULATION STATE
# ============================================================

points_matrix = np.zeros(
    (SIMULATIONS, N_DRIVERS),
    dtype=np.float32
)

for i, driver_id in enumerate(driver_ids):

    points_matrix[:, i] = current_points.get(
        driver_id,
        0.0
    )

prediction_features = base_features.copy()


# ============================================================
# PREPARE DNF RATES
# ============================================================

# We estimate DNF probability from completed 2026 races.
# This is a data-derived component rather than a manually
# chosen probability.

dnf_rates = np.zeros(N_DRIVERS)

status_col = find_column(
    race_results,
    ["status"],
    required=False
)

if status_col:

    dnf_keywords = (
        "retired|accident|collision|engine|mechanical|"
        "failure|gearbox|electrical|hydraulics|"
        "disqualified"
    )

    for i, driver_id in enumerate(driver_ids):

        driver_history = race_results[
            race_results[driver_id_col] == driver_id
        ].copy()

        if len(driver_history) > 0:

            statuses = (
                driver_history[status_col]
                .astype(str)
                .str.lower()
            )

            dnf_mask = statuses.str.contains(
                dnf_keywords,
                regex=True,
                na=False
            )

            dnf_rates[i] = float(
                dnf_mask.mean()
            )

# Cap extreme estimates caused by very small samples.
dnf_rates = np.clip(
    dnf_rates,
    0.01,
    0.25
)


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

print("\nPreparing race predictions...")

for race_number, round_number in enumerate(
    REMAINING_ROUNDS,
    start=1
):

    print(
        f"\nRace {race_number}/{len(REMAINING_ROUNDS)} "
        f"→ Round {round_number}"
    )

    X = prediction_features[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    raw_prediction = model.predict(X)

    model_uncertainty = get_model_uncertainty(
        model,
        X
    )

    # Combine empirical historical error with RF spread.
    #
    # Historical error is the primary calibration source.
    uncertainty = np.sqrt(
        HISTORICAL_ERROR_SCALE ** 2
        + model_uncertainty ** 2
    )

    uncertainty *= RACE_ERROR_SCALE

    uncertainty = np.maximum(
        uncertainty,
        MIN_RACE_UNCERTAINTY
    )

    print("   ML predictions generated.")

    # --------------------------------------------------------
    # Simulate race outcomes
    # --------------------------------------------------------

    noise = np.random.normal(
        loc=0.0,
        scale=uncertainty,
        size=(SIMULATIONS, N_DRIVERS)
    )

    sampled_scores = (
        raw_prediction.reshape(1, -1)
        + noise
    )

    # --------------------------------------------------------
    # Simulate DNF events
    # --------------------------------------------------------

    dnf_random = np.random.random(
        (SIMULATIONS, N_DRIVERS)
    )

    dnf_events = (
        dnf_random
        < dnf_rates.reshape(1, -1)
    )

    # A DNF gets a large finishing-position penalty.
    # The exact classified position is not modeled; this is
    # intentionally conservative for a first-pass simulation.
    sampled_scores = sampled_scores + (
        dnf_events * (N_DRIVERS + 5)
    )

    # Lower score = better finishing position.
    race_order = np.argsort(
        sampled_scores,
        axis=1
    )

    # --------------------------------------------------------
    # Convert race order to points
    # --------------------------------------------------------

    race_points = np.zeros_like(
        points_matrix
    )

    for position_index in range(N_DRIVERS):

        driver_indices = race_order[
            :,
            position_index
        ]

        points = RACE_POINTS.get(
            position_index + 1,
            0
        )

        if points > 0:

            race_points[
                np.arange(SIMULATIONS),
                driver_indices
            ] = points

    # --------------------------------------------------------
    # Remaining Sprint
    # --------------------------------------------------------

    if round_number in SPRINT_ROUNDS:

        print("   Singapore Sprint included.")

        sprint_noise = np.random.normal(
            loc=0.0,
            scale=uncertainty,
            size=(SIMULATIONS, N_DRIVERS)
        )

        sprint_scores = (
            raw_prediction.reshape(1, -1)
            + sprint_noise
        )

        sprint_dnf_random = np.random.random(
            (SIMULATIONS, N_DRIVERS)
        )

        sprint_dnf = (
            sprint_dnf_random
            < dnf_rates.reshape(1, -1)
        )

        sprint_scores = sprint_scores + (
            sprint_dnf * (N_DRIVERS + 5)
        )

        sprint_order = np.argsort(
            sprint_scores,
            axis=1
        )

        for position_index in range(
            min(8, N_DRIVERS)
        ):

            driver_indices = sprint_order[
                :,
                position_index
            ]

            sprint_points = SPRINT_POINTS[
                position_index + 1
            ]

            race_points[
                np.arange(SIMULATIONS),
                driver_indices
            ] += sprint_points

    # --------------------------------------------------------
    # Update championship points
    # --------------------------------------------------------

    points_matrix += race_points

    # --------------------------------------------------------
    # Calculate expected race result
    # --------------------------------------------------------

    expected_position = np.zeros(
        N_DRIVERS
    )

    for driver_index in range(
        N_DRIVERS
    ):

        driver_positions = np.empty(
            SIMULATIONS,
            dtype=np.int16
        )

        for position_index in range(
            N_DRIVERS
        ):

            mask = (
                race_order[:, position_index]
                == driver_index
            )

            driver_positions[mask] = (
                position_index + 1
            )

        expected_position[
            driver_index
        ] = np.mean(driver_positions)

    # --------------------------------------------------------
    # Update rolling driver features
    # --------------------------------------------------------

    for i, driver_id in enumerate(driver_ids):

        predicted_position = (
            expected_position[i]
        )

        predicted_points = float(
            np.mean(
                race_points[:, i]
            )
        )

        row_mask = (
            prediction_features["driver_id"]
            == driver_id
        )

        prediction_features.loc[
            row_mask,
            "driver_avg_finish_last_3"
        ] = update_rolling_feature(
            prediction_features.loc[
                row_mask,
                "driver_avg_finish_last_3"
            ].iloc[0],
            predicted_position,
            3
        )

        prediction_features.loc[
            row_mask,
            "driver_avg_finish_last_5"
        ] = update_rolling_feature(
            prediction_features.loc[
                row_mask,
                "driver_avg_finish_last_5"
            ].iloc[0],
            predicted_position,
            5
        )

        prediction_features.loc[
            row_mask,
            "driver_avg_points_last_3"
        ] = update_rolling_feature(
            prediction_features.loc[
                row_mask,
                "driver_avg_points_last_3"
            ].iloc[0],
            predicted_points,
            3
        )

        prediction_features.loc[
            row_mask,
            "driver_avg_points_last_5"
        ] = update_rolling_feature(
            prediction_features.loc[
                row_mask,
                "driver_avg_points_last_5"
            ].iloc[0],
            predicted_points,
            5
        )

    # --------------------------------------------------------
    # Update championship features
    # --------------------------------------------------------

    expected_total_points = np.mean(
        points_matrix,
        axis=0
    )

    championship_order = np.argsort(
        -expected_total_points
    )

    expected_championship_position = np.empty(
        N_DRIVERS
    )

    for position, driver_index in enumerate(
        championship_order
    ):

        expected_championship_position[
            driver_index
        ] = position + 1

    prediction_features[
        "driver_championship_points_before"
    ] = prediction_features[
        "driver_id"
    ].map(
        {
            driver_id: expected_total_points[i]
            for i, driver_id
            in enumerate(driver_ids)
        }
    )

    prediction_features[
        "driver_championship_position_before"
    ] = prediction_features[
        "driver_id"
    ].map(
        {
            driver_id:
            expected_championship_position[i]
            for i, driver_id
            in enumerate(driver_ids)
        }
    )

    # --------------------------------------------------------
    # Constructor championship features
    # --------------------------------------------------------

    constructor_expected_points = {}

    for constructor in drivers[
        "constructor"
    ].unique():

        indices = [
            i
            for i, current_constructor
            in enumerate(driver_constructors)
            if current_constructor == constructor
        ]

        constructor_expected_points[
            constructor
        ] = float(
            np.sum(
                expected_total_points[
                    indices
                ]
            )
        )

    constructor_order = sorted(
        constructor_expected_points,
        key=constructor_expected_points.get,
        reverse=True
    )

    constructor_position_map = {
        constructor: position + 1
        for position, constructor
        in enumerate(constructor_order)
    }

    prediction_features[
        "constructor_championship_points_before"
    ] = prediction_features[
        "constructor"
    ].map(
        constructor_expected_points
    )

    prediction_features[
        "constructor_championship_position_before"
    ] = prediction_features[
        "constructor"
    ].map(
        constructor_position_map
    )

    print(
        f"   Completed race {race_number}/"
        f"{len(REMAINING_ROUNDS)}"
    )


# ============================================================
# FINAL CHAMPIONSHIP RESULTS
# ============================================================

print(
    "\nCalculating final championship probabilities..."
)

final_ranking = np.argsort(
    -points_matrix,
    axis=1
)

final_positions = np.empty(
    (SIMULATIONS, N_DRIVERS),
    dtype=np.int16
)

for position_index in range(N_DRIVERS):

    driver_indices = final_ranking[
        :,
        position_index
    ]

    final_positions[
        np.arange(SIMULATIONS),
        driver_indices
    ] = position_index + 1


# ============================================================
# BUILD ONE RESULT PER DRIVER
# ============================================================

results = []

for i, driver_id in enumerate(driver_ids):

    final_points = points_matrix[
        :,
        i
    ]

    positions = final_positions[
        :,
        i
    ]

    championship_wins = int(
        np.sum(
            positions == 1
        )
    )

    most_likely_position = (
        most_common_integer(
            positions
        )
    )

    expected_position = float(
        np.mean(positions)
    )

    championship_probability = (
        championship_wins
        / SIMULATIONS
        * 100
    )

    results.append({

        "driver_id":
            driver_id,

        "driver":
            driver_names[i],

        "constructor":
            driver_constructors[i],

        "current_points":
            round(
                current_points.get(
                    driver_id,
                    0.0
                ),
                2
            ),

        "expected_final_points":
            round(
                float(
                    np.mean(final_points)
                ),
                2
            ),

        "final_points_std":
            round(
                float(
                    np.std(final_points)
                ),
                2
            ),

        # Mean position across 10,000 simulations.
        "expected_final_position":
            round(
                expected_position,
                2
            ),

        # Most frequently observed integer position.
        "most_likely_final_position":
            most_likely_position,

        "championship_probability":
            round(
                championship_probability,
                2
            ),

        "championship_wins":
            championship_wins,
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

# Sort by championship probability, then by expected position.
results_df = results_df.sort_values(
    [
        "championship_probability",
        "expected_final_position",
    ],
    ascending=[
        False,
        True,
    ]
).reset_index(drop=True)


# ============================================================
# VALIDATION CHECKS
# ============================================================

probability_total = results_df[
    "championship_probability"
].sum()

if not np.isclose(
    probability_total,
    100.0,
    atol=0.11
):

    raise ValueError(
        f"Championship probabilities do not sum "
        f"to approximately 100%. Current total: "
        f"{probability_total:.2f}%"
    )

if len(results_df) != N_DRIVERS:

    raise ValueError(
        f"Expected {N_DRIVERS} drivers in the final "
        f"results but found {len(results_df)}."
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO SIMULATION COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)

print("\nChampionship probabilities:\n")

display_columns = [
    "driver",
    "constructor",
    "current_points",
    "expected_final_points",
    "expected_final_position",
    "most_likely_final_position",
    "championship_probability",
]

print(
    results_df[
        display_columns
    ].to_string(index=False)
)

print(
    "\nProbability total:",
    round(
        probability_total,
        2
    ),
    "%"
)

print(
    "Total drivers:",
    len(results_df)
)

print("=" * 70)
