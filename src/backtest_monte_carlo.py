import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ============================================================
# HISTORICAL MONTE CARLO BACKTEST
# ============================================================
#
# Purpose:
#   Test whether our championship-probability engine is
#   calibrated using completed F1 seasons.
#
# Method:
#   1. Choose a historical season.
#   2. Pretend we are after Round 15.
#   3. Train ONLY on seasons before that target season.
#   4. Simulate the remaining races 10,000 times.
#   5. Compare championship probabilities with the actual
#      champion.
#
# This avoids using future-season training data.
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

SIMULATIONS = 10_000
CHECKPOINT_ROUND = 15

# Seasons with enough races after Round 15 for a useful test.
# 2021-2025 gives us five independent historical tests.
TARGET_SEASONS = [2021, 2022, 2023, 2024, 2025]

RANDOM_SEED = 42

# Same points used by the 2026 simulator.
RACE_POINTS = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

SPRINT_POINTS = {
    1: 8, 2: 7, 3: 6, 4: 5,
    5: 4, 6: 3, 7: 2, 8: 1
}

# These match the pre-qualifying model used for 2026.
NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "driver",
    "constructor",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Historical 2025 test MAE of the pre-qualifying model.
# Used as an empirical race-error scale.
HISTORICAL_ERROR_SCALE = 3.85

MIN_UNCERTAINTY = 2.0


# ============================================================
# HELPERS
# ============================================================

def build_model():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf),
    ])


def update_rolling(old, new, window):
    if pd.isna(old):
        return new
    return (old * (window - 1) + new) / window


def add_points_from_order(order, points_dict, point_matrix):
    n_sims, n_drivers = order.shape

    for pos in range(n_drivers):
        pts = points_dict.get(pos + 1, 0)
        if pts:
            point_matrix[
                np.arange(n_sims),
                order[:, pos]
            ] += pts


def prepare_dnf_rates(
    race_results,
    driver_ids,
    driver_id_col,
    status_col,
    checkpoint
):
    rates = np.zeros(len(driver_ids))

    if status_col is None:
        return np.full(len(driver_ids), 0.05)

    dnf_pattern = (
        "retired|accident|collision|engine|mechanical|"
        "failure|gearbox|electrical|hydraulics|"
        "disqualified"
    )

    for i, driver_id in enumerate(driver_ids):

        hist = race_results[
            (race_results[driver_id_col] == driver_id)
            & (race_results["round"] <= checkpoint)
        ]

        if len(hist) == 0:
            rates[i] = 0.05
            continue

        statuses = (
            hist[status_col]
            .astype(str)
            .str.lower()
        )

        rates[i] = float(
            statuses.str.contains(
                dnf_pattern,
                regex=True,
                na=False
            ).mean()
        )

    return np.clip(rates, 0.01, 0.25)


def most_common_position(values):
    counts = np.bincount(
        values.astype(int)
    )
    counts[0] = 0
    return int(np.argmax(counts))


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("=" * 72)
print("F1 HISTORICAL MONTE CARLO BACKTEST")
print("=" * 72)

print("\nLoading historical datasets...")

feature_dataset = pd.read_csv(
    os.path.join(DATA_DIR, "feature_dataset_v3.csv")
)

race_results = pd.read_csv(
    os.path.join(DATA_DIR, "race_results_2018_2025.csv")
)

qualifying_results = pd.read_csv(
    os.path.join(DATA_DIR, "qualifying_results_2018_2025.csv")
)

sprint_results = pd.read_csv(
    os.path.join(DATA_DIR, "sprint_results_2018_2025.csv")
)

print(f"Feature rows: {len(feature_dataset):,}")
print(f"Race rows: {len(race_results):,}")


# ============================================================
# NORMALIZE / RECONCILE DRIVER IDENTIFIERS
# ============================================================

# feature_dataset_v3.csv uses `driver` as its driver identifier,
# while race_results_2018_2025.csv also contains `driver_id`.
# The backtest needs driver_id for joining to standings and for
# tracking drivers through the simulation, so recover it from the
# race-results dataset without changing the feature dataset file.

required_feature_columns = [
    "season",
    "round",
    "driver",
    "constructor",
]

missing = [
    c for c in required_feature_columns
    if c not in feature_dataset.columns
]

if missing:
    raise ValueError(
        f"feature_dataset_v3.csv is missing: {missing}"
    )

if "driver_id" not in race_results.columns:
    raise ValueError(
        "race_results_2018_2025.csv is missing: ['driver_id']"
    )

# Build a stable season/round/driver mapping from the raw race data.
driver_id_map = (
    race_results[
        ["season", "round", "driver", "constructor", "driver_id"]
    ]
    .drop_duplicates(
        ["season", "round", "driver", "constructor"]
    )
)

# Guard against ambiguous mappings.
duplicate_driver_keys = (
    driver_id_map
    .groupby(["season", "round", "driver", "constructor"])["driver_id"]
    .nunique()
)

if (duplicate_driver_keys > 1).any():
    raise ValueError(
        "Ambiguous driver_id mapping found in race_results_2018_2025.csv."
    )

feature_dataset = feature_dataset.merge(
    driver_id_map,
    on=["season", "round", "driver", "constructor"],
    how="left",
    validate="one_to_one",
)

if feature_dataset["driver_id"].isna().any():
    missing_rows = feature_dataset.loc[
        feature_dataset["driver_id"].isna(),
        ["season", "round", "driver", "constructor"]
    ].drop_duplicates()

    raise ValueError(
        "Could not map driver_id for some feature rows:\n"
        + missing_rows.head(20).to_string(index=False)
    )

feature_dataset["driver_id"] = feature_dataset["driver_id"].astype(str)

# Keep the raw race-results driver IDs in the same type for all
# downstream joins/comparisons in this script.
race_results["driver_id"] = race_results["driver_id"].astype(str)

print("Driver IDs reconciled from race_results_2018_2025.csv.")


# ============================================================
# BACKTEST ONE SEASON
# ============================================================

def backtest_season(target_season):

    print("\n" + "-" * 72)
    print(f"BACKTEST SEASON: {target_season}")
    print("-" * 72)

    # --------------------------------------------------------
    # Train only on earlier seasons
    # --------------------------------------------------------

    train = feature_dataset[
        feature_dataset["season"] < target_season
    ].copy()

    train = train.dropna(
        subset=["position"]
    )

    if train.empty:
        raise ValueError(
            f"No training data before {target_season}."
        )

    model = build_model()

    model.fit(
        train[FEATURE_COLUMNS],
        train["position"]
    )

    print(
        f"Training seasons: "
        f"{sorted(train['season'].unique().tolist())}"
    )

    # --------------------------------------------------------
    # Determine final/remaining rounds
    # --------------------------------------------------------

    season_races = race_results[
        race_results["season"] == target_season
    ].copy()

    max_round = int(
        season_races["round"].max()
    )

    if max_round <= CHECKPOINT_ROUND:
        print(
            f"Skipping {target_season}: "
            f"season has only {max_round} rounds."
        )
        return None

    remaining_rounds = list(
        range(
            CHECKPOINT_ROUND + 1,
            max_round + 1
        )
    )

    # --------------------------------------------------------
    # Current drivers after checkpoint
    # --------------------------------------------------------

    checkpoint_races = season_races[
        season_races["round"] == CHECKPOINT_ROUND
    ].copy()

    drivers = checkpoint_races[
        ["driver_id", "driver", "constructor"]
    ].drop_duplicates(
        "driver_id"
    ).reset_index(drop=True)

    driver_ids = drivers["driver_id"].tolist()
    driver_names = drivers["driver"].tolist()
    driver_constructors = drivers["constructor"].tolist()

    n_drivers = len(drivers)

    # --------------------------------------------------------
    # Current standings
    # --------------------------------------------------------

    # Reconstruct championship standings directly from race + sprint
    # results because the historical standings CSV is only partially populated.
    checkpoint_results = season_races[
        season_races["round"] <= CHECKPOINT_ROUND
    ].copy()

    checkpoint_results["points_num"] = pd.to_numeric(
        checkpoint_results["points"], errors="coerce"
    ).fillna(0)

    current_points_series = (
        checkpoint_results.groupby("driver_id")["points_num"].sum()
    )

    current_wins_series = (
        checkpoint_results.assign(
            position_num=pd.to_numeric(
                checkpoint_results["position"], errors="coerce"
            )
        )
        .groupby("driver_id")["position_num"]
        .apply(lambda x: int((x == 1).sum()))
    )

    # Add Sprint points to the checkpoint championship totals.
    target_sprints = sprint_results[
        (sprint_results["season"] == target_season)
        & (sprint_results["round"] <= CHECKPOINT_ROUND)
    ].copy()

    if not target_sprints.empty:
        target_sprints["points_num"] = pd.to_numeric(
            target_sprints["points"], errors="coerce"
        ).fillna(0)

        sprint_points = (
            target_sprints.groupby("driver_id")["points_num"].sum()
        )
        current_points_series = (
            current_points_series.add(sprint_points, fill_value=0)
        )

    current_points = {
        driver_id: float(current_points_series.get(driver_id, 0.0))
        for driver_id in driver_ids
    }

    current_wins = {
        driver_id: int(current_wins_series.get(driver_id, 0))
        for driver_id in driver_ids
    }

    checkpoint_order = sorted(
        driver_ids,
        key=lambda d: (
            -current_points.get(d, 0.0),
            -current_wins.get(d, 0),
            d
        )
    )

    current_positions = {
        driver_id: position + 1
        for position, driver_id in enumerate(checkpoint_order)
    }

    # Reconstruct the actual final championship from the complete season.
    final_results = season_races.copy()
    final_results["points_num"] = pd.to_numeric(
        final_results["points"], errors="coerce"
    ).fillna(0)

    final_points_series = (
        final_results.groupby("driver_id")["points_num"].sum()
    )

    final_sprints = sprint_results[
        sprint_results["season"] == target_season
    ].copy()

    if not final_sprints.empty:
        final_sprints["points_num"] = pd.to_numeric(
            final_sprints["points"], errors="coerce"
        ).fillna(0)

        final_sprint_points = (
            final_sprints.groupby("driver_id")["points_num"].sum()
        )

        final_points_series = (
            final_points_series.add(final_sprint_points, fill_value=0)
        )

    final_wins_series = (
        final_results.assign(
            position_num=pd.to_numeric(
                final_results["position"], errors="coerce"
            )
        )
        .groupby("driver_id")["position_num"]
        .apply(lambda x: int((x == 1).sum()))
    )

    actual_final_order = sorted(
        final_points_series.index.tolist(),
        key=lambda d: (
            -final_points_series.get(d, 0.0),
            -final_wins_series.get(d, 0),
            d
        )
    )

    actual_champion_id = actual_final_order[0]

    actual_champion_name = (
        final_results.loc[
            final_results["driver_id"] == actual_champion_id,
            "driver"
        ].iloc[0]
    )

    # --------------------------------------------------------
    # Constructor state at checkpoint
    # --------------------------------------------------------

    race_numeric = season_races[
        season_races["round"] <= CHECKPOINT_ROUND
    ].copy()

    race_numeric["position_num"] = pd.to_numeric(
        race_numeric["position"],
        errors="coerce"
    )

    race_numeric["points_num"] = pd.to_numeric(
        race_numeric["points"],
        errors="coerce"
    ).fillna(0)

    constructor_summary = (
        race_numeric
        .groupby("constructor")
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
            constructor_summary["constructor"],
            constructor_summary["points"]
        )
    )

    constructor_positions = dict(
        zip(
            constructor_summary["constructor"],
            constructor_summary["position"]
        )
    )

    constructor_wins = dict(
        zip(
            constructor_summary["constructor"],
            constructor_summary["wins"]
        )
    )

    # --------------------------------------------------------
    # Starting features for Round 16
    # --------------------------------------------------------

    start_features = feature_dataset[
        (feature_dataset["season"] == target_season)
        & (feature_dataset["round"] == CHECKPOINT_ROUND + 1)
    ].copy()

    base = drivers.merge(
        start_features[
            [
                "driver_id",
                "driver",
                "constructor",
            ] + [
                c for c in NUMERIC_FEATURES
                if c in start_features.columns
            ]
        ],
        on=[
            "driver_id",
            "driver",
            "constructor",
        ],
        how="left",
    )

    # Override championship values with checkpoint standings.
    base[
        "driver_championship_position_before"
    ] = base["driver_id"].map(
        current_positions
    )

    base[
        "driver_championship_points_before"
    ] = base["driver_id"].map(
        current_points
    )

    base[
        "driver_wins_before"
    ] = base["driver_id"].map(
        current_wins
    )

    base[
        "constructor_championship_position_before"
    ] = base["constructor"].map(
        constructor_positions
    )

    base[
        "constructor_championship_points_before"
    ] = base["constructor"].map(
        constructor_points
    )

    base[
        "constructor_wins_before"
    ] = base["constructor"].map(
        constructor_wins
    )

    # --------------------------------------------------------
    # DNF rates using only information before checkpoint
    # --------------------------------------------------------

    status_col = (
        "status"
        if "status" in race_results.columns
        else None
    )

    dnf_rates = prepare_dnf_rates(
        season_races,
        driver_ids,
        "driver_id",
        status_col,
        CHECKPOINT_ROUND
    )

    # --------------------------------------------------------
    # Simulated championship points
    # --------------------------------------------------------

    points_matrix = np.zeros(
        (SIMULATIONS, n_drivers),
        dtype=np.float32
    )

    for i, driver_id in enumerate(driver_ids):
        points_matrix[:, i] = current_points.get(
            driver_id,
            0.0
        )

    prediction_features = base.copy()

    # --------------------------------------------------------
    # Remaining Sprint rounds
    # --------------------------------------------------------

    sprint_rounds = set(
        sprint_results[
            sprint_results["season"] == target_season
        ]["round"].unique()
    )

    # --------------------------------------------------------
    # Simulate remaining races
    # --------------------------------------------------------

    for round_number in remaining_rounds:

        X = prediction_features[
            FEATURE_COLUMNS
        ].copy()

        predictions = model.predict(X)

        # Historical prediction error is the main uncertainty.
        uncertainty = np.full(
            n_drivers,
            HISTORICAL_ERROR_SCALE
        )

        # A small model-specific component.
        try:
            preprocessor = model.named_steps[
                "preprocessor"
            ]
            rf = model.named_steps["model"]

            transformed = preprocessor.transform(X)

            tree_predictions = np.array([
                tree.predict(transformed)
                for tree in rf.estimators_
            ])

            tree_std = np.std(
                tree_predictions,
                axis=0
            )

            uncertainty = np.sqrt(
                uncertainty ** 2
                + tree_std ** 2
            )

        except Exception:
            pass

        uncertainty = np.maximum(
            uncertainty,
            MIN_UNCERTAINTY
        )

        # Race variation
        noise = np.random.normal(
            0,
            uncertainty,
            size=(SIMULATIONS, n_drivers)
        )

        sampled_scores = (
            predictions.reshape(1, -1)
            + noise
        )

        # DNF variation
        dnf_random = np.random.random(
            (SIMULATIONS, n_drivers)
        )

        dnf_events = (
            dnf_random
            < dnf_rates.reshape(1, -1)
        )

        sampled_scores += (
            dnf_events
            * (n_drivers + 5)
        )

        race_order = np.argsort(
            sampled_scores,
            axis=1
        )

        race_points = np.zeros_like(
            points_matrix
        )

        add_points_from_order(
            race_order,
            RACE_POINTS,
            race_points
        )

        # ----------------------------------------------------
        # Sprint
        # ----------------------------------------------------

        if round_number in sprint_rounds:

            sprint_noise = np.random.normal(
                0,
                uncertainty,
                size=(SIMULATIONS, n_drivers)
            )

            sprint_scores = (
                predictions.reshape(1, -1)
                + sprint_noise
            )

            sprint_dnf = (
                np.random.random(
                    (SIMULATIONS, n_drivers)
                )
                < dnf_rates.reshape(1, -1)
            )

            sprint_scores += (
                sprint_dnf
                * (n_drivers + 5)
            )

            sprint_order = np.argsort(
                sprint_scores,
                axis=1
            )

            add_points_from_order(
                sprint_order,
                SPRINT_POINTS,
                race_points
            )

        points_matrix += race_points

        # ----------------------------------------------------
        # Lightweight feature updates
        # ----------------------------------------------------

        expected_position = np.zeros(
            n_drivers
        )

        for driver_index in range(n_drivers):

            driver_positions = np.empty(
                SIMULATIONS,
                dtype=np.int16
            )

            for pos in range(n_drivers):

                mask = (
                    race_order[:, pos]
                    == driver_index
                )

                driver_positions[mask] = (
                    pos + 1
                )

            expected_position[
                driver_index
            ] = np.mean(
                driver_positions
            )

        for i, driver_id in enumerate(driver_ids):

            row_mask = (
                prediction_features["driver_id"]
                == driver_id
            )

            expected_points = float(
                np.mean(
                    race_points[:, i]
                )
            )

            for feature, new_value, window in [
                (
                    "driver_avg_finish_last_3",
                    expected_position[i],
                    3
                ),
                (
                    "driver_avg_finish_last_5",
                    expected_position[i],
                    5
                ),
                (
                    "driver_avg_points_last_3",
                    expected_points,
                    3
                ),
                (
                    "driver_avg_points_last_5",
                    expected_points,
                    5
                ),
            ]:

                prediction_features.loc[
                    row_mask,
                    feature
                ] = update_rolling(
                    prediction_features.loc[
                        row_mask,
                        feature
                    ].iloc[0],
                    new_value,
                    window
                )

        # ----------------------------------------------------
        # Update expected championship state
        # ----------------------------------------------------

        expected_total = np.mean(
            points_matrix,
            axis=0
        )

        order = np.argsort(
            -expected_total
        )

        expected_position_championship = (
            np.empty(n_drivers)
        )

        for pos, driver_index in enumerate(order):
            expected_position_championship[
                driver_index
            ] = pos + 1

        prediction_features[
            "driver_championship_points_before"
        ] = prediction_features[
            "driver_id"
        ].map(
            {
                driver_id:
                expected_total[i]
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
                expected_position_championship[i]
                for i, driver_id
                in enumerate(driver_ids)
            }
        )

        # Constructor expected points.
        constructor_expected = {}

        for constructor in drivers[
            "constructor"
        ].unique():

            indices = [
                i
                for i, c in enumerate(
                    driver_constructors
                )
                if c == constructor
            ]

            constructor_expected[
                constructor
            ] = float(
                np.sum(
                    expected_total[indices]
                )
            )

        constructor_order = sorted(
            constructor_expected,
            key=constructor_expected.get,
            reverse=True
        )

        constructor_position_map = {
            constructor:
            position + 1
            for position, constructor
            in enumerate(constructor_order)
        }

        prediction_features[
            "constructor_championship_points_before"
        ] = prediction_features[
            "constructor"
        ].map(
            constructor_expected
        )

        prediction_features[
            "constructor_championship_position_before"
        ] = prediction_features[
            "constructor"
        ].map(
            constructor_position_map
        )

    # --------------------------------------------------------
    # Final simulation results
    # --------------------------------------------------------

    final_order = np.argsort(
        -points_matrix,
        axis=1
    )

    final_positions = np.empty(
        (SIMULATIONS, n_drivers),
        dtype=np.int16
    )

    for position in range(n_drivers):

        final_positions[
            np.arange(SIMULATIONS),
            final_order[:, position]
        ] = position + 1

    rows = []

    for i, driver_id in enumerate(driver_ids):

        positions = final_positions[:, i]
        final_points = points_matrix[:, i]

        championship_wins = int(
            np.sum(positions == 1)
        )

        probability = (
            championship_wins
            / SIMULATIONS
            * 100
        )

        rows.append({
            "season": target_season,
            "driver_id": driver_id,
            "driver": driver_names[i],
            "actual_champion":
                driver_id == actual_champion_id,
            "championship_probability":
                probability,
            "expected_final_position":
                float(np.mean(positions)),
            "most_likely_final_position":
                most_common_position(positions),
            "expected_final_points":
                float(np.mean(final_points)),
            "championship_wins":
                championship_wins,
        })

    result = pd.DataFrame(rows)

    predicted_champion = result.loc[
        result["championship_probability"].idxmax()
    ]

    print(
        f"Actual champion: "
        f"{actual_champion_name}"
    )

    print(
        f"Predicted champion: "
        f"{predicted_champion['driver']}"
    )

    print(
        f"Predicted probability: "
        f"{predicted_champion['championship_probability']:.2f}%"
    )

    print(
        f"Actual champion probability: "
        f"{result.loc[result['actual_champion'], 'championship_probability'].iloc[0]:.2f}%"
    )

    return result


# ============================================================
# RUN BACKTESTS
# ============================================================

all_results = []

for season in TARGET_SEASONS:

    try:
        result = backtest_season(season)

        if result is not None:
            all_results.append(result)

    except Exception as exc:

        print(
            f"\nERROR in {season}: {exc}"
        )
        raise


if not all_results:
    raise RuntimeError(
        "No historical backtests were completed."
    )


backtest_results = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# SEASON-LEVEL SUMMARY
# ============================================================

summary_rows = []

for season in TARGET_SEASONS:

    season_result = backtest_results[
        backtest_results["season"] == season
    ]

    if season_result.empty:
        continue

    predicted = season_result.loc[
        season_result[
            "championship_probability"
        ].idxmax()
    ]

    actual = season_result[
        season_result["actual_champion"]
    ].iloc[0]

    summary_rows.append({
        "season": season,
        "actual_champion":
            actual["driver"],
        "actual_champion_probability":
            actual["championship_probability"],
        "predicted_champion":
            predicted["driver"],
        "predicted_champion_probability":
            predicted["championship_probability"],
        "correct":
            predicted["driver"]
            == actual["driver"],
    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# CALIBRATION METRICS
# ============================================================

# One probability per historical season:
# probability assigned to the actual champion.
actual_champion_probabilities = (
    summary[
        "actual_champion_probability"
    ].values
    / 100.0
)

# Brier score for the actual champion event.
#
# For each season:
#   actual outcome = 1
#   predicted probability = P(actual champion)
#
# Lower is better.
brier_score = float(
    np.mean(
        (actual_champion_probabilities - 1.0) ** 2
    )
)

champion_accuracy = float(
    summary["correct"].mean()
)

average_actual_champion_probability = float(
    summary[
        "actual_champion_probability"
    ].mean()
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    DATA_DIR,
    "monte_carlo_backtest_driver_results.csv"
)

summary_file = os.path.join(
    DATA_DIR,
    "monte_carlo_backtest_summary.csv"
)

backtest_results.to_csv(
    results_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 72)
print("HISTORICAL MONTE CARLO BACKTEST COMPLETE")
print("=" * 72)

print("\nSeason-level results:\n")

print(
    summary.to_string(
        index=False
    )
)

print("\nCalibration metrics:")
print(
    f"Champion prediction accuracy: "
    f"{champion_accuracy * 100:.1f}%"
)

print(
    f"Average probability assigned to "
    f"actual champion: "
    f"{average_actual_champion_probability:.2f}%"
)

print(
    f"Brier score: "
    f"{brier_score:.4f}"
)

print(
    f"\nDriver-level results saved to:\n"
    f"{results_file}"
)

print(
    f"\nSummary saved to:\n"
    f"{summary_file}"
)

print("=" * 72)
