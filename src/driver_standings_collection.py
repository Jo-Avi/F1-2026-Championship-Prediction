import pandas as pd
import os


# =========================================================
# LOAD DATA
# =========================================================

race_file = "data/race_results_2018_2025.csv"
sprint_file = "data/sprint_results_2018_2025.csv"

race_df = pd.read_csv(race_file)
sprint_df = pd.read_csv(sprint_file)


# =========================================================
# CLEAN NUMERIC COLUMNS
# =========================================================

race_df["points"] = pd.to_numeric(
    race_df["points"],
    errors="coerce"
).fillna(0)

race_df["position"] = pd.to_numeric(
    race_df["position"],
    errors="coerce"
)

sprint_df["points"] = pd.to_numeric(
    sprint_df["points"],
    errors="coerce"
).fillna(0)

sprint_df["position"] = pd.to_numeric(
    sprint_df["position"],
    errors="coerce"
)


# =========================================================
# SPRINT POINTS BY DRIVER / ROUND
# =========================================================

sprint_points = (
    sprint_df
    .groupby(
        [
            "season",
            "round",
            "driver_id"
        ],
        as_index=False
    )["points"]
    .sum()
    .rename(
        columns={
            "points": "sprint_points"
        }
    )
)


# =========================================================
# ADD SPRINT POINTS TO RACE DATA
# =========================================================

race_df = race_df.merge(
    sprint_points,
    on=[
        "season",
        "round",
        "driver_id"
    ],
    how="left"
)

race_df["sprint_points"] = (
    race_df["sprint_points"]
    .fillna(0)
)


# =========================================================
# TOTAL POINTS FOR EACH DRIVER IN EACH ROUND
# =========================================================

race_df["round_points"] = (
    race_df["points"]
    + race_df["sprint_points"]
)


# =========================================================
# GRAND PRIX WINS
# =========================================================

# A Grand Prix win is a race finishing position of 1.
# Sprint victories are NOT counted as Grand Prix wins.

race_df["race_win"] = (
    race_df["position"] == 1
).astype(int)


# =========================================================
# SORT DATA
# =========================================================

race_df = race_df.sort_values(
    [
        "season",
        "round"
    ]
)


# =========================================================
# BUILD CUMULATIVE DRIVER STANDINGS
# =========================================================

all_standings = []


for season in sorted(
    race_df["season"].unique()
):

    season_df = race_df[
        race_df["season"] == season
    ].copy()

    rounds = sorted(
        season_df["round"].unique()
    )

    print(
        f"\nCollecting driver standings "
        f"for {season}..."
    )

    print(
        f"Rounds to process: {len(rounds)}"
    )

    for round_number in rounds:

        current = season_df[
            season_df["round"] <= round_number
        ]

        # ---------------------------------------------
        # Aggregate championship points and wins
        # ---------------------------------------------

        standings = (
            current
            .groupby(
                [
                    "driver_id",
                    "driver"
                ],
                as_index=False
            )
            .agg(
                points=(
                    "round_points",
                    "sum"
                ),
                wins=(
                    "race_win",
                    "sum"
                )
            )
        )

        # ---------------------------------------------
        # Get constructor associated with driver
        # for this round
        # ---------------------------------------------

        current_round = season_df[
            season_df["round"] == round_number
        ][
            [
                "driver_id",
                "constructor"
            ]
        ].drop_duplicates(
            "driver_id"
        )

        standings = standings.merge(
            current_round,
            on="driver_id",
            how="left"
        )

        # ---------------------------------------------
        # Sort championship standings
        # ---------------------------------------------

        standings = standings.sort_values(
            [
                "points",
                "wins"
            ],
            ascending=[
                False,
                False
            ]
        ).reset_index(
            drop=True
        )

        standings["position"] = (
            standings.index + 1
        )

        standings["season"] = season
        standings["round"] = round_number

        # ---------------------------------------------
        # Reorder columns
        # ---------------------------------------------

        standings = standings[
            [
                "season",
                "round",
                "driver_id",
                "driver",
                "position",
                "points",
                "wins",
                "constructor"
            ]
        ]

        all_standings.append(
            standings
        )

        print(
            f"  Round {round_number}: "
            f"{len(standings)} drivers"
        )


# =========================================================
# COMBINE ALL STANDINGS
# =========================================================

df = pd.concat(
    all_standings,
    ignore_index=True
)


# =========================================================
# SORT
# =========================================================

df = df.sort_values(
    [
        "season",
        "round",
        "position"
    ]
).reset_index(
    drop=True
)


# =========================================================
# SAVE
# =========================================================

os.makedirs(
    "data",
    exist_ok=True
)

output_file = (
    "data/driver_standings_2018_2025.csv"
)

df.to_csv(
    output_file,
    index=False
)


# =========================================================
# VERIFICATION
# =========================================================

print(
    "\n" + "=" * 60
)

print(
    "DRIVER STANDINGS COLLECTION COMPLETE"
)

print(
    "=" * 60
)

print(
    "Rows:",
    len(df)
)

print(
    "Seasons:",
    df["season"].nunique()
)

print(
    "Season-round combinations:",
    df[
        ["season", "round"]
    ].drop_duplicates().shape[0]
)

print(
    "\nRounds per season:"
)

print(
    df.groupby("season")[
        "round"
    ].nunique()
)

print(
    "\nSample:"
)

print(
    df.head(10)
)

print(
    f"\nSaved to: {output_file}"
)