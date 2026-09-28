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


# =========================================================
# SPRINT POINTS BY CONSTRUCTOR / ROUND
# =========================================================

sprint_points = (
    sprint_df
    .groupby(
        [
            "season",
            "round",
            "constructor"
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
# RACE POINTS BY CONSTRUCTOR / ROUND
# =========================================================

race_points = (
    race_df
    .groupby(
        [
            "season",
            "round",
            "constructor"
        ],
        as_index=False
    )["points"]
    .sum()
    .rename(
        columns={
            "points": "race_points"
        }
    )
)


# =========================================================
# COMBINE RACE + SPRINT POINTS
# =========================================================

constructor_rounds = race_points.merge(
    sprint_points,
    on=[
        "season",
        "round",
        "constructor"
    ],
    how="outer"
)

constructor_rounds["race_points"] = (
    constructor_rounds["race_points"]
    .fillna(0)
)

constructor_rounds["sprint_points"] = (
    constructor_rounds["sprint_points"]
    .fillna(0)
)

constructor_rounds["round_points"] = (
    constructor_rounds["race_points"]
    + constructor_rounds["sprint_points"]
)


# =========================================================
# CONSTRUCTOR RACE WINS
# =========================================================

race_wins = (
    race_df[
        race_df["position"] == 1
    ]
    .groupby(
        [
            "season",
            "round",
            "constructor"
        ],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "race_wins"
        }
    )
)

constructor_rounds = constructor_rounds.merge(
    race_wins,
    on=[
        "season",
        "round",
        "constructor"
    ],
    how="left"
)

constructor_rounds["race_wins"] = (
    constructor_rounds["race_wins"]
    .fillna(0)
)


# =========================================================
# SORT
# =========================================================

constructor_rounds = constructor_rounds.sort_values(
    [
        "season",
        "round",
        "constructor"
    ]
)


# =========================================================
# BUILD CUMULATIVE CONSTRUCTOR STANDINGS
# =========================================================

all_standings = []


for season in sorted(
    constructor_rounds["season"].unique()
):

    season_df = constructor_rounds[
        constructor_rounds["season"] == season
    ].copy()

    rounds = sorted(
        season_df["round"].unique()
    )

    print(
        f"\nCollecting constructor standings "
        f"for {season}..."
    )

    print(
        f"Rounds to process: {len(rounds)}"
    )

    for round_number in rounds:

        current = season_df[
            season_df["round"] <= round_number
        ]

        # -------------------------------------------------
        # Cumulative championship points and wins
        # -------------------------------------------------

        standings = (
            current
            .groupby(
                "constructor",
                as_index=False
            )
            .agg(
                points=(
                    "round_points",
                    "sum"
                ),
                wins=(
                    "race_wins",
                    "sum"
                )
            )
        )

        # -------------------------------------------------
        # Sort championship standings
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Reorder columns
        # -------------------------------------------------

        standings = standings[
            [
                "season",
                "round",
                "constructor",
                "position",
                "points",
                "wins"
            ]
        ]

        all_standings.append(
            standings
        )

        print(
            f"  Round {round_number}: "
            f"{len(standings)} constructors"
        )


# =========================================================
# COMBINE
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
    "data/constructor_standings_2018_2025.csv"
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
    "CONSTRUCTOR STANDINGS COLLECTION COMPLETE"
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