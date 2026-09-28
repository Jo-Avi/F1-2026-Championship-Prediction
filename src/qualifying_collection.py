import requests
import pandas as pd
import os
import time

BASE_URL = "https://api.jolpi.ca/ergast/f1"

HEADERS = {
    "User-Agent": "F1-Championship-Predictor/1.0"
}

START_YEAR = 2018
END_YEAR = 2025

REQUEST_DELAY = 1.0
MAX_RETRIES = 6


def get_request(url, params=None):

    for attempt in range(MAX_RETRIES):

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        if response.status_code == 200:
            return response

        if response.status_code == 429:

            wait_time = 5 * (2 ** attempt)

            print(
                f"    Rate limited. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)
            continue

        print(
            f"    Request failed: "
            f"HTTP {response.status_code}"
        )

        return None

    print("    Maximum retries reached.")

    return None


# ---------------------------------------------------------
# Use completed race dataset to determine all rounds
# ---------------------------------------------------------

race_df = pd.read_csv(
    "data/race_results_2018_2025.csv"
)

race_rounds = (
    race_df[
        ["season", "round"]
    ]
    .drop_duplicates()
    .sort_values(
        ["season", "round"]
    )
)

all_qualifying = []


# ---------------------------------------------------------
# Collect qualifying results
# ---------------------------------------------------------

for season in range(
    START_YEAR,
    END_YEAR + 1
):

    rounds = (
        race_rounds[
            race_rounds["season"] == season
        ]["round"]
        .tolist()
    )

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Collecting qualifying for {season}..."
    )

    print(
        f"Rounds: {len(rounds)}"
    )

    print(
        f"{'=' * 60}"
    )

    for round_number in rounds:

        print(
            f"  Round {round_number}"
        )

        url = (
            f"{BASE_URL}/"
            f"{season}/"
            f"{round_number}/"
            f"qualifying/"
        )

        response = get_request(
            url,
            params={
                "limit": 100
            }
        )

        if response is None:
            continue

        data = response.json()

        try:

            race = (
                data[
                    "MRData"
                ][
                    "RaceTable"
                ][
                    "Races"
                ][0]
            )

            results = race.get(
                "QualifyingResults",
                []
            )

        except (
            KeyError,
            IndexError
        ):

            print(
                "    No qualifying data"
            )

            continue

        for result in results:

            driver = result.get(
                "Driver",
                {}
            )

            constructor = result.get(
                "Constructor",
                {}
            )

            all_qualifying.append({

                "season": season,

                "round": round_number,

                "driver_id": driver.get(
                    "driverId"
                ),

                "driver": (
                    driver.get(
                        "givenName",
                        ""
                    )
                    + " "
                    + driver.get(
                        "familyName",
                        ""
                    )
                ).strip(),

                "constructor": constructor.get(
                    "name"
                ),

                "qualifying_position": pd.to_numeric(
                    result.get("position"),
                    errors="coerce"
                ),

                "Q1": result.get("Q1"),

                "Q2": result.get("Q2"),

                "Q3": result.get("Q3")
            })

        time.sleep(
            REQUEST_DELAY
        )


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(
    all_qualifying
)

df = df.sort_values(
    [
        "season",
        "round",
        "qualifying_position"
    ]
).reset_index(
    drop=True
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

os.makedirs(
    "data",
    exist_ok=True
)

output_file = (
    "data/qualifying_results_2018_2025.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "QUALIFYING COLLECTION COMPLETE"
)

print(
    "=" * 60
)

print(
    "Total rows:",
    len(df)
)

print(
    "Seasons:",
    df["season"].nunique()
)

print(
    "Total races:",
    df[
        ["season", "round"]
    ].drop_duplicates().shape[0]
)

print(
    "\nQualifying races per season:"
)

print(
    df.groupby("season")[
        "round"
    ].nunique()
)

print(
    f"\nSaved to: {output_file}"
)