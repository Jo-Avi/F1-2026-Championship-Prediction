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

# Wait between successful API requests
REQUEST_DELAY = 1.0

# Maximum retry attempts after HTTP 429
MAX_RETRIES = 6


def get_request(url, params=None):
    """
    Make a rate-limit-safe GET request.
    """

    for attempt in range(MAX_RETRIES):

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        # Successful request
        if response.status_code == 200:
            return response

        # Rate limited
        if response.status_code == 429:

            wait_time = 5 * (2 ** attempt)

            print(
                f"    Rate limited (429). "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)

            continue

        print(
            f"    Request failed: "
            f"HTTP {response.status_code}"
        )

        return None

    print(
        "    Maximum retries reached."
    )

    return None


# ---------------------------------------------------------
# Collect race results
# ---------------------------------------------------------

all_results = []

expected_races = {}


for season in range(
    START_YEAR,
    END_YEAR + 1
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Collecting race calendar for {season}..."
    )

    print(
        f"{'=' * 60}"
    )

    # -----------------------------------------------------
    # Get complete race calendar
    # -----------------------------------------------------

    races_url = (
        f"{BASE_URL}/{season}/races/"
    )

    response = get_request(
        races_url,
        params={
            "limit": 100
        }
    )

    if response is None:
        print(
            f"Could not get calendar for {season}"
        )
        continue

    data = response.json()

    try:

        races = (
            data[
                "MRData"
            ][
                "RaceTable"
            ][
                "Races"
            ]
        )

    except KeyError:

        print(
            f"No calendar found for {season}"
        )

        continue

    expected_races[season] = len(races)

    print(
        f"Races found: {len(races)}"
    )

    # -----------------------------------------------------
    # Get results for every race
    # -----------------------------------------------------

    for race in races:

        round_number = int(
            race["round"]
        )

        race_name = race[
            "raceName"
        ]

        print(
            f"  Round {round_number}: "
            f"{race_name}"
        )

        results_url = (
            f"{BASE_URL}/"
            f"{season}/"
            f"{round_number}/"
            f"results/"
        )

        response = get_request(
            results_url,
            params={
                "limit": 100
            }
        )

        if response is None:

            print(
                f"    Could not collect "
                f"Round {round_number}"
            )

            continue

        result_data = response.json()

        try:

            race_data = (
                result_data[
                    "MRData"
                ][
                    "RaceTable"
                ][
                    "Races"
                ][0]
            )

        except (
            KeyError,
            IndexError
        ):

            print(
                "    No results found"
            )

            continue

        results = race_data.get(
            "Results",
            []
        )

        for result in results:

            driver = result.get(
                "Driver",
                {}
            )

            constructor = result.get(
                "Constructor",
                {}
            )

            fastest_lap = result.get(
                "FastestLap",
                {}
            )

            fastest_lap_number = (
                fastest_lap.get("lap")
                if fastest_lap
                else None
            )

            all_results.append({

                "season": season,

                "round": round_number,

                "race": race_name,

                "circuit": (
                    race[
                        "Circuit"
                    ].get(
                        "circuitName"
                    )
                ),

                "driver_id": (
                    driver.get(
                        "driverId"
                    )
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

                "constructor": (
                    constructor.get(
                        "name"
                    )
                ),

                "grid": pd.to_numeric(
                    result.get(
                        "grid"
                    ),
                    errors="coerce"
                ),

                "position": pd.to_numeric(
                    result.get(
                        "position"
                    ),
                    errors="coerce"
                ),

                "points": pd.to_numeric(
                    result.get(
                        "points",
                        0
                    ),
                    errors="coerce"
                ),

                "status": result.get(
                    "status"
                ),

                "laps": pd.to_numeric(
                    result.get(
                        "laps"
                    ),
                    errors="coerce"
                ),

                "fastest_lap": pd.to_numeric(
                    fastest_lap_number,
                    errors="coerce"
                )
            })

        # Delay after every race request
        time.sleep(
            REQUEST_DELAY
        )


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(
    all_results
)

if df.empty:

    print(
        "\nNo data was collected."
    )

    raise SystemExit


df = df.sort_values(
    [
        "season",
        "round",
        "position"
    ]
).reset_index(
    drop=True
)


# ---------------------------------------------------------
# Verify collected seasons
# ---------------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "COLLECTION VERIFICATION"
)

print(
    "=" * 60
)

actual_races = (
    df[
        [
            "season",
            "round"
        ]
    ]
    .drop_duplicates()
    .groupby("season")
    .size()
)


print(
    "\nExpected races:"
)

for season, count in expected_races.items():

    print(
        f"{season}: {count}"
    )


print(
    "\nCollected races:"
)

print(
    actual_races
)


# ---------------------------------------------------------
# Check for missing races
# ---------------------------------------------------------

missing_races = []

for season, expected in expected_races.items():

    actual = actual_races.get(
        season,
        0
    )

    if actual != expected:

        missing_races.append({
            "season": season,
            "expected": expected,
            "collected": actual
        })


if missing_races:

    print(
        "\nWARNING: Some races are missing!"
    )

    print(
        pd.DataFrame(
            missing_races
        )
    )

    print(
        "\nThe dataset will NOT be treated "
        "as complete."
    )

else:

    print(
        "\nAll seasons are complete."
    )


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

os.makedirs(
    "data",
    exist_ok=True
)

output_file = (
    "data/race_results_2018_2025.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "RACE RESULTS COLLECTION COMPLETE"
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
        [
            "season",
            "round"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    "\nRaces per season:"
)

print(
    df.groupby("season")[
        "round"
    ].nunique()
)

print(
    f"\nSaved to: {output_file}"
)