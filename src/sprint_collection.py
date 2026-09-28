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


all_sprints = []


for season in range(
    START_YEAR,
    END_YEAR + 1
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Checking sprint sessions for {season}..."
    )

    print(
        f"{'=' * 60}"
    )

    # Get complete race calendar
    races_url = (
        f"{BASE_URL}/{season}/races/"
    )

    response = get_request(
        races_url,
        params={"limit": 100}
    )

    if response is None:
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
        continue

    for race in races:

        round_number = int(
            race["round"]
        )

        race_name = race["raceName"]

        # -------------------------------------------------
        # Check whether this round has a sprint
        # -------------------------------------------------

        sprint_url = (
            f"{BASE_URL}/"
            f"{season}/"
            f"{round_number}/"
            f"sprint/"
        )

        response = get_request(
            sprint_url,
            params={"limit": 100}
        )

        if response is None:
            continue

        sprint_data = response.json()

        try:

            sprint_race = (
                sprint_data[
                    "MRData"
                ][
                    "RaceTable"
                ][
                    "Races"
                ][0]
            )

            results = sprint_race.get(
                "SprintResults",
                []
            )

        except (
            KeyError,
            IndexError
        ):

            # Normal race with no sprint
            continue

        if not results:
            continue

        print(
            f"  Sprint found: "
            f"Round {round_number} - "
            f"{race_name}"
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

            all_sprints.append({

                "season": season,

                "round": round_number,

                "race": race_name,

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

                "grid": pd.to_numeric(
                    result.get("grid"),
                    errors="coerce"
                ),

                "position": pd.to_numeric(
                    result.get("position"),
                    errors="coerce"
                ),

                "points": pd.to_numeric(
                    result.get("points", 0),
                    errors="coerce"
                ),

                "status": result.get(
                    "status"
                ),

                "laps": pd.to_numeric(
                    result.get("laps"),
                    errors="coerce"
                )
            })

        time.sleep(
            REQUEST_DELAY
        )


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(
    all_sprints
)

if not df.empty:

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
# Save
# ---------------------------------------------------------

os.makedirs(
    "data",
    exist_ok=True
)

output_file = (
    "data/sprint_results_2018_2025.csv"
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
    "SPRINT COLLECTION COMPLETE"
)

print(
    "=" * 60
)

print(
    "Total rows:",
    len(df)
)

print(
    "Seasons with sprint data:",
    df["season"].nunique()
    if not df.empty
    else 0
)

print(
    "Total sprint sessions:",
    df[
        ["season", "round"]
    ].drop_duplicates().shape[0]
    if not df.empty
    else 0
)

if not df.empty:

    print(
        "\nSprint sessions per season:"
    )

    print(
        df.groupby("season")[
            "round"
        ].nunique()
    )

print(
    f"\nSaved to: {output_file}"
)