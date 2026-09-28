import os
import requests
import pandas as pd


BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASON = 2026

HEADERS = {
    "User-Agent": "F1-Championship-Predictor/1.0"
}


def get_json(endpoint):
    """Fetch every page and combine race result lists by round.

    The API's `limit` applies to result rows, not whole races. A single page
    can therefore end partway through a race, and relying on just the first
    100 records silently drops later rounds (or drivers in the last round).
    """
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    mrdata = payload["MRData"]
    total = int(mrdata.get("total", 0))
    limit = int(mrdata.get("limit", 100))
    race_table = mrdata.get("RaceTable")
    if not race_table or total <= limit:
        return payload

    races_by_round = {}
    for race in race_table.get("Races", []):
        races_by_round[race["round"]] = race

    for offset in range(limit, total, limit):
        separator = "&" if "?" in endpoint else "?"
        page_url = f"{BASE_URL}/{endpoint}{separator}offset={offset}"
        page = requests.get(page_url, headers=HEADERS, timeout=30)
        page.raise_for_status()
        page_mrdata = page.json()["MRData"]
        for race in page_mrdata.get("RaceTable", {}).get("Races", []):
            key = race["round"]
            if key not in races_by_round:
                races_by_round[key] = race
                continue
            existing = races_by_round[key]
            for result_key in ("Results", "QualifyingResults", "SprintResults"):
                if result_key in race:
                    existing.setdefault(result_key, []).extend(race[result_key])

    race_table["Races"] = sorted(
        races_by_round.values(), key=lambda race: int(race["round"])
    )
    mrdata["limit"] = str(total)
    return payload


def collect_race_results():

    print("Collecting 2026 race results...")

    data = get_json(f"{SEASON}/results/?limit=100")

    races = data["MRData"]["RaceTable"]["Races"]

    records = []

    for race in races:

        for result in race.get("Results", []):

            records.append({
                "season": SEASON,
                "round": int(race["round"]),
                "race": race["raceName"],
                "circuit": race["Circuit"]["circuitName"],

                "driver_id": result["Driver"]["driverId"],
                "driver": (
                    result["Driver"]["givenName"]
                    + " "
                    + result["Driver"]["familyName"]
                ),

                "constructor": result["Constructor"]["name"],

                "grid": int(result["grid"]),
                "position": int(result["position"]),
                "points": float(result["points"]),
                "status": result["status"],
                "laps": int(result["laps"])
            })

    df = pd.DataFrame(records)

    print("Race result rows:", len(df))

    return df


def collect_qualifying_results():

    print("Collecting 2026 qualifying results...")

    data = get_json(f"{SEASON}/qualifying/?limit=100")

    races = data["MRData"]["RaceTable"]["Races"]

    records = []

    for race in races:

        for result in race.get("QualifyingResults", []):

            records.append({
                "season": SEASON,
                "round": int(race["round"]),

                "driver_id": result["Driver"]["driverId"],

                "driver": (
                    result["Driver"]["givenName"]
                    + " "
                    + result["Driver"]["familyName"]
                ),

                "constructor": result["Constructor"]["name"],

                "qualifying_position": int(result["position"]),

                "Q1": result.get("Q1"),
                "Q2": result.get("Q2"),
                "Q3": result.get("Q3")
            })

    df = pd.DataFrame(records)

    print("Qualifying rows:", len(df))

    return df


def collect_sprint_results():

    print("Collecting 2026 sprint results...")

    data = get_json(f"{SEASON}/sprint/?limit=100")

    races = data["MRData"]["RaceTable"]["Races"]

    records = []

    for race in races:

        for result in race.get("SprintResults", []):

            records.append({
                "season": SEASON,
                "round": int(race["round"]),

                "driver_id": result["Driver"]["driverId"],

                "driver": (
                    result["Driver"]["givenName"]
                    + " "
                    + result["Driver"]["familyName"]
                ),

                "constructor": result["Constructor"]["name"],

                "grid": int(result["grid"]),
                "position": int(result["position"]),
                "points": float(result["points"]),
                "status": result["status"]
            })

    df = pd.DataFrame(records)

    print("Sprint rows:", len(df))

    return df


def collect_driver_standings():

    print("Collecting 2026 driver standings...")

    data = get_json(
        f"{SEASON}/driverstandings/?limit=100"
    )

    standings_lists = (
        data["MRData"]["StandingsTable"]
        .get("StandingsLists", [])
    )

    records = []

    for standings in standings_lists:

        round_number = int(standings["round"])

        for driver in standings.get("DriverStandings", []):

            records.append({
                "season": SEASON,
                "round": round_number,

                "driver_id": driver["Driver"]["driverId"],

                "driver": (
                    driver["Driver"]["givenName"]
                    + " "
                    + driver["Driver"]["familyName"]
                ),

                "championship_position": int(
                    driver["position"]
                ),

                "championship_points": float(
                    driver["points"]
                ),

                "wins": int(driver["wins"])
            })

    df = pd.DataFrame(records)

    print("Driver standings rows:", len(df))

    return df


def collect_constructor_standings():

    print("Collecting 2026 constructor standings...")

    data = get_json(
        f"{SEASON}/constructorstandings/?limit=100"
    )

    standings_lists = (
        data["MRData"]["StandingsTable"]
        .get("StandingsLists", [])
    )

    records = []

    for standings in standings_lists:

        round_number = int(standings["round"])

        for constructor in standings.get(
            "ConstructorStandings", []
        ):

            records.append({
                "season": SEASON,
                "round": round_number,

                "constructor_id": constructor[
                    "Constructor"
                ]["constructorId"],

                "constructor": constructor[
                    "Constructor"
                ]["name"],

                "championship_position": int(
                    constructor["position"]
                ),

                "championship_points": float(
                    constructor["points"]
                ),

                "wins": int(
                    constructor["wins"]
                )
            })

    df = pd.DataFrame(records)

    print(
        "Constructor standings rows:",
        len(df)
    )

    return df


# ==================================================
# MAIN
# ==================================================

print("=" * 60)
print("COLLECTING 2026 F1 DATA")
print("=" * 60)

os.makedirs("data", exist_ok=True)


race_df = collect_race_results()

qualifying_df = collect_qualifying_results()

sprint_df = collect_sprint_results()

driver_standings_df = collect_driver_standings()

constructor_standings_df = collect_constructor_standings()


# --------------------------------------------------
# Save files
# --------------------------------------------------

race_df.to_csv(
    "data/race_results_2026.csv",
    index=False
)

qualifying_df.to_csv(
    "data/qualifying_results_2026.csv",
    index=False
)

sprint_df.to_csv(
    "data/sprint_results_2026.csv",
    index=False
)

driver_standings_df.to_csv(
    "data/driver_standings_2026.csv",
    index=False
)

constructor_standings_df.to_csv(
    "data/constructor_standings_2026.csv",
    index=False
)


print("\n" + "=" * 60)
print("2026 DATA COLLECTION COMPLETE")
print("=" * 60)

print("\nSaved files:")

print("data/race_results_2026.csv")
print("data/qualifying_results_2026.csv")
print("data/sprint_results_2026.csv")
print("data/driver_standings_2026.csv")
print("data/constructor_standings_2026.csv")


print("\n2026 race rounds collected:")

if not race_df.empty:
    print(
        sorted(
            race_df["round"].unique()
        )
    )

print("\nDONE")
