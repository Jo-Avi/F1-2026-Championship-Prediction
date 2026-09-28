import pandas as pd


df = pd.read_csv("data/master_race_dataset.csv")

print("=" * 60)
print("MASTER DATASET QUALITY CHECK")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nExpected race-result rows:")
race_df = pd.read_csv("data/race_results_2018_2025.csv")
print(len(race_df))

print("\nActual master rows:")
print(len(df))


# ---------------------------------------------------------
# CHECK DUPLICATES
# ---------------------------------------------------------

key_columns = [
    "season",
    "round",
    "driver_id"
]

duplicates = df[
    df.duplicated(
        subset=key_columns,
        keep=False
    )
]

print("\nDuplicate driver-race records:")
print(len(duplicates))

if not duplicates.empty:
    print("\nExamples:")
    print(
        duplicates[
            key_columns +
            ["driver", "constructor", "position",
             "qualifying_position"]
        ].head(20)
    )


# ---------------------------------------------------------
# CHECK QUALIFYING DATA
# ---------------------------------------------------------

qualifying_df = pd.read_csv(
    "data/qualifying_results_2018_2025.csv"
)

qualifying_duplicates = qualifying_df[
    qualifying_df.duplicated(
        subset=key_columns,
        keep=False
    )
]

print("\nDuplicate records in qualifying dataset:")
print(len(qualifying_duplicates))

if not qualifying_duplicates.empty:
    print("\nQualifying duplicate examples:")
    print(
        qualifying_duplicates[
            key_columns +
            ["driver", "qualifying_position"]
        ].head(20)
    )


# ---------------------------------------------------------
# CHECK RACE DATA
# ---------------------------------------------------------

race_duplicates = race_df[
    race_df.duplicated(
        subset=key_columns,
        keep=False
    )
]

print("\nDuplicate records in race dataset:")
print(len(race_duplicates))


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values:")
print(
    df.isnull().sum()
)


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)