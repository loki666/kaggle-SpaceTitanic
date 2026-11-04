"""Feature engineering for the Spaceship Titanic competition.

The train and test frames are processed together so that group-level
statistics (people sharing a PassengerId group, a cabin or a surname) and
imputation rules see the full passenger manifest. Nothing here uses the
target column, so there is no leakage from doing this on the concatenation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
TARGET = "Transported"

CATEGORICAL = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "AgeGroup",
    "CabinRegion",
    # spatial interactions: the event looks localised, so where a cabin sits matters
    "DeckSide",
    "DeckSideBand",
    "PlanetDeck",
    # behavioural proxy for where the passenger was: pod, cabin, spa/VR, venues
    "LocationProxy",
]
NUMERIC = [
    "Age",
    "CabinNum",
    "GroupSize",
    "FamilySize",
    "TotalSpend",
    "NoSpending",
    "LuxurySpend",
    "BasicSpend",
    "LogVenueOverLuxury",
    "SpendCount",
    "GroupSpendMean",
    "GroupCryoRate",
    "GroupNoSpendRate",
    "CabinMates",
    "CabinSpendMean",
    "NeighbourSpendMean",
    "CabinMissing",
    "NameMissing",
    "SpendMissingCount",
] + SPEND_COLS

NEIGHBOUR_RADIUS = 20  # cabins either side, on the same deck and side


def _split_ids(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["PassengerId"].str.split("_", expand=True)
    df["Group"] = parts[0]
    df["GroupPos"] = parts[1].astype(int)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    return df


def _split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    df["CabinMissing"] = df["Cabin"].isna().astype(int)
    parts = df["Cabin"].str.split("/", expand=True)
    if parts.shape[1] < 3:  # all-NaN column edge case
        parts = parts.reindex(columns=range(3))
    df["Deck"] = parts[0]
    df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
    df["Side"] = parts[2]
    return df


def _split_name(df: pd.DataFrame) -> pd.DataFrame:
    df["NameMissing"] = df["Name"].isna().astype(int)
    df["Surname"] = df["Name"].str.split(" ").str[-1]
    df["FamilySize"] = df.groupby("Surname")["PassengerId"].transform("count")
    df["FamilySize"] = df["FamilySize"].fillna(0).astype(int)
    return df


def _fill_from_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Fill a missing categorical value with the value shared by the rest of
    the passenger's group (groups travel together and share e.g. HomePlanet)."""
    group_mode = (
        df.dropna(subset=[col])
        .groupby("Group")[col]
        .agg(lambda s: s.mode().iloc[0])
    )
    fill = df["Group"].map(group_mode)
    df[col] = df[col].fillna(fill)
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    # --- spending ---------------------------------------------------------
    df["SpendMissingCount"] = df[SPEND_COLS].isna().sum(axis=1)
    # People in cryosleep cannot spend: fill their missing spend with 0.
    cryo_true = df["CryoSleep"] == True  # noqa: E712 (object dtype with NaN)
    for c in SPEND_COLS:
        df.loc[cryo_true & df[c].isna(), c] = 0.0
    # Anyone with positive spend is certainly awake.
    any_spend = df[SPEND_COLS].fillna(0).sum(axis=1) > 0
    df.loc[df["CryoSleep"].isna() & any_spend, "CryoSleep"] = False
    # Remaining missing spend: 0 (median of every spend column is 0).
    df[SPEND_COLS] = df[SPEND_COLS].fillna(0.0)
    # Remaining missing CryoSleep: zero spenders are most often asleep.
    zero_spend = df[SPEND_COLS].sum(axis=1) == 0
    df.loc[df["CryoSleep"].isna() & zero_spend, "CryoSleep"] = True
    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(bool)

    # --- HomePlanet -------------------------------------------------------
    df = _fill_from_group(df, "HomePlanet")
    # Decks A/B/C/T are exclusively Europa, deck G exclusively Earth.
    df.loc[df["HomePlanet"].isna() & df["Deck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = "Europa"
    df.loc[df["HomePlanet"].isna() & (df["Deck"] == "G"), "HomePlanet"] = "Earth"
    df["HomePlanet"] = df["HomePlanet"].fillna("Earth")

    # --- Destination / VIP ------------------------------------------------
    df = _fill_from_group(df, "Destination")
    df["Destination"] = df["Destination"].fillna("TRAPPIST-1e")
    df["VIP"] = df["VIP"].fillna(False).astype(bool)

    # --- Cabin ------------------------------------------------------------
    df = _fill_from_group(df, "Deck")
    df = _fill_from_group(df, "Side")
    df = _fill_from_group(df, "CabinNum")
    # Fallback deck by home planet (the most common deck for each planet).
    deck_by_planet = {"Earth": "G", "Europa": "B", "Mars": "F"}
    df["Deck"] = df["Deck"].fillna(df["HomePlanet"].map(deck_by_planet)).fillna("F")
    df["Side"] = df["Side"].fillna("S")
    df["CabinNum"] = df["CabinNum"].fillna(df["CabinNum"].median())

    # --- Age --------------------------------------------------------------
    df["AgeMissing"] = df["Age"].isna().astype(int)
    age_median = df.groupby(["HomePlanet", "CryoSleep"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_median).fillna(df["Age"].median())
    return df


def _neighbour_mean(df: pd.DataFrame, col: str, radius: int) -> np.ndarray:
    """Mean of ``col`` over passengers on the same deck/side whose cabin number
    is within ``radius`` of the passenger's own (self included)."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby(["Deck", "Side"]):
        order = np.argsort(g["CabinNum"].to_numpy())
        num = g["CabinNum"].to_numpy()[order]
        val = g[col].to_numpy(dtype=float)[order]
        idx = g.index.to_numpy()[order]
        csum = np.concatenate([[0.0], np.cumsum(val)])
        lo = np.searchsorted(num, num - radius)
        hi = np.searchsorted(num, num + radius, side="right")
        out[idx] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def _location_proxy(df: pd.DataFrame) -> pd.Series:
    """Where the passenger most likely was, inferred from cryosleep and from
    which services their spending went to."""
    total = df["TotalSpend"].replace(0, np.nan)
    cabin = df["RoomService"] / total
    spa_vr = (df["Spa"] + df["VRDeck"]) / total
    venue = (df["FoodCourt"] + df["ShoppingMall"]) / total
    loc = pd.Series("mixed", index=df.index)
    loc[venue > 0.5] = "venue"
    loc[spa_vr > 0.5] = "spa_vr"
    loc[cabin > 0.5] = "cabin"
    loc[df["TotalSpend"] == 0] = "nospend"
    loc[df["CryoSleep"].astype(bool)] = "pod"
    return loc


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["NoSpending"] = (df["TotalSpend"] == 0).astype(int)
    df["LuxurySpend"] = df[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
    df["BasicSpend"] = df[["FoodCourt", "ShoppingMall"]].sum(axis=1)
    df["LogVenueOverLuxury"] = np.log1p(df["BasicSpend"]) - np.log1p(df["LuxurySpend"])
    df["SpendCount"] = (df[SPEND_COLS] > 0).sum(axis=1)
    df["LocationProxy"] = _location_proxy(df)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 30, 50, 200],
        labels=["child", "teen", "young", "late20s", "adult", "senior"],
    ).astype(str)

    # Cabin numbers cluster into regions that correlate with the target.
    df["CabinRegion"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 300, 600, 900, 1200, 1500, 1800, 10_000],
        labels=[f"R{i}" for i in range(7)],
    ).astype(str)

    df["GroupSpendMean"] = df.groupby("Group")["TotalSpend"].transform("mean")
    df["GroupCryoRate"] = df.groupby("Group")["CryoSleep"].transform("mean")
    df["GroupNoSpendRate"] = df.groupby("Group")["NoSpending"].transform("mean")
    cabin_key = ["Deck", "CabinNum", "Side"]
    df["CabinMates"] = df.groupby(cabin_key)["PassengerId"].transform("count")
    df["CabinSpendMean"] = df.groupby(cabin_key)["TotalSpend"].transform("mean")
    df["NeighbourSpendMean"] = _neighbour_mean(df, "TotalSpend", NEIGHBOUR_RADIUS)

    df["DeckSide"] = df["Deck"] + "_" + df["Side"]
    df["DeckSideBand"] = df["DeckSide"] + "_" + (df["CabinNum"] // 200).astype(int).astype(str)
    df["PlanetDeck"] = df["HomePlanet"] + "_" + df["Deck"]
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_train_with_target, X_test) feature frames.

    Both frames keep ``PassengerId``; the train frame keeps ``Transported`` as int.
    """
    train = train.copy()
    test = test.copy()
    train["_is_train"] = 1
    test["_is_train"] = 0
    if TARGET not in test.columns:
        test[TARGET] = np.nan
    df = pd.concat([train, test], ignore_index=True)

    df = _split_ids(df)
    df = _split_cabin(df)
    df = _split_name(df)
    df = _impute(df)
    df = _derive(df)

    for c in ["CryoSleep", "VIP"]:
        df[c] = df[c].astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)

    keep = ["PassengerId"] + CATEGORICAL + NUMERIC
    out_train = df[df["_is_train"] == 1][keep + [TARGET]].reset_index(drop=True)
    out_train[TARGET] = out_train[TARGET].astype(int)
    out_test = df[df["_is_train"] == 0][keep].reset_index(drop=True)
    return out_train, out_test


def feature_columns() -> list[str]:
    return CATEGORICAL + NUMERIC
