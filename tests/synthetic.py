"""Generate a small random frame with the competition schema.

Only used to check that the pipeline runs; the values carry no signal and
any accuracy measured on it is meaningless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spaceship.features import SPEND_COLS


def make_frame(n: int, seed: int, with_target: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.sort(rng.integers(1, n // 2 + 1, size=n))
    pos = pd.Series(groups).groupby(groups).cumcount() + 1
    df = pd.DataFrame(
        {
            "PassengerId": [f"{g:04d}_{p:02d}" for g, p in zip(groups, pos)],
            "HomePlanet": rng.choice(["Earth", "Europa", "Mars", None], size=n, p=[0.5, 0.25, 0.2, 0.05]),
            "CryoSleep": rng.choice([True, False, None], size=n, p=[0.35, 0.6, 0.05]),
            "Cabin": [
                None if rng.random() < 0.03 else f"{rng.choice(list('ABCDEFGT'))}/{rng.integers(0, 1900)}/{rng.choice(['P', 'S'])}"
                for _ in range(n)
            ],
            "Destination": rng.choice(["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22", None], size=n, p=[0.65, 0.2, 0.1, 0.05]),
            "Age": np.where(rng.random(n) < 0.03, np.nan, rng.integers(0, 80, size=n).astype(float)),
            "VIP": rng.choice([True, False, None], size=n, p=[0.02, 0.95, 0.03]),
            "Name": [None if rng.random() < 0.03 else f"{rng.choice(['Ann', 'Bob', 'Cy'])} {rng.choice(['Lee', 'Oak', 'Vox', 'Ray'])}" for _ in range(n)],
        }
    )
    for c in SPEND_COLS:
        spend = np.where(rng.random(n) < 0.6, 0.0, rng.exponential(300, size=n).round())
        spend = np.where(rng.random(n) < 0.03, np.nan, spend)
        df[c] = spend
    if with_target:
        df["Transported"] = rng.random(n) < 0.5
    return df
