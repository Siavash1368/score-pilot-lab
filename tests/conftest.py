"""Synthetic scorecard data shared by the tests: a Bernoulli outcome driven by three attributes
(one riskier, one safer, one categorical) with informative missingness, no database needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synth():
    rng = np.random.default_rng(7)
    n = 20000
    util = rng.uniform(0, 120, n)                      # riskier as it rises
    age = rng.gamma(4, 40, n)                          # safer as it rises
    inq = rng.poisson(1.0, n).astype(float)            # riskier
    purpose = rng.choice(["card", "debt", "home", "other", "rare1", "rare2"], n, p=[0.3, 0.5, 0.1, 0.08, 0.01, 0.01])
    eta = -2.6 + 0.012 * util - 0.004 * age + 0.25 * inq + np.where(purpose == "card", -0.3, 0.0)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    age = np.where(rng.uniform(size=n) < 0.1, np.nan, age)          # 10% missing, at random
    df = pd.DataFrame({"util": util, "age": age, "inq": inq, "purpose": purpose, "bad": y})
    df["fico_mid"] = 700 - 0.5 * util + 0.1 * np.nan_to_num(age, nan=150) - 5 * inq + rng.normal(0, 25, n)
    df["split"] = np.where(np.arange(n) < 12000, "train", np.where(np.arange(n) < 16000, "validate", "oot"))
    df["vintage_year"] = np.where(df.split == "train", 2013, np.where(df.split == "validate", 2015, 2016))
    return df
