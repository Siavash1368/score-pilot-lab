"""Attribute cleaning (NB03): caps, missing semantics and leakage control."""

import numpy as np
import pandas as pd

from scorelab import attributes as A


def test_percent_cap_keeps_missing_as_missing():
    # A missing utilisation means "no such account", not 150% utilised.
    df = pd.DataFrame({"revol_util": [10.0, 400.0, np.nan], "bc_util": [np.nan, 90.0, 900.0],
                       "inq_last_6mths": [1.0, -1.0, np.nan]})
    out = A.build_attributes(df, names=["revol_util", "bc_util", "inq_last_6mths"])
    assert out["revol_util"].tolist()[:2] == [10.0, A.PERCENT_CAP] and np.isnan(out["revol_util"].iloc[2])
    assert np.isnan(out["bc_util"].iloc[0]) and out["bc_util"].iloc[2] == A.PERCENT_CAP
    assert out["revol_util"].isna().sum() == df["revol_util"].isna().sum()
    # Negative counts are reporting artefacts and become missing; missing stays missing.
    assert out["inq_last_6mths"].iloc[0] == 1.0 and out["inq_last_6mths"].iloc[1:].isna().all()


def test_every_attribute_source_is_a_permitted_input():
    assert A.leakage_check().empty
