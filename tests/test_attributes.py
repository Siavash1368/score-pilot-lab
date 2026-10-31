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


def test_bucket_table_counts_and_missing_row():
    df = pd.DataFrame({"x": [0, 0, 1, 2, 5, np.nan, np.nan], "bad": [0, 1, 0, 1, 1, 0, 0]})
    t = A.bucket_table(df, "x", [0, 1, 2])
    assert list(t["x"]) == ["<= 0", "(0, 1]", "(1, 2]", "> 2", "Missing"]
    assert list(t.n) == [2, 1, 1, 1, 2] and list(t.bads) == [1, 0, 1, 1, 0]
    assert t.bad_rate.iloc[0] == 0.5 and abs(t.share.sum() - 1.0) < 1e-3     # shares are rounded to 4 places


def test_no_bankcard_limit_makes_bankcard_fields_missing():
    df = pd.DataFrame({"total_bc_limit": [0.0, np.nan, 5000.0], "bc_open_to_buy": [0.0, 10.0, 0.0],
                       "percent_bc_gt_75": [np.nan, 50.0, 100.0], "bc_util": [0.0, 20.0, np.nan]})
    out = A.build_attributes(df, names=["bc_util"])
    assert out.loc[:1, ["bc_open_to_buy", "percent_bc_gt_75", "bc_util"]].isna().all().all()   # no limit -> no bankcard
    assert out.loc[2, "bc_open_to_buy"] == 0.0 and out.loc[2, "percent_bc_gt_75"] == 100.0    # real zero kept
    assert out["total_bc_limit"].tolist()[0] == 0.0                                          # the limit itself is kept
    assert set(A.MISSING_REASONS) == {"bc_open_to_buy", "percent_bc_gt_75"}
