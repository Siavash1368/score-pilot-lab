import numpy as np
import pandas as pd
import pytest

from scorelab import binning as B


def test_numeric_monotone_direction_and_minimums(synth):
    tr = synth[synth.split == "train"]
    s = B.fit_numeric(tr["util"], tr["bad"], direction=1, max_bins=6, min_bin_share=0.05, min_bads=30)
    t = s.table()
    assert s.kind == "numeric" and s.direction == 1 and s.monotone
    assert t.bad_rate.is_monotonic_increasing
    assert (t.n >= 0.05 * len(tr)).all() and (t.n_bad >= 30).all()
    assert len(s.edges) <= 6 and np.isinf(s.edges[-1])
    assert abs(sum((t.n - t.n_bad) / (t.n - t.n_bad).sum() - t.n_bad / t.n_bad.sum()) ) < 1e-9  # shares sum to 0 difference
    assert s.iv > 0.05


def test_auto_direction_finds_safer_and_missing_bin(synth):
    tr = synth[synth.split == "train"]
    s = B.fit_numeric(tr["age"], tr["bad"], direction="auto", min_bads=30)
    assert s.direction == -1
    assert s.missing_own_bin and s.labels[s.missing_bin] == B.MISSING_LABEL
    idx = B.bin_index(tr["age"], s)
    assert (idx[tr["age"].isna().to_numpy()] == s.missing_bin).all()


def test_woe_sign_convention(synth):
    tr = synth[synth.split == "train"]
    s = B.fit_numeric(tr["inq"], tr["bad"], direction=1)
    t = s.table()
    # higher bad rate -> lower WOE (WOE = ln good/bad)
    assert t.woe.is_monotonic_decreasing


def test_fixed_edges_and_transform_roundtrip(synth, tmp_path):
    tr = synth[synth.split == "train"]
    s = B.fit_numeric_with_edges(tr["util"], tr["bad"], [30, 60, 90], name="util")
    assert s.labels[:4] == ["(-inf, 30]", "(30, 60]", "(60, 90]", "> 90"]
    specs = {"util": s, "age": B.fit_numeric(tr["age"], tr["bad"], direction=-1)}
    W = B.transform_woe(tr, specs)
    assert list(W.columns) == ["util_woe", "age_woe"] and len(W) == len(tr)
    B.save_specs(specs, tmp_path / "bins.json"); loaded = B.load_specs(tmp_path / "bins.json")
    assert np.allclose(B.transform_woe(tr, loaded).values, W.values)


def test_categorical_pools_rare_levels(synth):
    tr = synth[synth.split == "train"]
    s = B.fit_categorical(tr["purpose"], tr["bad"].astype(float), name="purpose", min_bin_share=0.02, max_bins=4)
    assert s.kind == "categorical" and len(s.n) <= 4
    assert s.groups["rare1"] == s.groups["rare2"] == s.other_bin
    unseen = pd.Series(["never_seen", "card"])
    idx = B.bin_index(unseen, s)
    assert idx[0] == s.other_bin and idx[1] == s.groups["card"]


def test_csi_zero_on_same_sample_and_positive_on_shift(synth):
    tr = synth[synth.split == "train"]
    s = B.fit_numeric(tr["util"], tr["bad"], direction=1)
    assert B.csi(s, tr["util"])[0] == pytest.approx(0.0, abs=1e-9)
    assert B.csi(s, tr["util"] + 30)[0] > 0.1


def test_iv_summary_strength_labels(synth):
    tr = synth[synth.split == "train"]
    specs = B.fit_all(tr, ["util", "age", "inq"], directions={"util": 1, "age": -1, "inq": 1})
    out = B.iv_summary(specs)
    assert set(out.attribute) == {"util", "age", "inq"} and out.iv.is_monotonic_decreasing
    assert set(out.strength) <= {"useless", "weak", "medium", "strong", "suspicious"}


def test_forced_missing_bin_below_the_floors(synth):
    tr = synth[synth.split == "train"].copy()
    tr.loc[tr.index[:60], "util"] = float("nan")                  # 0.5% missing: below the 5%/2 share floor
    default = B.fit_numeric(tr["util"], tr["bad"], direction=1)
    forced = B.fit_numeric(tr["util"], tr["bad"], direction=1, force_missing_bin=True)
    assert not default.missing_own_bin and default.labels[default.missing_bin].endswith("+ Missing")
    assert forced.missing_own_bin and forced.labels[-1] == B.MISSING_LABEL and forced.n[-1] == 60
    assert np.isfinite(forced.woe[-1])                              # smoothed, finite even when small
    specs = B.fit_all(tr, ["util", "age"], directions={"util": 1, "age": -1}, missing_own_bin=["util"])
    assert specs["util"].missing_own_bin
    prof = B.missing_profile(tr, ["util"])
    assert prof.query("missing").n.sum() == 60 and prof.n.sum() == len(tr)
