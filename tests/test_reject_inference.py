"""Reject inference on a simulated reject region, including self-training (NB06)."""

import numpy as np
import pandas as pd
import pytest

from scorelab import reject_inference as R


@pytest.fixture(scope="module")
def sim(synth):
    dev = synth[synth.split != "oot"].copy()
    return R.simulate_reject_region(dev, incumbent="fico_mid", reject_rate=0.2)


def fit_fn(f):
    import statsmodels.api as sm
    X = sm.add_constant(f[["util", "inq"]].astype(float))
    w = f["weight"] if "weight" in f else None
    return sm.GLM(f["bad"].astype(float), X, family=sm.families.Binomial(), freq_weights=w).fit()


def score_fn(m, f):
    import statsmodels.api as sm
    return np.asarray(m.predict(sm.add_constant(f[["util", "inq"]].astype(float), has_constant="add")))


def test_self_training_labels_every_decline(sim):
    aug, hist = R.self_training(sim, fit_fn, score_fn, step=0.25)
    dec = aug[aug.declined]
    assert dec.bad.notna().all() and set(dec.bad.unique()) <= {0.0, 1.0}
    assert len(aug) == len(sim) and hist.n_pseudo_labelled.sum() == len(dec)
    assert hist.n_remaining.iloc[-1] == 0 and len(hist) == 4
    # The pseudo bad rate tracks the model's expectation, not a 0.5 threshold.
    assert 0.5 * dec.bad_hidden.mean() < dec.bad.mean() < 2 * dec.bad_hidden.mean()
    assert (hist.pseudo_label_accuracy > 0.6).all()


def test_compare_methods_includes_self_training(sim, synth):
    oot = synth[synth.split == "oot"]
    cmp = R.compare_methods(sim, fit_fn, score_fn, oot, multipliers=(1.0,))
    assert "self_training" in set(cmp.method)
    row = cmp.set_index("method").loc["self_training"]
    assert 0 < row.inferred_bad_rate_declined < 1 and row.gini_eval_set > 0.1
    no_st = R.compare_methods(sim, fit_fn, score_fn, oot, multipliers=(1.0,), include_self_training=False)
    assert "self_training" not in set(no_st.method)
