"""Discrete-time hazard: event timing matches the label rule, person-period expansion, CIF (NB07d)."""

import numpy as np
import pandas as pd
import pytest

from scorelab import config, survival as SV


def test_event_times_match_the_label_rule():
    df = pd.DataFrame({
        "loan_status": ["Charged Off", "Charged Off", "Fully Paid", "Fully Paid", "Current", "Late (31-120 days)", "Default"],
        "months_to_last_payment": [10, 25, 7, 30, 26, 19, np.nan]})
    out = SV.event_times(df)
    # Charged off with last payment in month 10 -> 90 DPD in month 14: default.
    # Last payment in month 25 -> 90 DPD in month 29 > 24: censored at 24 (good under the label).
    # Paid off in month 7: prepay at 7. Paid off after the window: censored at 24.
    # Late with last payment in month 19 -> 90 DPD at 23: default. Never paid -> 90 DPD at month 4.
    assert list(out.exit_type) == ["default", "censored", "prepay", "censored", "censored", "default", "default"]
    assert list(out.exit_month) == [14, 24, 7, 24, 24, 23, 4]
    fm, d90 = config.MONTHS_LAST_PAYMENT_TO_FIRST_MISS, config.MONTHS_FIRST_MISS_TO_90DPD
    assert out.exit_month.iloc[0] == 10 + fm + d90


def test_person_period_rows_and_outcomes():
    df = pd.DataFrame({"exit_month": [3, 2, 1], "exit_type": ["default", "prepay", "censored"], "x": [0.1, 0.2, 0.3]})
    pp = SV.person_period(df, ["x"])
    assert len(pp) == 6 and list(pp.month) == [1, 2, 3, 1, 2, 1]
    assert list(pp.outcome) == [0, 0, 1, 0, 2, 0] and list(pp.x) == [0.1] * 3 + [0.2] * 2 + [0.3]


@pytest.fixture(scope="module")
def simulated():
    """Loans with constant monthly hazards of default (rising with x) and prepayment."""
    rng = np.random.default_rng(5)
    n = 6000
    x = rng.normal(size=n)
    hd = 1 / (1 + np.exp(-(-5.0 + 0.8 * x))); hp = np.full(n, 0.02)
    exit_month, exit_type = np.full(n, 24), np.array(["censored"] * n, dtype=object)
    alive = np.ones(n, dtype=bool)
    for t in range(1, 25):
        u = rng.uniform(size=n)
        d = alive & (u < hd); p = alive & ~d & (u < hd + hp)
        exit_month[d | p] = t; exit_type[d] = "default"; exit_type[p] = "prepay"
        alive &= ~(d | p)
    return pd.DataFrame({"x": x, "exit_month": exit_month, "exit_type": exit_type})


def test_hazard_model_cif_matches_empirical(simulated):
    pp = SV.person_period(simulated, ["x"])
    hm = SV.HazardModel(["x"]).fit(pp)
    cif = hm.cumulative_incidence(simulated)
    assert cif.shape == (len(simulated), 24) and np.all(np.diff(cif, axis=1) >= -1e-12)
    emp = SV.empirical_cif(simulated)
    assert cif[:, -1].mean() == pytest.approx(emp.cif_default.iloc[-1], abs=0.01)
    # Prepayment as censoring always gives a higher default probability than the competing-risk CIF.
    assert np.all(hm.cumulative_incidence(simulated, competing=False)[:, -1] >= cif[:, -1] - 1e-12)
    # Three outcomes, continue (0) as the reference: coefficients are log-hazards against continuing.
    assert hm.classes_ == [SV.CONTINUE, SV.DEFAULT, SV.PREPAY]
    base, cov = hm.baseline_table()
    assert len(base) == 8 and cov.loc[0, "default_coef"] == pytest.approx(0.8, abs=0.2)
    assert cov.loc[0, "prepay_coef"] == pytest.approx(0.0, abs=0.15)            # prepay hazard does not depend on x
    assert base.prepay_coef.mean() == pytest.approx(np.log(0.02 / 0.98), abs=0.3)


def test_predicted_cif_by_group_and_empirical_by_group(simulated):
    pp = SV.person_period(simulated, ["x"])
    hm = SV.HazardModel(["x"]).fit(pp)
    band = pd.Series(np.where(simulated.x > 0, "risky", "safe"), name="band")
    pred = SV.predicted_cif_by_group(hm.cumulative_incidence(simulated), band)
    emp = SV.empirical_cif(simulated.assign(band=band), group="band")
    last = pred[pred.month == 24].set_index("band").cif_default_pred
    assert last["risky"] > last["safe"]
    e24 = emp[emp.month == 24].set_index("band").cif_default
    assert last["risky"] == pytest.approx(e24["risky"], abs=0.02)
