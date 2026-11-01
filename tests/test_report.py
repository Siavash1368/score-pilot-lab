"""Pilot report pieces added for Stage 3: indeterminate sensitivity, score scaling, caveat and appendix (NB08)."""

import numpy as np
import pandas as pd
import pytest

from scorelab import report as REP, scorecard as S, config


@pytest.fixture(scope="module")
def scored(synth):
    oot = synth[synth.split == "oot"].copy()
    eta = -2.6 + 0.012 * oot["util"] + 0.25 * oot["inq"]
    card = S.Scorecard(attributes=[], intercept=0.0, coef={})
    oot["eta"] = eta; oot["p_bad"] = 1 / (1 + np.exp(-eta)); oot["score"] = S.scale_probability(oot["p_bad"], card)
    oot["term_months"] = np.where(np.arange(len(oot)) % 3 == 0, 60, 36)
    return oot, card


def test_scale_probability_matches_card_scaling(scored):
    _, card = scored
    # At the base odds the score is the base score; doubling the odds adds PDO points.
    p_base = 1 / (1 + card.base_odds)
    assert S.scale_probability([p_base], card)[0] == card.base_score
    p_double = 1 / (1 + 2 * card.base_odds)
    assert S.scale_probability([p_double], card)[0] == card.base_score + card.pdo
    assert S.scale_probability([1e-12], card)[0] == config.SCORE_MAX            # clipped to the scale


def test_indeterminate_sensitivity_three_treatments(scored):
    oot, _ = scored
    lab, ind = oot.iloc[:3000], oot.iloc[3000:3200].assign(bad=np.nan)
    t = REP.indeterminate_sensitivity(lab, ind, {"card": "score", "fico": "fico_mid"}).set_index("treatment")
    n_bad = int(lab.bad.sum())
    assert t.loc["excluded (rule in force)", "bad_rate"] == round(n_bad / 3000, 4)
    assert t.loc["counted as good", "bad_rate"] == round(n_bad / 3200, 4)
    assert t.loc["counted as bad", "bad_rate"] == round((n_bad + 200) / 3200, 4)
    assert {"gini_card", "ks_card", "gini_fico", "ks_fico"} <= set(t.columns)


def test_report_prints_incumbent_caveat_and_appendix(scored, tmp_path):
    oot, _ = scored
    ex = REP.pilot_exhibits(oot, oot["score"], n_boot=20, level="test")
    md = REP.assemble_pilot_report(ex, "t", "l", incumbent_caveat="FICO is truncated", path=tmp_path / "r.md",
                                   extra_sections={"Indeterminates": "table here"},
                                   appendix=REP.challenger_appendix("GBM", ex, note="recalibrated"))
    assert md.count("FICO is truncated") == 4                                     # discrimination, gains, swap set, segments
    assert "## Indeterminates" in md and "## Appendix: GBM" in md and (tmp_path / "r.md").exists()


def test_swap_set_note_printed_in_section_and_appendix(scored):
    oot, _ = scored
    ex = REP.pilot_exhibits(oot, oot["score"], n_boot=20, level="test")
    note = "Cutoff applied to booked loans only."
    md = REP.assemble_pilot_report(ex, "t", "l", swap_set_note=note,
                                   appendix=REP.challenger_appendix("GBM", ex, swap_set_note=note))
    swap = md[md.index("## 3. Swap set"):md.index("## 4. Calibration")]
    assert note in swap and md.count(note) == 2
