# Pilot report: bureau-attribute scorecard vs FICO

*Lender: Lending Club (out-of-time 2016 vintage) · Validation level: out-of-time (2016 vintage) · n = 419,611 · bad rate = 12.06% · incumbent = fico_mid · approval rate for swap set = 70%*

## 1. Discrimination

| score | gini | gini_se | ks |
|---|---|---|---|
| challenger | 0.2838 | 0.0025 | 0.2078 |
| incumbent | 0.1805 | 0.0026 | 0.1354 |

Gini lift challenger - incumbent: **0.1033** (DeLong z = 35.65, p = 0; bootstrap 95% CI 0.0978 to 0.1091). Level: out-of-time (2016 vintage).

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*

DRAFT (Claude, 2026-11-01, for review): On the 2016 out-of-time vintage the card's Gini is 0.2838 against 0.1805 for the FICO midpoint, a lift of 0.1033 (bootstrap 95% CI 0.0978 to 0.1091). FICO is truncated in this population because Lending Club used it to approve, so this is lift over a truncated incumbent, not over FICO on a full applicant population.

## 2. Gains tables

Challenger:

| band | n | bads | bad_rate | cum_share_population | cum_share_bads | cum_share_goods | ks_at_band | lift |
|---|---|---|---|---|---|---|---|---|
| 1 | 41962 | 9151 | 0.2181 | 0.1000 | 0.1809 | 0.0889 | 0.0920 | 1.8090 |
| 2 | 41961 | 7669 | 0.1828 | 0.2000 | 0.3325 | 0.1818 | 0.1507 | 1.5160 |
| 3 | 41961 | 6598 | 0.1572 | 0.3000 | 0.4629 | 0.2777 | 0.1852 | 1.3040 |
| 4 | 41961 | 5861 | 0.1397 | 0.4000 | 0.5787 | 0.3755 | 0.2032 | 1.1590 |
| 5 | 41961 | 5222 | 0.1244 | 0.5000 | 0.6820 | 0.4751 | 0.2069 | 1.0320 |
| 6 | 41961 | 4526 | 0.1079 | 0.6000 | 0.7714 | 0.5765 | 0.1949 | 0.8950 |
| 7 | 41961 | 3850 | 0.0918 | 0.7000 | 0.8475 | 0.6798 | 0.1677 | 0.7610 |
| 8 | 41961 | 3296 | 0.0785 | 0.8000 | 0.9127 | 0.7846 | 0.1281 | 0.6510 |
| 9 | 41961 | 2617 | 0.0624 | 0.9000 | 0.9644 | 0.8912 | 0.0732 | 0.5180 |
| 10 | 41961 | 1801 | 0.0429 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.3560 |

Incumbent:

| band | n | bads | bad_rate | cum_share_population | cum_share_bads | cum_share_goods | ks_at_band | lift |
|---|---|---|---|---|---|---|---|---|
| 1 | 41962 | 6916 | 0.1648 | 0.1000 | 0.1367 | 0.0950 | 0.0417 | 1.3670 |
| 2 | 41961 | 6608 | 0.1575 | 0.2000 | 0.2673 | 0.1908 | 0.0765 | 1.3060 |
| 3 | 41961 | 6265 | 0.1493 | 0.3000 | 0.3912 | 0.2875 | 0.1037 | 1.2380 |
| 4 | 41961 | 6001 | 0.1430 | 0.4000 | 0.5098 | 0.3850 | 0.1248 | 1.1860 |
| 5 | 41961 | 5387 | 0.1284 | 0.5000 | 0.6163 | 0.4841 | 0.1322 | 1.0650 |
| 6 | 41961 | 5145 | 0.1226 | 0.6000 | 0.7180 | 0.5838 | 0.1342 | 1.0170 |
| 7 | 41961 | 4622 | 0.1101 | 0.7000 | 0.8093 | 0.6850 | 0.1243 | 0.9130 |
| 8 | 41961 | 3916 | 0.0933 | 0.8000 | 0.8867 | 0.7881 | 0.0986 | 0.7740 |
| 9 | 41961 | 3382 | 0.0806 | 0.9000 | 0.9536 | 0.8927 | 0.0609 | 0.6690 |
| 10 | 41961 | 2349 | 0.0560 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.4640 |

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*

DRAFT (Claude, 2026-11-01, for review): The card's riskiest decile has a bad rate of 21.81% and holds 18.09% of all bads; FICO's riskiest decile has 16.48% and holds 13.67%. The card concentrates more of the bads in the loans it would decline first.

## 3. Swap set at matched approval rate

**Population.** The 70% cutoff is applied to loans Lending Club already approved (the booked 2016 vintage). The swap set therefore shows how each score would reorder approvals within the booked population; it does not show outcomes for applicants Lending Club declined, whose outcomes are not observed (NB06).

|  | challenger_approve | challenger_decline |
|---|---|---|
| incumbent_approve | 226763 | 66965 |
| incumbent_decline | 66965 | 58918 |

Swap-ins: 66,965 loans, bad rate 11.99%. Swap-outs: 66,965 loans, bad rate 17.48%. Approved-book bad rate: incumbent 10.50%, challenger 9.25% (125 bads avoided per 10,000 approvals).

| approval_rate | n_swap_in | n_swap_out | bad_rate_swap_in | bad_rate_swap_out | bad_rate_both | bad_rate_approved_incumbent | bad_rate_approved_challenger | bads_avoided_per_10k_approved |
|---|---|---|---|---|---|---|---|---|
| 0.5000 | 73166 | 73166 | 0.0986 | 0.1436 | 0.0650 | 0.0924 | 0.0767 | 157.0975 |
| 0.6000 | 72877 | 72877 | 0.1090 | 0.1576 | 0.0747 | 0.0987 | 0.0846 | 140.8445 |
| 0.7000 | 66965 | 66965 | 0.1199 | 0.1748 | 0.0844 | 0.1050 | 0.0925 | 125.2519 |
| 0.8000 | 53955 | 53955 | 0.1308 | 0.1921 | 0.0948 | 0.1105 | 0.1006 | 98.4542 |
| 0.9000 | 32796 | 32796 | 0.1447 | 0.2135 | 0.1064 | 0.1157 | 0.1098 | 59.7643 |

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*

DRAFT (Claude, 2026-11-01, for review): Within loans Lending Club already approved, a 70% cutoff on each score approves the same 293,728 loans in number. The 66,965 loans the card would approve instead of FICO's go bad at 11.99%; the 66,965 it would decline instead go bad at 17.48%. The approved book's bad rate falls from 10.50% to 9.25%, 125 fewer bads per 10,000 approvals. The gain is 157 per 10,000 at a 50% approval rate and 60 at 90%. This is a reordering of booked loans; it says nothing about applicants Lending Club declined.

## 4. Calibration

| band | n | observed | predicted | lower_95 | upper_95 | within_interval | binomial_p |
|---|---|---|---|---|---|---|---|
| [440.0, 460.0) | 122 | 0.3033 | 0.2861 | 0.2059 | 0.3664 | True | 0.7406 |
| [460.0, 480.0) | 5143 | 0.2578 | 0.2296 | 0.2181 | 0.2411 | False | 0.0000 |
| [480.0, 500.0) | 32626 | 0.2142 | 0.1793 | 0.1751 | 0.1834 | False | 0.0000 |
| [500.0, 520.0) | 78150 | 0.1733 | 0.1364 | 0.1340 | 0.1388 | False | 0.0000 |
| [520.0, 540.0) | 102411 | 0.1322 | 0.1018 | 0.0999 | 0.1036 | False | 0.0000 |
| [540.0, 560.0) | 95107 | 0.0955 | 0.0748 | 0.0731 | 0.0765 | False | 0.0000 |
| [560.0, 580.0) | 63810 | 0.0668 | 0.0546 | 0.0529 | 0.0564 | False | 0.0000 |
| [580.0, 600.0) | 30698 | 0.0465 | 0.0396 | 0.0374 | 0.0418 | False | 0.0000 |
| [600.0, 620.0) | 9722 | 0.0359 | 0.0288 | 0.0254 | 0.0321 | False | 0.0001 |
| [620.0, 640.0) | 1765 | 0.0227 | 0.0208 | 0.0142 | 0.0275 | True | 0.6368 |
| [640.0, 660.0) | 57 | 0.0000 | 0.0154 | 0.0000 | 0.0474 | True | 0.8247 |

Calibration slope 1.012 (95% CI 0.992 to 1.031), intercept 0.287; intercept-only shift 0.262; mean predicted 9.60% vs observed 12.06%. Hosmer-Lemeshow chi2 3061.8 (p = 0). Log-loss 0.3581 vs null 0.3681; Brier 0.1040 (reliability 0.00072, resolution 0.00273).

DRAFT (Claude, 2026-11-01, for review): As developed, the card ranks 2016 correctly (slope 1.012, 95% CI 0.992 to 1.031) but under-predicts the level: 9.60% predicted against 12.06% observed, a calibration-in-the-large shift of 0.262 log-odds, with 8 of 11 score bands outside their interval. Recalibrating only the intercept on 2015 (prior correction, +0.198 log-odds, from the 2013-2014 bad rate of 9.60% to 2015's 11.46%) brings 2016 to 11.41% predicted, a shift of 0.064, with 3 of 11 bands outside (500-560, still under-predicted by 0.6 to 1.2 points). Like for like, the GBM recalibrated on 2015 predicts 11.17% (shift 0.090, slope 0.946). Recalibrated, the two are equally well calibrated in level (Brier 0.1033 card, 0.1029 GBM); the card's slope is closer to 1.

## 4a. Calibration raw and recalibrated on 2015: card and GBM

| model | recalibration_on_2015 | mean_predicted | mean_observed | calib_slope | slope_ci_low | slope_ci_high | calib_intercept | calib_in_the_large | brier |
|---|---|---|---|---|---|---|---|---|---|
| Scorecard | none (raw) | 0.0960 | 0.1206 | 1.0117 | 0.9922 | 1.0312 | 0.2869 | 0.2617 | 0.1040 |
| Scorecard | intercept (prior correction) | 0.1141 | 0.1206 | 1.0117 | 0.9922 | 1.0312 | 0.0866 | 0.0637 | 0.1033 |
| Tuned GBM | none (raw) | 0.0947 | 0.1206 | 0.9207 | 0.9041 | 0.9373 | 0.1095 | 0.2800 | 0.1035 |
| Tuned GBM | intercept and slope (logistic) | 0.1117 | 0.1206 | 0.9463 | 0.9292 | 0.9633 | -0.0157 | 0.0898 | 0.1029 |

Scorecard with its intercept recalibrated on 2015, by score band:

| band | n | observed | predicted | lower_95 | upper_95 | within_interval | binomial_p |
|---|---|---|---|---|---|---|---|
| [440.0, 460.0) | 122 | 0.3033 | 0.3282 | 0.2449 | 0.4115 | True | 0.6303 |
| [460.0, 480.0) | 5143 | 0.2578 | 0.2664 | 0.2543 | 0.2785 | True | 0.1683 |
| [480.0, 500.0) | 32626 | 0.2142 | 0.2102 | 0.2058 | 0.2146 | True | 0.0768 |
| [500.0, 520.0) | 78150 | 0.1733 | 0.1614 | 0.1589 | 0.1640 | False | 0.0000 |
| [520.0, 540.0) | 102411 | 0.1322 | 0.1214 | 0.1194 | 0.1234 | False | 0.0000 |
| [540.0, 560.0) | 95107 | 0.0955 | 0.0897 | 0.0879 | 0.0915 | False | 0.0000 |
| [560.0, 580.0) | 63810 | 0.0668 | 0.0658 | 0.0639 | 0.0677 | True | 0.3218 |
| [580.0, 600.0) | 30698 | 0.0465 | 0.0478 | 0.0454 | 0.0502 | True | 0.2859 |
| [600.0, 620.0) | 9722 | 0.0359 | 0.0348 | 0.0312 | 0.0385 | True | 0.5857 |
| [620.0, 640.0) | 1765 | 0.0227 | 0.0253 | 0.0180 | 0.0326 | True | 0.5398 |
| [640.0, 660.0) | 57 | 0.0000 | 0.0187 | 0.0000 | 0.0539 | True | 0.6806 |

## 4b. Indeterminate sensitivity (2016)

| treatment | n | bads | bad_rate | gini_card | ks_card | gini_gbm | ks_gbm | gini_fico | ks_fico |
|---|---|---|---|---|---|---|---|---|---|
| excluded (rule in force) | 419611 | 50591 | 0.1206 | 0.2838 | 0.2078 | 0.3053 | 0.2210 | 0.1805 | 0.1354 |
| counted as good | 425618 | 50591 | 0.1189 | 0.2799 | 0.2050 | 0.3012 | 0.2180 | 0.1776 | 0.1332 |
| counted as bad | 425618 | 56598 | 0.1330 | 0.2797 | 0.2046 | 0.3005 | 0.2170 | 0.1803 | 0.1356 |

## 5. Segment performance

By `term_months`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 36 | 314300 | 0.1101 | 0.2847 | 0.2073 | 0.1834 | 0.1373 |
| 60 | 105311 | 0.1517 | 0.3029 | 0.2243 | 0.1856 | 0.1396 |

By `file_depth`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 11-20 | 138514 | 0.1208 | 0.2872 | 0.2094 | 0.1698 | 0.1268 |
| 21-35 | 173707 | 0.1188 | 0.3035 | 0.2217 | 0.1951 | 0.1453 |
| thick (>35) | 69304 | 0.1253 | 0.2927 | 0.2186 | 0.2023 | 0.1535 |
| thin (<=10) | 38086 | 0.1194 | 0.2101 | 0.1592 | 0.1128 | 0.0888 |

By `purpose`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| car | 4709 | 0.0873 | 0.2593 | 0.1948 | 0.1729 | 0.1468 |
| credit_card | 88833 | 0.1004 | 0.3061 | 0.2228 | 0.2182 | 0.1628 |
| debt_consolidation | 239189 | 0.1286 | 0.2815 | 0.2058 | 0.1788 | 0.1345 |
| home_improvement | 30361 | 0.1053 | 0.2797 | 0.2074 | 0.1787 | 0.1314 |
| house | 1938 | 0.1533 | 0.3076 | 0.2411 | 0.2083 | 0.1764 |
| major_purchase | 10197 | 0.1156 | 0.2628 | 0.2062 | 0.1394 | 0.1039 |
| medical | 5286 | 0.1377 | 0.1978 | 0.1586 | 0.1190 | 0.1034 |
| moving | 3153 | 0.1468 | 0.2378 | 0.1908 | 0.1031 | 0.0837 |
| other | 27810 | 0.1215 | 0.2512 | 0.1830 | 0.1242 | 0.0934 |
| renewable_energy | 297 | 0.1650 | 0.0941 | 0.1185 | 0.0512 | 0.0746 |
| small_business | 4642 | 0.1805 | 0.2564 | 0.2015 | 0.1461 | 0.1189 |
| vacation | 3195 | 0.1183 | 0.2151 | 0.1480 | 0.1358 | 0.1185 |
| wedding | 1 | 0.0000 |  |  |  |  |

By `vintage_quarter`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 2016Q1 | 129403 | 0.1208 | 0.3134 | 0.2314 | 0.2048 | 0.1548 |
| 2016Q2 | 95085 | 0.1220 | 0.2900 | 0.2122 | 0.1832 | 0.1353 |
| 2016Q3 | 97144 | 0.1261 | 0.2610 | 0.1898 | 0.1616 | 0.1228 |
| 2016Q4 | 97979 | 0.1133 | 0.2575 | 0.1897 | 0.1632 | 0.1214 |

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*

DRAFT (Claude, 2026-11-01, for review): The card beats FICO in every term, file-depth and purpose segment with enough loans: 36-month 0.2847 against 0.1834, 60-month 0.3029 against 0.1856, thin files 0.2101 against 0.1128. Both scores lose discrimination through 2016 (card 0.3134 in Q1 to 0.2575 in Q4). Purposes with fewer than 1,000 loans (renewable energy 297, wedding 1) are too small to read.

## 6. Stability

PSI of the pilot population's scores against development: **0.0009** (verdict stable; chi-square critical value at this n 0.0001).

| lower | upper | dev_share | cur_share | psi_part |
|---|---|---|---|---|
| -inf | 501.0000 | 0.0961 | 0.0968 | 0.0000 |
| 501.0000 | 513.0000 | 0.1004 | 0.1036 | 0.0001 |
| 513.0000 | 522.0000 | 0.0972 | 0.0995 | 0.0001 |
| 522.0000 | 530.0000 | 0.0982 | 0.0968 | 0.0000 |
| 530.0000 | 538.0000 | 0.1012 | 0.0997 | 0.0000 |
| 538.0000 | 546.0000 | 0.0999 | 0.0974 | 0.0001 |
| 546.0000 | 555.0000 | 0.1055 | 0.1018 | 0.0001 |
| 555.0000 | 565.0000 | 0.1003 | 0.0985 | 0.0000 |
| 565.0000 | 578.0000 | 0.0960 | 0.0941 | 0.0000 |
| 578.0000 | inf | 0.1052 | 0.1118 | 0.0004 |

DRAFT (Claude, 2026-11-01, for review): The PSI of 2016 scores against development is 0.0009, stable. It is statistically significant only because the sample is large (critical value 0.00009).

## 7. Fair lending

Adverse impact ratio at the approval cutoff:

| group | n | approval_rate | air | below_four_fifths | reference |
|---|---|---|---|---|---|
| other | 279230 | 0.7044 | 1.0000 | False | True |
| south | 140381 | 0.6915 | 0.9817 | False | False |

Standardized mean difference of scores:

| group | n | mean | std | smd_vs_reference | reference |
|---|---|---|---|---|---|
| other | 279230 | 539.6164 | 30.2000 | 0.0000 | True |
| south | 140381 | 538.3777 | 29.8804 | -0.0412 | False |

Performance and calibration by group:

| group | n | bad_rate | mean_score | gini | ks | calib_slope | calib_intercept |
|---|---|---|---|---|---|---|---|
| other | 279230 | 0.1191 | 539.6000 | 0.2853 | 0.2088 | 1.0120 | 0.2800 |
| south | 140381 | 0.1235 | 538.4000 | 0.2806 | 0.2058 | 1.0100 | 0.2980 |

DRAFT (Claude, 2026-11-01, for review): The proxy (Southern states against the rest) is a geographic placeholder, not a protected class: approval rates 69.15% and 70.44%, adverse impact ratio 0.9817, score SMD -0.041. This demonstrates the code path; fair-lending testing is out of scope for this data.

## Appendix: Tuned GBM, recalibrated on 2015

XGBoost tuned on 2015 (NB07b), probabilities mapped to the 2015 level with a logistic recalibration fitted on 2015, then placed on the card's score scale.

| score | gini | gini_se | ks |
|---|---|---|---|
| challenger | 0.3053 | 0.0025 | 0.2210 |
| incumbent | 0.1805 | 0.0026 | 0.1354 |

Gini lift over the incumbent: **0.1249** (DeLong p = 0; bootstrap 95% CI 0.1190 to 0.1306). Level: out-of-time (2016 vintage).

**Population.** The 70% cutoff is applied to loans Lending Club already approved (the booked 2016 vintage). The swap set therefore shows how each score would reorder approvals within the booked population; it does not show outcomes for applicants Lending Club declined, whose outcomes are not observed (NB06).

|  | challenger_approve | challenger_decline |
|---|---|---|
| incumbent_approve | 226495 | 67233 |
| incumbent_decline | 67233 | 58650 |

Swap-ins 67,233 (bad rate 11.51%); swap-outs 67,233 (bad rate 18.13%); approved-book bad rate 8.99% against 10.50% for the incumbent.

| band | n | observed | predicted | lower_95 | upper_95 | within_interval | binomial_p |
|---|---|---|---|---|---|---|---|
| [380.0, 400.0) | 3 | 0.3333 | 0.5542 | 0.0000 | 1.0000 | True | 0.8380 |
| [400.0, 420.0) | 76 | 0.3421 | 0.4578 | 0.3458 | 0.5698 | False | 0.0544 |
| [420.0, 440.0) | 802 | 0.2918 | 0.3767 | 0.3431 | 0.4102 | False | 0.0000 |
| [440.0, 460.0) | 4869 | 0.2810 | 0.3022 | 0.2893 | 0.3151 | False | 0.0013 |
| [460.0, 480.0) | 18926 | 0.2334 | 0.2366 | 0.2305 | 0.2427 | True | 0.3019 |
| [480.0, 500.0) | 50083 | 0.1953 | 0.1816 | 0.1782 | 0.1850 | False | 0.0000 |
| [500.0, 520.0) | 87285 | 0.1532 | 0.1372 | 0.1349 | 0.1395 | False | 0.0000 |
| [520.0, 540.0) | 99155 | 0.1136 | 0.1022 | 0.1003 | 0.1041 | False | 0.0000 |
| [540.0, 560.0) | 75561 | 0.0806 | 0.0752 | 0.0733 | 0.0771 | False | 0.0000 |
| [560.0, 580.0) | 44042 | 0.0583 | 0.0547 | 0.0525 | 0.0568 | False | 0.0008 |
| [580.0, 600.0) | 22234 | 0.0436 | 0.0394 | 0.0368 | 0.0419 | False | 0.0014 |
| [600.0, 620.0) | 10122 | 0.0347 | 0.0283 | 0.0250 | 0.0315 | False | 0.0002 |
| [620.0, 640.0) | 4182 | 0.0253 | 0.0203 | 0.0160 | 0.0245 | False | 0.0265 |
| [640.0, 660.0) | 1630 | 0.0196 | 0.0144 | 0.0086 | 0.0201 | True | 0.1025 |
| [660.0, 680.0) | 546 | 0.0128 | 0.0103 | 0.0019 | 0.0188 | True | 0.6745 |
| [680.0, 700.0) | 90 | 0.0000 | 0.0076 | 0.0000 | 0.0255 | True | 1.0000 |
| [700.0, 720.0) | 5 | 0.2000 | 0.0056 | 0.0000 | 0.0713 | False | 0.0557 |

Calibration slope 0.946, intercept -0.016, intercept-only shift 0.090; mean predicted 11.17% against observed 12.06%.

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*

## Caveats and assumptions

- Target is a proxy: one Lending Club loan per consumer, 90+ DPD reconstructed from the last payment date, 24 months from issue; VantageScore's definition observes 90+ DPD on any tradeline over 24 months from a bureau archive.
- Population is booked loans only (approved by Lending Club); see NB06 for the reject-inference sensitivity.
- Fair-lending groups use a documented proxy (proxy_group); no protected-class data exists in the file.
- Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.
- The tuned GBM in the appendix was recalibrated on 2015 (a two-parameter logistic map) before scoring 2016; its ranking is unchanged by the recalibration.
- Indeterminate loans (first miss inside 24 months, 90 DPD in months 25-27) are excluded from every exhibit except the indeterminate-sensitivity section.
- Sections 1-4 show the card as developed (2013-2014 intercept). Section 4a adds the card with its intercept recalibrated on 2015 by prior correction; scores, ranking and the swap set are unchanged by it.
