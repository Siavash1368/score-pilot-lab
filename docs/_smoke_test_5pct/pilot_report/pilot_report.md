# Pilot report: bureau-attribute scorecard vs FICO

*Lender: Lending Club (out-of-time 2016 vintage) · Validation level: out-of-time (2016 vintage) · n = 20,931 · bad rate = 12.15% · incumbent = fico_mid · approval rate for swap set = 70%*

## 1. Discrimination

| score | gini | gini_se | ks |
|---|---|---|---|
| challenger | 0.2535 | 0.0115 | 0.1888 |
| incumbent | 0.1832 | 0.0114 | 0.1384 |

Gini lift challenger - incumbent: **0.0703** (DeLong z = 5.10, p = 3.34e-07; bootstrap 95% CI 0.0415 to 0.0997). Level: out-of-time (2016 vintage).

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*



## 2. Gains tables

Challenger:

| band | n | bads | bad_rate | cum_share_population | cum_share_bads | cum_share_goods | ks_at_band | lift |
|---|---|---|---|---|---|---|---|---|
| 1 | 2094 | 449 | 0.2144 | 0.1000 | 0.1765 | 0.0895 | 0.0870 | 1.7640 |
| 2 | 2093 | 356 | 0.1701 | 0.2000 | 0.3164 | 0.1839 | 0.1325 | 1.4000 |
| 3 | 2093 | 337 | 0.1610 | 0.3000 | 0.4489 | 0.2794 | 0.1695 | 1.3250 |
| 4 | 2093 | 275 | 0.1314 | 0.4000 | 0.5570 | 0.3783 | 0.1787 | 1.0810 |
| 5 | 2093 | 270 | 0.1290 | 0.5000 | 0.6631 | 0.4775 | 0.1856 | 1.0610 |
| 6 | 2093 | 220 | 0.1051 | 0.6000 | 0.7496 | 0.5793 | 0.1703 | 0.8650 |
| 7 | 2093 | 196 | 0.0936 | 0.7000 | 0.8267 | 0.6825 | 0.1442 | 0.7700 |
| 8 | 2093 | 194 | 0.0927 | 0.8000 | 0.9029 | 0.7858 | 0.1171 | 0.7630 |
| 9 | 2093 | 140 | 0.0669 | 0.9000 | 0.9579 | 0.8920 | 0.0659 | 0.5500 |
| 10 | 2093 | 107 | 0.0511 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.4200 |

Incumbent:

| band | n | bads | bad_rate | cum_share_population | cum_share_bads | cum_share_goods | ks_at_band | lift |
|---|---|---|---|---|---|---|---|---|
| 1 | 2094 | 349 | 0.1667 | 0.1000 | 0.1372 | 0.0949 | 0.0423 | 1.3720 |
| 2 | 2093 | 328 | 0.1567 | 0.2000 | 0.2661 | 0.1909 | 0.0752 | 1.2890 |
| 3 | 2093 | 317 | 0.1515 | 0.3000 | 0.3907 | 0.2875 | 0.1032 | 1.2460 |
| 4 | 2093 | 286 | 0.1366 | 0.4000 | 0.5031 | 0.3858 | 0.1173 | 1.1240 |
| 5 | 2093 | 283 | 0.1352 | 0.5000 | 0.6144 | 0.4842 | 0.1302 | 1.1120 |
| 6 | 2093 | 278 | 0.1328 | 0.6000 | 0.7237 | 0.5829 | 0.1408 | 1.0930 |
| 7 | 2093 | 243 | 0.1161 | 0.7000 | 0.8192 | 0.6835 | 0.1357 | 0.9550 |
| 8 | 2093 | 187 | 0.0893 | 0.8000 | 0.8927 | 0.7872 | 0.1055 | 0.7350 |
| 9 | 2093 | 157 | 0.0750 | 0.9000 | 0.9544 | 0.8925 | 0.0619 | 0.6170 |
| 10 | 2093 | 116 | 0.0554 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.4560 |

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*



## 3. Swap set at matched approval rate

|  | challenger_approve | challenger_decline |
|---|---|---|
| incumbent_approve | 11709 | 3480 |
| incumbent_decline | 3328 | 2414 |

Swap-ins: 3,328 loans, bad rate 12.95%. Swap-outs: 3,480 loans, bad rate 17.27%. Approved-book bad rate: incumbent 10.74%, challenger 9.72% (102 bads avoided per 10,000 approvals).

| approval_rate | n_swap_in | n_swap_out | bad_rate_swap_in | bad_rate_swap_out | bad_rate_both | bad_rate_approved_incumbent | bad_rate_approved_challenger | bads_avoided_per_10k_approved |
|---|---|---|---|---|---|---|---|---|
| 0.5000 | 3926 | 3998 | 0.1111 | 0.1433 | 0.0657 | 0.0947 | 0.0824 | 122.3538 |
| 0.6000 | 3318 | 4295 | 0.1187 | 0.1567 | 0.0794 | 0.1038 | 0.0897 | 140.5425 |
| 0.7000 | 3328 | 3480 | 0.1295 | 0.1727 | 0.0880 | 0.1074 | 0.0972 | 102.2000 |
| 0.8000 | 2611 | 2826 | 0.1429 | 0.1851 | 0.0969 | 0.1115 | 0.1040 | 75.0101 |
| 0.9000 | 1622 | 1624 | 0.1523 | 0.2087 | 0.1082 | 0.1168 | 0.1120 | 48.4409 |

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*



## 4. Calibration

| band | n | observed | predicted | lower_95 | upper_95 | within_interval | binomial_p |
|---|---|---|---|---|---|---|---|
| [460.0, 480.0) | 156 | 0.2308 | 0.2267 | 0.1610 | 0.2924 | True | 0.9647 |
| [480.0, 500.0) | 1274 | 0.2088 | 0.1782 | 0.1572 | 0.1992 | False | 0.0056 |
| [500.0, 520.0) | 4259 | 0.1766 | 0.1365 | 0.1262 | 0.1468 | False | 0.0000 |
| [520.0, 540.0) | 5779 | 0.1293 | 0.1019 | 0.0941 | 0.1097 | False | 0.0000 |
| [540.0, 560.0) | 5077 | 0.0945 | 0.0749 | 0.0676 | 0.0821 | False | 0.0000 |
| [560.0, 580.0) | 3234 | 0.0665 | 0.0548 | 0.0470 | 0.0627 | False | 0.0052 |
| [580.0, 600.0) | 1051 | 0.0457 | 0.0403 | 0.0284 | 0.0521 | True | 0.4106 |
| [600.0, 620.0) | 101 | 0.0000 | 0.0304 | 0.0000 | 0.0639 | True | 0.0883 |

Calibration slope 1.022 (95% CI 0.925 to 1.120), intercept 0.303; intercept-only shift 0.255; mean predicted 9.73% vs observed 12.15%. Hosmer-Lemeshow chi2 148.9 (p = 0). Log-loss 0.3626 vs null 0.3700; Brier 0.1052 (reliability 0.00069, resolution 0.00221).



## 4b. Indeterminate sensitivity (2016)

| treatment | n | bads | bad_rate | gini_card | ks_card | gini_gbm | ks_gbm | gini_fico | ks_fico |
|---|---|---|---|---|---|---|---|---|---|
| excluded (rule in force) | 20931 | 2544 | 0.1215 | 0.2535 | 0.1888 | 0.2907 | 0.2136 | 0.1832 | 0.1384 |
| counted as good | 21245 | 2544 | 0.1197 | 0.2492 | 0.1860 | 0.2856 | 0.2095 | 0.1805 | 0.1361 |
| counted as bad | 21245 | 2858 | 0.1345 | 0.2536 | 0.1865 | 0.2924 | 0.2166 | 0.1808 | 0.1389 |

## 5. Segment performance

By `term_months`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 36 | 15667 | 0.1098 | 0.2668 | 0.1996 | 0.1894 | 0.1461 |
| 60 | 5264 | 0.1563 | 0.2424 | 0.1805 | 0.1810 | 0.1357 |

By `file_depth`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 11-20 | 6863 | 0.1224 | 0.2544 | 0.1919 | 0.1645 | 0.1348 |
| 21-35 | 8771 | 0.1177 | 0.2947 | 0.2157 | 0.2096 | 0.1613 |
| thick (>35) | 3376 | 0.1345 | 0.2359 | 0.1927 | 0.2039 | 0.1444 |
| thin (<=10) | 1921 | 0.1135 | 0.2014 | 0.1702 | 0.0878 | 0.0806 |

By `purpose`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| car | 232 | 0.1034 | 0.1020 | 0.1987 | 0.2829 | 0.2388 |
| credit_card | 4470 | 0.0984 | 0.2927 | 0.2288 | 0.2264 | 0.1831 |
| debt_consolidation | 11825 | 0.1318 | 0.2389 | 0.1794 | 0.1705 | 0.1284 |
| home_improvement | 1519 | 0.1047 | 0.2295 | 0.1613 | 0.1425 | 0.1153 |
| house | 112 | 0.1429 | 0.1283 | 0.1771 | 0.0508 | 0.2292 |
| major_purchase | 516 | 0.1143 | 0.3900 | 0.3759 | 0.2616 | 0.2611 |
| medical | 230 | 0.0913 | 0.1545 | 0.2089 | 0.2221 | 0.2543 |
| moving | 179 | 0.1564 | 0.2455 | 0.2997 | 0.0710 | 0.1540 |
| other | 1442 | 0.1151 | 0.2339 | 0.1909 | 0.1861 | 0.1381 |
| renewable_energy | 14 | 0.1429 |  |  |  |  |
| small_business | 222 | 0.2162 | 0.2700 | 0.2838 | 0.0728 | 0.1509 |
| vacation | 170 | 0.1353 | 0.3715 | 0.3295 | 0.1689 | 0.2233 |

By `vintage_quarter`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 2016Q1 | 6412 | 0.1299 | 0.2939 | 0.2188 | 0.1915 | 0.1492 |
| 2016Q2 | 4741 | 0.1232 | 0.2578 | 0.2045 | 0.1542 | 0.1253 |
| 2016Q3 | 4819 | 0.1224 | 0.2011 | 0.1514 | 0.1718 | 0.1484 |
| 2016Q4 | 4959 | 0.1083 | 0.2453 | 0.1899 | 0.2176 | 0.1739 |

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*



## 6. Stability

PSI of the pilot population's scores against development: **0.0104** (verdict stable; chi-square critical value at this n 0.0017).

| lower | upper | dev_share | cur_share | psi_part |
|---|---|---|---|---|
| -inf | 504.0000 | 0.0980 | 0.1042 | 0.0004 |
| 504.0000 | 516.0000 | 0.0972 | 0.1142 | 0.0027 |
| 516.0000 | 524.0000 | 0.0984 | 0.1059 | 0.0006 |
| 524.0000 | 532.0000 | 0.1031 | 0.1137 | 0.0010 |
| 532.0000 | 539.0000 | 0.0930 | 0.0957 | 0.0001 |
| 539.0000 | 547.0000 | 0.1064 | 0.1071 | 0.0000 |
| 547.0000 | 554.0000 | 0.1039 | 0.0926 | 0.0013 |
| 554.0000 | 563.0000 | 0.0966 | 0.0878 | 0.0008 |
| 563.0000 | 573.0000 | 0.0945 | 0.0795 | 0.0026 |
| 573.0000 | inf | 0.1089 | 0.0993 | 0.0009 |



## 7. Fair lending

Adverse impact ratio at the approval cutoff:

| group | n | approval_rate | air | below_four_fifths | reference |
|---|---|---|---|---|---|
| other | 13948 | 0.7155 | 0.9880 | False | False |
| south | 6983 | 0.7242 | 1.0000 | False | True |

Standardized mean difference of scores:

| group | n | mean | std | smd_vs_reference | reference |
|---|---|---|---|---|---|
| other | 13948 | 536.7838 | 26.0094 | -0.0143 | False |
| south | 6983 | 537.1554 | 25.9804 | 0.0000 | True |

Performance and calibration by group:

| group | n | bad_rate | mean_score | gini | ks | calib_slope | calib_intercept |
|---|---|---|---|---|---|---|---|
| other | 13948 | 0.1211 | 536.8000 | 0.2586 | 0.1923 | 1.0430 | 0.3420 |
| south | 6983 | 0.1224 | 537.2000 | 0.2433 | 0.1828 | 0.9810 | 0.2270 |



## Appendix: Tuned GBM, recalibrated on 2015

XGBoost tuned on 2015 (NB07b), probabilities mapped to the 2015 level with a logistic recalibration fitted on 2015, then placed on the card's score scale.

| score | gini | gini_se | ks |
|---|---|---|---|
| challenger | 0.2907 | 0.0113 | 0.2136 |
| incumbent | 0.1832 | 0.0114 | 0.1384 |

Gini lift over the incumbent: **0.1076** (DeLong p = 0; bootstrap 95% CI 0.0824 to 0.1357). Level: out-of-time (2016 vintage).

|  | challenger_approve | challenger_decline |
|---|---|---|
| incumbent_approve | 11658 | 3531 |
| incumbent_decline | 3035 | 2707 |

Swap-ins 3,035 (bad rate 11.70%); swap-outs 3,531 (bad rate 18.07%); approved-book bad rate 9.17% against 10.74% for the incumbent.

| band | n | observed | predicted | lower_95 | upper_95 | within_interval | binomial_p |
|---|---|---|---|---|---|---|---|
| [420.0, 440.0) | 15 | 0.2000 | 0.3712 | 0.1267 | 0.6157 | True | 0.2661 |
| [440.0, 460.0) | 245 | 0.2612 | 0.3023 | 0.2448 | 0.3599 | True | 0.1804 |
| [460.0, 480.0) | 1072 | 0.2276 | 0.2362 | 0.2108 | 0.2616 | True | 0.5343 |
| [480.0, 500.0) | 2718 | 0.1939 | 0.1818 | 0.1673 | 0.1963 | True | 0.1107 |
| [500.0, 520.0) | 4756 | 0.1478 | 0.1372 | 0.1274 | 0.1470 | False | 0.0365 |
| [520.0, 540.0) | 5144 | 0.1073 | 0.1024 | 0.0941 | 0.1107 | True | 0.2545 |
| [540.0, 560.0) | 3772 | 0.0803 | 0.0753 | 0.0669 | 0.0837 | True | 0.2525 |
| [560.0, 580.0) | 1978 | 0.0576 | 0.0551 | 0.0450 | 0.0651 | True | 0.6428 |
| [580.0, 600.0) | 841 | 0.0321 | 0.0393 | 0.0262 | 0.0525 | True | 0.3216 |
| [600.0, 620.0) | 331 | 0.0181 | 0.0283 | 0.0105 | 0.0462 | True | 0.3416 |
| [620.0, 640.0) | 59 | 0.0169 | 0.0207 | 0.0000 | 0.0570 | True | 1.0000 |

Calibration slope 0.980, intercept 0.009, intercept-only shift 0.048; mean predicted 11.67% against observed 12.15%.

*Note on the incumbent: Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.*

## Caveats and assumptions

- Target is a proxy: one Lending Club loan per consumer, 90+ DPD reconstructed from the last payment date, 24 months from issue; VantageScore's definition observes 90+ DPD on any tradeline over 24 months from a bureau archive.
- Population is booked loans only (approved by Lending Club); see NB06 for the reject-inference sensitivity.
- Fair-lending groups use a documented proxy (proxy_group); no protected-class data exists in the file.
- Lending Club used FICO in its approval, so FICO's range is truncated in this booked population (no loan below a FICO midpoint of 662 in these vintages). Its Gini here understates what FICO does on a full applicant population, and every comparison with FICO in this report measures lift over a truncated incumbent.
- The tuned GBM in the appendix was recalibrated on 2015 (a two-parameter logistic map) before scoring 2016; its ranking is unchanged by the recalibration.
- Indeterminate loans (first miss inside 24 months, 90 DPD in months 25-27) are excluded from every exhibit except the indeterminate-sensitivity section.
