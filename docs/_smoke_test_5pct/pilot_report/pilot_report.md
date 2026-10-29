# Pilot report: bureau-attribute scorecard vs FICO

*Lender: Lending Club (out-of-time 2016 vintage) · Validation level: out-of-time (2016 vintage) · n = 21,500 · bad rate = 12.16% · incumbent = fico_mid · approval rate for swap set = 70%*

## 1. Discrimination

| score | gini | gini_se | ks |
|---|---|---|---|
| challenger | 0.2481 | 0.0112 | 0.1784 |
| incumbent | 0.1668 | 0.0114 | 0.1239 |

Gini lift challenger - incumbent: **0.0813** (DeLong z = 5.90, p = 3.66e-09; bootstrap 95% CI 0.0540 to 0.1122). Level: out-of-time (2016 vintage).



## 2. Gains tables

Challenger:

| band | n | bads | bad_rate | cum_share_population | cum_share_bads | cum_share_goods | ks_at_band | lift |
|---|---|---|---|---|---|---|---|---|
| 1 | 2150 | 411 | 0.1912 | 0.1000 | 0.1572 | 0.0921 | 0.0651 | 1.5730 |
| 2 | 2150 | 423 | 0.1967 | 0.2000 | 0.3191 | 0.1835 | 0.1356 | 1.6180 |
| 3 | 2150 | 324 | 0.1507 | 0.3000 | 0.4430 | 0.2802 | 0.1628 | 1.2390 |
| 4 | 2150 | 277 | 0.1288 | 0.4000 | 0.5490 | 0.3794 | 0.1696 | 1.0590 |
| 5 | 2150 | 280 | 0.1302 | 0.5000 | 0.6561 | 0.4784 | 0.1777 | 1.0710 |
| 6 | 2150 | 244 | 0.1135 | 0.6000 | 0.7494 | 0.5793 | 0.1701 | 0.9340 |
| 7 | 2150 | 222 | 0.1033 | 0.7000 | 0.8344 | 0.6814 | 0.1530 | 0.8500 |
| 8 | 2150 | 188 | 0.0874 | 0.8000 | 0.9063 | 0.7853 | 0.1210 | 0.7190 |
| 9 | 2150 | 148 | 0.0688 | 0.9000 | 0.9629 | 0.8913 | 0.0716 | 0.5660 |
| 10 | 2150 | 97 | 0.0451 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.3710 |

Incumbent:

| band | n | bads | bad_rate | cum_share_population | cum_share_bads | cum_share_goods | ks_at_band | lift |
|---|---|---|---|---|---|---|---|---|
| 1 | 2150 | 354 | 0.1647 | 0.1000 | 0.1354 | 0.0951 | 0.0403 | 1.3550 |
| 2 | 2150 | 333 | 0.1549 | 0.2000 | 0.2628 | 0.1913 | 0.0715 | 1.2740 |
| 3 | 2150 | 309 | 0.1437 | 0.3000 | 0.3810 | 0.2888 | 0.0922 | 1.1820 |
| 4 | 2150 | 305 | 0.1419 | 0.4000 | 0.4977 | 0.3865 | 0.1112 | 1.1670 |
| 5 | 2150 | 273 | 0.1270 | 0.5000 | 0.6021 | 0.4859 | 0.1162 | 1.0450 |
| 6 | 2150 | 284 | 0.1321 | 0.6000 | 0.7108 | 0.5847 | 0.1261 | 1.0870 |
| 7 | 2150 | 236 | 0.1098 | 0.7000 | 0.8011 | 0.6860 | 0.1151 | 0.9030 |
| 8 | 2150 | 209 | 0.0972 | 0.8000 | 0.8810 | 0.7888 | 0.0922 | 0.7990 |
| 9 | 2150 | 192 | 0.0893 | 0.9000 | 0.9545 | 0.8925 | 0.0620 | 0.7340 |
| 10 | 2150 | 119 | 0.0553 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.4550 |



## 3. Swap set at matched approval rate

|  | challenger_approve | challenger_decline |
|---|---|---|
| incumbent_approve | 11753 | 3842 |
| incumbent_decline | 3309 | 2596 |

Swap-ins: 3,309 loans, bad rate 12.66%. Swap-outs: 3,842 loans, bad rate 17.36%. Approved-book bad rate: incumbent 10.95%, challenger 9.69% (126 bads avoided per 10,000 approvals).

| approval_rate | n_swap_in | n_swap_out | bad_rate_swap_in | bad_rate_swap_out | bad_rate_both | bad_rate_approved_incumbent | bad_rate_approved_challenger | bads_avoided_per_10k_approved |
|---|---|---|---|---|---|---|---|---|
| 0.5000 | 4081 | 3995 | 0.1029 | 0.1374 | 0.0727 | 0.0967 | 0.0841 | 126.4557 |
| 0.6000 | 3576 | 4336 | 0.1219 | 0.1584 | 0.0812 | 0.1052 | 0.0922 | 129.4144 |
| 0.7000 | 3309 | 3842 | 0.1266 | 0.1736 | 0.0885 | 0.1095 | 0.0969 | 125.9187 |
| 0.8000 | 2677 | 3020 | 0.1449 | 0.1911 | 0.0958 | 0.1122 | 0.1034 | 87.4746 |
| 0.9000 | 1664 | 1544 | 0.1617 | 0.1949 | 0.1103 | 0.1171 | 0.1147 | 23.5065 |



## 4. Calibration

| band | n | observed | predicted | lower_95 | upper_95 | within_interval | binomial_p |
|---|---|---|---|---|---|---|---|
| [460.0, 480.0) | 177 | 0.2373 | 0.2275 | 0.1657 | 0.2892 | True | 0.8124 |
| [480.0, 500.0) | 1434 | 0.1862 | 0.1775 | 0.1577 | 0.1973 | True | 0.4068 |
| [500.0, 520.0) | 4608 | 0.1777 | 0.1361 | 0.1262 | 0.1460 | False | 0.0000 |
| [520.0, 540.0) | 5895 | 0.1262 | 0.1021 | 0.0944 | 0.1098 | False | 0.0000 |
| [540.0, 560.0) | 5035 | 0.0975 | 0.0750 | 0.0677 | 0.0822 | False | 0.0000 |
| [560.0, 580.0) | 3188 | 0.0643 | 0.0548 | 0.0469 | 0.0626 | False | 0.0223 |
| [580.0, 600.0) | 1078 | 0.0408 | 0.0404 | 0.0287 | 0.0522 | True | 0.9903 |
| [600.0, 620.0) | 85 | 0.0235 | 0.0302 | 0.0000 | 0.0666 | True | 1.0000 |

Calibration slope 0.995 (95% CI 0.899 to 1.092), intercept 0.229; intercept-only shift 0.239; mean predicted 9.87% vs observed 12.16%. Hosmer-Lemeshow chi2 157.9 (p = 0). Log-loss 0.3628 vs null 0.3701; Brier 0.1053 (reliability 0.00071, resolution 0.00216).



## 5. Segment performance

By `term_months`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 36 | 16024 | 0.1096 | 0.2628 | 0.1911 | 0.1820 | 0.1377 |
| 60 | 5476 | 0.1565 | 0.2366 | 0.1726 | 0.1454 | 0.1046 |

By `file_depth`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 11-20 | 7237 | 0.1264 | 0.2643 | 0.1971 | 0.1546 | 0.1154 |
| 21-35 | 8842 | 0.1135 | 0.2577 | 0.1874 | 0.1693 | 0.1276 |
| thick (>35) | 3475 | 0.1304 | 0.2454 | 0.2018 | 0.2418 | 0.1828 |
| thin (<=10) | 1946 | 0.1244 | 0.2131 | 0.1683 | 0.0536 | 0.0851 |

By `purpose`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| car | 243 | 0.0700 | 0.1973 | 0.2236 | 0.2041 | 0.2192 |
| credit_card | 4559 | 0.1013 | 0.3123 | 0.2317 | 0.2105 | 0.1600 |
| debt_consolidation | 12340 | 0.1299 | 0.2254 | 0.1674 | 0.1578 | 0.1188 |
| home_improvement | 1511 | 0.1145 | 0.3276 | 0.2319 | 0.1754 | 0.1501 |
| house | 97 | 0.1031 | 0.2126 | 0.3287 | 0.2793 | 0.3678 |
| major_purchase | 513 | 0.1326 | 0.1227 | 0.1119 | 0.0318 | 0.0991 |
| medical | 265 | 0.1509 | 0.2298 | 0.2344 | 0.2236 | 0.2483 |
| moving | 179 | 0.1620 | 0.3630 | 0.3841 | 0.0984 | 0.1398 |
| other | 1405 | 0.1146 | 0.2002 | 0.1659 | 0.1808 | 0.1381 |
| renewable_energy | 15 | 0.1333 |  |  |  |  |
| small_business | 229 | 0.1572 | 0.2025 | 0.1920 | 0.1012 | 0.1117 |
| vacation | 144 | 0.0903 | 0.2284 | 0.2631 | 0.0006 | 0.1574 |

By `vintage_quarter`:

| group | n | bad_rate | gini_challenger | ks_challenger | gini_incumbent | ks_incumbent |
|---|---|---|---|---|---|---|
| 2016Q1 | 6656 | 0.1185 | 0.2571 | 0.1903 | 0.1948 | 0.1532 |
| 2016Q2 | 4767 | 0.1267 | 0.2746 | 0.2069 | 0.1703 | 0.1326 |
| 2016Q3 | 4843 | 0.1303 | 0.2427 | 0.1940 | 0.1355 | 0.0984 |
| 2016Q4 | 5234 | 0.1127 | 0.2146 | 0.1571 | 0.1586 | 0.1198 |



## 6. Stability

PSI of the pilot population's scores against development: **0.0179** (verdict stable; chi-square critical value at this n 0.0016).

| lower | upper | dev_share | cur_share | psi_part |
|---|---|---|---|---|
| -inf | 505.0000 | 0.0935 | 0.1153 | 0.0046 |
| 505.0000 | 517.0000 | 0.1060 | 0.1260 | 0.0035 |
| 517.0000 | 523.0000 | 0.0731 | 0.0868 | 0.0023 |
| 523.0000 | 531.0000 | 0.1237 | 0.1183 | 0.0002 |
| 531.0000 | 539.0000 | 0.1035 | 0.1031 | 0.0000 |
| 539.0000 | 546.0000 | 0.0941 | 0.0918 | 0.0001 |
| 546.0000 | 554.0000 | 0.1004 | 0.0950 | 0.0003 |
| 554.0000 | 563.0000 | 0.1024 | 0.0904 | 0.0015 |
| 563.0000 | 573.0000 | 0.0958 | 0.0767 | 0.0042 |
| 573.0000 | inf | 0.1075 | 0.0966 | 0.0012 |



## 7. Fair lending

Adverse impact ratio at the approval cutoff:

| group | n | approval_rate | air | below_four_fifths | reference |
|---|---|---|---|---|---|
| other | 14340 | 0.7046 | 1.0000 | False | True |
| south | 7160 | 0.6925 | 0.9828 | False | False |

Standardized mean difference of scores:

| group | n | mean | std | smd_vs_reference | reference |
|---|---|---|---|---|---|
| other | 14340 | 536.1281 | 26.1183 | 0.0000 | True |
| south | 7160 | 535.6547 | 26.1134 | -0.0181 | False |

Performance and calibration by group:

| group | n | bad_rate | mean_score | gini | ks | calib_slope | calib_intercept |
|---|---|---|---|---|---|---|---|
| other | 14340 | 0.1215 | 536.1000 | 0.2463 | 0.1740 | 0.9840 | 0.2070 |
| south | 7160 | 0.1216 | 535.7000 | 0.2518 | 0.1926 | 1.0190 | 0.2740 |



## Caveats and assumptions

- Target is a proxy: one Lending Club loan per consumer, 90+ DPD reconstructed from the last payment date, 24 months from issue; VantageScore's definition observes 90+ DPD on any tradeline over 24 months from a bureau archive.
- Population is booked loans only (approved by Lending Club); see NB06 for the reject-inference sensitivity.
- Fair-lending groups use a documented proxy (proxy_group); no protected-class data exists in the file.
