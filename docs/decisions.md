# Decisions log

Dated log of design decisions.

## 2026-10-25 — Exclusive vintage split

`v_split` labels are exclusive: train 2012–2014, validate 2015, oot = all of 2016, monitor
2017Q1–2018Q4, pre-2012 vintages = NULL (unused). The spec's monitor range began at 2016Q3, which
overlapped the OOT year; `config.MONITOR_VINTAGES` is now `("2017Q1", "2018Q4")`.

