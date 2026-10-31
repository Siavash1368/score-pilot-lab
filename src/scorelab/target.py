"""Target definition: 24-month performance window, indeterminates and censoring (NB02).

Lending Club publishes one snapshot per loan (status and last payment date at the data cut), not
a monthly payment history. The label therefore reconstructs *when* a loan reached 90+ DPD from
``last_pymnt_d`` and ``loan_status`` using the timing constants in :mod:`scorelab.config`:

    first missed due date = last payment month + 1   (month 1 if the loan never paid)
    60 DPD                = first miss + 2
    90 DPD                = first miss + 3

For a loan issued at month 0 with a 24-month window:

    bad            delinquent status and 90 DPD month <= 24
    indeterminate  delinquent status, first miss <= 24 and 90 DPD month > 24
    good           everything else that is observable, including loans that only became
                   delinquent after the window closed

A loan is *observable* when ``issue_d + window <= PERFORMANCE_SNAPSHOT``; labels are NULL
otherwise. The censored monitor vintages carry an early indicator instead: 60+ DPD within
``EARLY_WINDOW_MONTHS``.

Everything here is SQL executed by DuckDB, exposed in one view (``v_target``) so the notebooks,
the MCP server and the agent read the same label. Only ``build_db.py`` and
``scripts/refresh_views.py`` call :func:`create_views`.
"""

from __future__ import annotations

from math import log

import duckdb
import pandas as pd

from scorelab import config

MODEL_SPLITS = ("train", "validate", "oot")


# --- SQL builders -------------------------------------------------------------

def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def target_view_sql(window: int | None = None, snapshot: str | None = None) -> str:
    """SQL that labels every accepted loan. Reads outcome fields from ``accepted`` only.

    ``window`` and ``snapshot`` default to the config values; pass others for sensitivity runs.
    """
    w = window or config.PERFORMANCE_WINDOW_MONTHS
    snap = snapshot or config.PERFORMANCE_SNAPSHOT
    ew = config.EARLY_WINDOW_MONTHS
    fm = config.MONTHS_LAST_PAYMENT_TO_FIRST_MISS
    d60 = config.MONTHS_FIRST_MISS_TO_60DPD
    d90 = config.MONTHS_FIRST_MISS_TO_90DPD
    bad = _sql_list(config.BAD_STATUSES)
    dlq = _sql_list(config.DELINQUENT_STATUSES)
    policy = config.CREDIT_POLICY_PREFIX.replace("'", "''")
    return f"""
    WITH base AS (
        SELECT id,
               issue_d,
               year(issue_d)                                   AS vintage_year,
               year(issue_d) || 'Q' || quarter(issue_d)        AS vintage_quarter,
               TRY_CAST(regexp_extract(term, '[0-9]+') AS INTEGER) AS term_months,
               loan_status,
               datediff('month', issue_d, last_pymnt_d)        AS months_to_last_payment,
               loan_status IN ({bad}) OR loan_status IN ({dlq}) AS is_delinquent,
               CASE WHEN loan_status LIKE '{policy}%' THEN 'credit_policy' END AS exclusion,
               CAST(issue_d + INTERVAL {w} MONTH  <= DATE '{snap}' AS INTEGER) AS observable,
               CAST(issue_d + INTERVAL {ew} MONTH <= DATE '{snap}' AS INTEGER) AS early_observable
        FROM accepted
    ), timing AS (
        SELECT *,
               COALESCE(months_to_last_payment, 0) + {fm}          AS first_miss_month,
               COALESCE(months_to_last_payment, 0) + {fm} + {d60}  AS dpd60_month,
               COALESCE(months_to_last_payment, 0) + {fm} + {d90}  AS dpd90_month
        FROM base
    )
    SELECT id, issue_d, vintage_year, vintage_quarter, term_months, loan_status,
           months_to_last_payment, exclusion, observable, early_observable,
           CASE WHEN observable = 0 THEN NULL
                WHEN is_delinquent AND dpd90_month <= {w} THEN 1 ELSE 0 END AS bad,
           CASE WHEN observable = 0 THEN NULL
                WHEN is_delinquent AND first_miss_month <= {w} AND dpd90_month > {w} THEN 1
                ELSE 0 END AS indeterminate,
           CASE WHEN observable = 0 THEN NULL
                WHEN is_delinquent AND dpd90_month <= {w} THEN 'bad'
                WHEN is_delinquent AND first_miss_month <= {w} THEN 'indeterminate'
                ELSE 'good' END AS label,
           CASE WHEN early_observable = 0 THEN NULL
                WHEN is_delinquent AND dpd60_month <= {ew} THEN 1 ELSE 0 END AS early_bad
    FROM timing
    """


def split_view_sql() -> str:
    """SQL for ``v_split``: exclusive vintage split on top of ``v_target``."""
    train = ", ".join(str(y) for y in config.TRAIN_VINTAGES)
    mon_lo, mon_hi = config.MONITOR_VINTAGES
    return f"""
    SELECT *,
           CASE WHEN vintage_year IN ({train}) THEN 'train'
                WHEN vintage_year = {config.VALIDATE_VINTAGE} THEN 'validate'
                WHEN vintage_year = {config.OOT_VINTAGE} THEN 'oot'
                WHEN vintage_quarter BETWEEN '{mon_lo}' AND '{mon_hi}' THEN 'monitor'
           END AS split
    FROM v_target
    """


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    """(Re)create ``v_target`` and ``v_split``. Needs a write connection."""
    con.execute("DROP VIEW IF EXISTS v_split")
    con.execute("DROP VIEW IF EXISTS v_target")
    con.execute(f"CREATE VIEW v_target AS {target_view_sql()}")
    con.execute(f"CREATE VIEW v_split AS {split_view_sql()}")


# --- Summaries for NB02 -------------------------------------------------------

def label_summary(con: duckdb.DuckDBPyConnection, by: str = "vintage_year") -> pd.DataFrame:
    """Counts and rates of good / bad / indeterminate among observable loans, per ``by``."""
    if by not in {"vintage_year", "vintage_quarter", "split", "term_months"}:
        raise ValueError(by)
    return con.sql(f"""
        SELECT {by},
               COUNT(*)                                            AS n_all,
               SUM(observable)                                     AS n_observable,
               SUM(CASE WHEN label = 'bad' THEN 1 ELSE 0 END)      AS n_bad,
               SUM(CASE WHEN label = 'indeterminate' THEN 1 ELSE 0 END) AS n_indeterminate,
               SUM(CASE WHEN label = 'good' THEN 1 ELSE 0 END)     AS n_good,
               ROUND(AVG(bad), 4)                                  AS bad_rate_incl_indet,
               ROUND(SUM(CASE WHEN label = 'bad' THEN 1 ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN label IN ('bad', 'good') THEN 1 ELSE 0 END), 0), 4)
                                                                   AS bad_rate_excl_indet,
               ROUND(AVG(early_bad), 4)                            AS early_bad_rate_12m
        FROM v_split
        WHERE exclusion IS NULL
        GROUP BY 1 ORDER BY 1
    """).df()


def vintage_curves(con: duckdb.DuckDBPyConnection, max_month: int = 36,
                   by_term: bool = False) -> pd.DataFrame:
    """Cumulative share of loans that reached 90+ DPD by month h since issue, per vintage year.

    At each horizon only loans observable at that horizon are counted, so the curve for a young
    vintage stops where the snapshot stops it. Columns: vintage_year[, term_months], horizon,
    n_observable, cum_bad_rate.
    """
    grp = "vintage_year, term_months" if by_term else "vintage_year"
    fm, d90 = config.MONTHS_LAST_PAYMENT_TO_FIRST_MISS, config.MONTHS_FIRST_MISS_TO_90DPD
    bad, dlq = _sql_list(config.BAD_STATUSES), _sql_list(config.DELINQUENT_STATUSES)
    return con.sql(f"""
        WITH loans AS (
            SELECT year(issue_d) AS vintage_year,
                   TRY_CAST(regexp_extract(term, '[0-9]+') AS INTEGER) AS term_months,
                   issue_d,
                   (loan_status IN ({bad}) OR loan_status IN ({dlq})) AS is_delinquent,
                   COALESCE(datediff('month', issue_d, last_pymnt_d), 0) + {fm} + {d90} AS dpd90_month
            FROM accepted
            WHERE loan_status NOT LIKE '{config.CREDIT_POLICY_PREFIX}%'
              AND year(issue_d) >= {min(config.TRAIN_VINTAGES)}
        ), horizons AS (
            SELECT range AS horizon FROM range(1, {max_month} + 1)
        )
        SELECT {grp}, horizon,
               COUNT(*) AS n_observable,
               ROUND(AVG(CASE WHEN is_delinquent AND dpd90_month <= horizon THEN 1 ELSE 0 END), 4)
                   AS cum_bad_rate
        FROM loans CROSS JOIN horizons
        WHERE issue_d + INTERVAL (horizon) MONTH <= DATE '{config.PERFORMANCE_SNAPSHOT}'
        GROUP BY {grp}, horizon
        ORDER BY {grp}, horizon
    """).df()


def window_sensitivity(con: duckdb.DuckDBPyConnection,
                       windows: tuple[int, ...] = (12, 18, 24, 30, 36)) -> pd.DataFrame:
    """Bad rate, indeterminate share and capture of eventual charge-offs under alternative windows.

    Restricted to the modelling vintages (train, validate, oot) so the windows are compared on the
    same loans wherever each window is observable.
    """
    rows = []
    years = (*config.TRAIN_VINTAGES, config.VALIDATE_VINTAGE, config.OOT_VINTAGE)
    yrs = ", ".join(str(y) for y in years)
    bad = _sql_list(config.BAD_STATUSES)
    for w in windows:
        df = con.sql(f"""
            WITH t AS ({target_view_sql(window=w)})
            SELECT {w} AS window_months,
                   SUM(observable) AS n_observable,
                   ROUND(SUM(CASE WHEN label = 'bad' THEN 1 ELSE 0 END)
                         / NULLIF(SUM(CASE WHEN label IN ('bad','good') THEN 1 ELSE 0 END), 0), 4)
                       AS bad_rate_excl_indet,
                   ROUND(SUM(CASE WHEN label = 'indeterminate' THEN 1 ELSE 0 END)
                         / NULLIF(SUM(observable), 0), 4) AS indeterminate_share,
                   ROUND(SUM(CASE WHEN label = 'bad' THEN 1 ELSE 0 END)
                         / NULLIF(SUM(CASE WHEN observable = 1 AND loan_status IN ({bad})
                                         THEN 1 ELSE 0 END), 0), 4)
                       AS share_of_eventual_chargeoffs_captured
            FROM t
            WHERE vintage_year IN ({yrs}) AND exclusion IS NULL
        """).df()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def indeterminate_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """What the indeterminate loans are: status at snapshot and term, modelling vintages only."""
    yrs = ", ".join(str(y) for y in (*config.TRAIN_VINTAGES, config.VALIDATE_VINTAGE,
                                     config.OOT_VINTAGE))
    return con.sql(f"""
        SELECT loan_status, term_months,
               COUNT(*) AS n,
               MIN(months_to_last_payment) AS min_months_to_last_payment,
               MAX(months_to_last_payment) AS max_months_to_last_payment
        FROM v_target
        WHERE label = 'indeterminate' AND vintage_year IN ({yrs})
        GROUP BY 1, 2 ORDER BY n DESC
    """).df()


# --- Modelling frame ----------------------------------------------------------

def sample_predicate(sample_frac: float | None, seed: int = 42, id_col: str = "id") -> str:
    """``AND ...`` clause that keeps a reproducible ``sample_frac`` share of rows, chosen by ``id``.

    A row is kept when md5("<id>:<seed>") mod 10^6 < sample_frac * 10^6. Unlike DuckDB's
    ``USING SAMPLE (bernoulli, seed)``, which is only repeatable on a single thread and depends on
    row order after a join, the result depends on the id alone: the same rows on every run and
    thread count, and a smaller fraction is always a subset of a larger one with the same seed."""
    if not sample_frac:
        return ""
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")
    return (f"AND md5_number(CAST({id_col} AS VARCHAR) || ':{int(seed)}') % 1000000 "
            f"< {int(round(sample_frac * 1_000_000))}")


def modeling_frame(con: duckdb.DuckDBPyConnection,
                   splits: tuple[str, ...] = MODEL_SPLITS,
                   include_indeterminate: bool = False,
                   columns: list[str] | None = None,
                   sample_frac: float | None = None,
                   seed: int = 42) -> pd.DataFrame:
    """Observable, non-excluded loans in ``splits`` with their as-of-application inputs.

    Adds ``fico_mid`` (the incumbent score) and ``split``, ``bad``, ``label``. Indeterminates are
    dropped unless ``include_indeterminate`` is True (then ``bad`` is NULL for them).
    ``sample_frac`` keeps a reproducible share of loans chosen by id (:func:`sample_predicate`), for
    exploration on a small machine: the full frame is ~1.3M rows x ~80 columns. Rows come back
    ordered by id, so models that subsample or shuffle rows are reproducible run to run.
    """
    sp = _sql_list(splits)
    cols = ", ".join(f"a.{c}" for c in columns) if columns else "a.*"
    label_filter = "" if include_indeterminate else "AND s.label IN ('bad', 'good')"
    sample = sample_predicate(sample_frac, seed, id_col="s.id")
    return con.sql(f"""
        SELECT s.split, s.vintage_year, s.vintage_quarter, s.term_months, s.label,
               CASE WHEN s.label = 'bad' THEN 1 WHEN s.label = 'good' THEN 0 END AS bad,
               {config.INCUMBENT_SCORE_SQL} AS {config.INCUMBENT_SCORE_NAME},
               {cols}
        FROM v_split s
        JOIN v_accepted_asof a USING (id)
        WHERE s.split IN ({sp}) AND s.observable = 1 AND s.exclusion IS NULL {label_filter}
          {sample}
        ORDER BY s.id   -- fixed row order: subsampling, bagging and shuffles depend on it
    """).df()


# --- Prior correction -----------------------------------------------------------

def prior_correct_intercept(alpha: float, sample_bad_rate: float, population_bad_rate: float) -> float:
    """Intercept on the population after fitting on a sample with a different bad rate.

    Oversampling bads shifts only the intercept of the log-odds (King & Zeng, 2001):
    alpha_pop = alpha_sample - ln[ (p_s (1 - p_pop)) / (p_pop (1 - p_s)) ].
    """
    if not (0 < sample_bad_rate < 1 and 0 < population_bad_rate < 1):
        raise ValueError("bad rates must be strictly between 0 and 1")
    return alpha - log((sample_bad_rate * (1 - population_bad_rate))
                       / (population_bad_rate * (1 - sample_bad_rate)))
