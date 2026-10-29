"""Coarse classing and weight of evidence (NB04).

Self-contained monotone binning so every step is inspectable (no optbinning dependency at run
time; swap it in later if you prefer). Numeric attributes are pre-binned on quantiles, then
adjacent bins are merged until (1) the bad rate is monotone in the chosen direction, (2) every
bin holds at least ``min_bin_share`` of the population and ``min_bads`` bads, and (3) there are
at most ``max_bins`` bins, merging the adjacent pair that loses the least information value at
each step. Missing values form their own bin when they hold enough bads, otherwise they join the
bin with the closest bad rate (recorded in the spec).

WOE_j = ln(share of goods in j / share of bads in j), smoothed by 0.5 so no bin is infinite;
IV = sum_j (good share - bad share) * WOE_j, the Jeffreys divergence between the good and bad
distributions of the binned attribute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

SMOOTH = 0.5
MISSING_LABEL = "Missing"
OTHER_LABEL = "Other (rare levels)"


@dataclass
class BinSpec:
    name: str
    kind: str                                  # "numeric" | "categorical"
    edges: list[float] = field(default_factory=list)       # numeric: upper bounds, last = inf, bin i = (edges[i-1], edges[i]]
    groups: dict[str, int] = field(default_factory=dict)   # categorical: level -> bin index
    other_bin: int | None = None               # categorical: bin for unseen levels
    labels: list[str] = field(default_factory=list)
    n: list[int] = field(default_factory=list)
    n_bad: list[int] = field(default_factory=list)
    woe: list[float] = field(default_factory=list)
    missing_bin: int | None = None             # index of the bin that receives missing values
    missing_own_bin: bool = False
    direction: int = 0                         # +1 bad rate increases with x, -1 decreases, 0 none
    iv: float = 0.0
    monotone: bool = True

    # ---- derived -------------------------------------------------------------
    def table(self) -> pd.DataFrame:
        n = np.array(self.n, float); nb = np.array(self.n_bad, float); ng = n - nb
        tot_g, tot_b = ng.sum(), nb.sum()
        dist_g = (ng + SMOOTH) / (tot_g + SMOOTH * len(n)); dist_b = (nb + SMOOTH) / (tot_b + SMOOTH * len(n))
        iv_part = (dist_g - dist_b) * np.array(self.woe)
        return pd.DataFrame({"bin": range(len(n)), "label": self.labels, "n": n.astype(int),
                             "share": np.round(n / n.sum(), 4), "n_bad": nb.astype(int),
                             "bad_rate": np.round(nb / np.where(n == 0, 1, n), 4),
                             "woe": np.round(self.woe, 4), "iv_part": np.round(iv_part, 4)})

    def dev_shares(self) -> np.ndarray:
        n = np.array(self.n, float)
        return n / n.sum()


# --- WOE arithmetic -----------------------------------------------------------------

def _woe_iv(n: np.ndarray, n_bad: np.ndarray) -> tuple[np.ndarray, float]:
    n = np.asarray(n, float); n_bad = np.asarray(n_bad, float); n_good = n - n_bad
    k = len(n)
    dist_g = (n_good + SMOOTH) / (n_good.sum() + SMOOTH * k)
    dist_b = (n_bad + SMOOTH) / (n_bad.sum() + SMOOTH * k)
    woe = np.log(dist_g / dist_b)
    return woe, float(((dist_g - dist_b) * woe).sum())


def _merge(n, nb, i):
    """Merge bin i with i+1 (in-place on python lists)."""
    n[i] += n[i + 1]; nb[i] += nb[i + 1]
    del n[i + 1]; del nb[i + 1]


def _rates(n, nb):
    return [b / x if x else 0.0 for x, b in zip(n, nb)]


def _enforce_monotone(n, nb, edges, direction):
    """Pool adjacent violators until bad rate is monotone (direction +1 increasing, -1 decreasing)."""
    changed = True
    while changed and len(n) > 1:
        changed = False
        r = _rates(n, nb)
        for i in range(len(n) - 1):
            if (direction > 0 and r[i] > r[i + 1]) or (direction < 0 and r[i] < r[i + 1]):
                _merge(n, nb, i); del edges[i]; changed = True; break


def _enforce_minimums(n, nb, edges, min_n, min_bads):
    """Merge bins below the population or bad-count minimum into the neighbour with the closer rate."""
    changed = True
    while changed and len(n) > 1:
        changed = False
        r = _rates(n, nb)
        for i in range(len(n)):
            if n[i] < min_n or nb[i] < min_bads:
                if i == 0: j = 0
                elif i == len(n) - 1: j = i - 1
                else: j = i - 1 if abs(r[i] - r[i - 1]) <= abs(r[i] - r[i + 1]) else i
                _merge(n, nb, j); del edges[j]; changed = True; break


def _reduce_to_max(n, nb, edges, max_bins):
    """Merge the adjacent pair whose merge loses the least IV until len <= max_bins."""
    while len(n) > max_bins:
        best, best_iv = None, -1.0
        for i in range(len(n) - 1):
            n2, nb2 = list(n), list(nb); _merge(n2, nb2, i)
            _, iv2 = _woe_iv(np.array(n2), np.array(nb2))
            if iv2 > best_iv: best, best_iv = i, iv2
        _merge(n, nb, best); del edges[best]


def _is_monotone(rates, direction):
    if direction == 0 or len(rates) < 2: return True
    d = np.diff(rates)
    return bool(np.all(d >= 0) if direction > 0 else np.all(d <= 0))


# --- Numeric ---------------------------------------------------------------------------

def _numeric_counts(x: np.ndarray, y: np.ndarray, edges: list[float]):
    idx = np.searchsorted(np.array(edges[:-1]), x, side="left")   # bin i: (edges[i-1], edges[i]]
    n = np.bincount(idx, minlength=len(edges)).tolist()
    nb = np.bincount(idx, weights=y, minlength=len(edges)).astype(int).tolist()
    return n, nb


def fit_numeric(x: pd.Series, y: pd.Series, name: str | None = None, max_bins: int = 8, prebins: int = 20,
                min_bin_share: float = 0.05, min_bads: int = 50, direction: int | str = "auto") -> BinSpec:
    """Monotone coarse classes for a numeric attribute. ``direction``: +1, -1, 0 (none) or 'auto'
    (try both and keep the higher IV; ties go to the sign of the raw correlation). ``max_bins`` counts
    non-missing bins; a missing bin, when it earns one, is added on top."""
    name = name or x.name
    xv = x.to_numpy(dtype=float); yv = y.to_numpy(dtype=float)
    miss = np.isnan(xv)
    xs, ys = xv[~miss], yv[~miss]
    total = len(xv)
    min_n = max(int(np.ceil(min_bin_share * total)), 1)

    def run(d):
        qs = np.unique(np.quantile(xs, np.linspace(0, 1, prebins + 1)[1:-1])) if len(xs) else np.array([])
        edges = [float(q) for q in qs] + [np.inf]
        n, nb = _numeric_counts(xs, ys, edges)
        # drop empty prebins
        keep = [i for i, c in enumerate(n) if c > 0]
        n = [n[i] for i in keep]; nb = [nb[i] for i in keep]; edges = [edges[i] for i in keep]; edges[-1] = np.inf
        if d != 0: _enforce_monotone(n, nb, edges, d)
        _enforce_minimums(n, nb, edges, min_n, min_bads)
        if d != 0: _enforce_monotone(n, nb, edges, d)
        _reduce_to_max(n, nb, edges, max_bins)
        woe, iv = _woe_iv(np.array(n), np.array(nb))
        return n, nb, edges, woe, iv

    if direction == "auto":
        cands = {}
        for d in (1, -1):
            if len(xs) > 0: cands[d] = run(d)
        if not cands:
            d_best = 0; n, nb, edges, woe, iv = [], [], [], np.array([]), 0.0
        else:
            d_best = max(cands, key=lambda d: cands[d][4])
            if len(cands) == 2 and abs(cands[1][4] - cands[-1][4]) < 1e-9:
                corr = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 and xs.std() > 0 else 0.0
                d_best = 1 if corr >= 0 else -1
            n, nb, edges, woe, iv = cands[d_best]
    else:
        d_best = int(direction)
        n, nb, edges, woe, iv = run(d_best) if len(xs) else ([], [], [], np.array([]), 0.0)

    labels = []
    lo = -np.inf
    for e in edges:
        labels.append(f"({lo:g}, {e:g}]" if np.isfinite(e) else f"> {lo:g}" if np.isfinite(lo) else "all")
        lo = e
    n, nb = list(n), list(nb); woe = list(woe)

    missing_bin, own = None, False
    n_miss = int(miss.sum()); nb_miss = int(yv[miss].sum())
    if n_miss > 0:
        if nb_miss >= min_bads and n_miss >= min_n // 2:
            n.append(n_miss); nb.append(nb_miss); labels.append(MISSING_LABEL)
            woe_all, iv = _woe_iv(np.array(n), np.array(nb)); woe = list(woe_all)
            missing_bin, own = len(n) - 1, True
        else:
            r_miss = nb_miss / n_miss
            rates = _rates(n, nb)
            j = int(np.argmin([abs(r - r_miss) for r in rates])) if rates else 0
            if rates:
                n[j] += n_miss; nb[j] += nb_miss
                labels[j] += " + Missing"
                woe_all, iv = _woe_iv(np.array(n), np.array(nb)); woe = list(woe_all)
            missing_bin, own = j, False
    rates = _rates(n[: len(edges)], nb[: len(edges)])
    return BinSpec(name=name, kind="numeric", edges=[float(e) for e in edges], labels=labels, n=[int(v) for v in n],
                   n_bad=[int(v) for v in nb], woe=[float(w) for w in woe], missing_bin=missing_bin,
                   missing_own_bin=own, direction=d_best, iv=float(iv), monotone=_is_monotone(rates, d_best))



def fit_numeric_with_edges(x: pd.Series, y: pd.Series, edges: list[float], name: str | None = None,
                           min_bads: int = 50) -> BinSpec:
    """A spec from fixed upper edges (your override, or thresholds from the GBM). Missing gets its own
    bin when it has at least ``min_bads`` bads, else joins the closest-rate bin."""
    name = name or x.name
    edges = sorted(float(e) for e in edges)
    if not edges or np.isfinite(edges[-1]): edges = edges + [np.inf]
    xv = x.to_numpy(dtype=float); yv = y.to_numpy(dtype=float); miss = np.isnan(xv)
    n, nb = _numeric_counts(xv[~miss], yv[~miss], edges)
    labels, lo = [], -np.inf
    for e in edges:
        labels.append(f"({lo:g}, {e:g}]" if np.isfinite(e) else f"> {lo:g}" if np.isfinite(lo) else "all"); lo = e
    n, nb = list(n), list(nb)
    missing_bin, own = None, False
    n_miss, nb_miss = int(miss.sum()), int(yv[miss].sum())
    if n_miss:
        if nb_miss >= min_bads:
            n.append(n_miss); nb.append(nb_miss); labels.append(MISSING_LABEL); missing_bin, own = len(n) - 1, True
        else:
            r = _rates(n, nb); j = int(np.argmin([abs(v - nb_miss / n_miss) for v in r]))
            n[j] += n_miss; nb[j] += nb_miss; labels[j] += " + Missing"; missing_bin = j
    woe, iv = _woe_iv(np.array(n), np.array(nb))
    rates = _rates(n[: len(edges)], nb[: len(edges)])
    d = 1 if len(rates) < 2 or rates[-1] >= rates[0] else -1
    return BinSpec(name=name, kind="numeric", edges=edges, labels=labels, n=[int(v) for v in n], n_bad=[int(v) for v in nb],
                   woe=[float(w) for w in woe], missing_bin=missing_bin, missing_own_bin=own, direction=d, iv=float(iv),
                   monotone=_is_monotone(rates, d))


# --- Categorical ---------------------------------------------------------------------

def fit_categorical(x: pd.Series, y: pd.Series, name: str | None = None, max_bins: int = 6,
                    min_bin_share: float = 0.02, min_bads: int = 50) -> BinSpec:
    """Group levels by bad rate: rare levels pooled as 'Other', then adjacent (by rate) merges to max_bins."""
    name = name or x.name
    s = x.astype("object").where(x.notna(), MISSING_LABEL)
    stats = pd.DataFrame({"n": s.groupby(s).size(), "nb": y.groupby(s).sum()}).fillna(0)
    total = len(s); min_n = max(int(np.ceil(min_bin_share * total)), 1)
    rare = stats.index[(stats.n < min_n) & (stats.index != MISSING_LABEL)]
    level_group = {lvl: (OTHER_LABEL if lvl in set(rare) else lvl) for lvl in stats.index}
    g = s.map(level_group)
    stats = pd.DataFrame({"n": g.groupby(g).size(), "nb": y.groupby(g).sum()}).fillna(0)
    stats["rate"] = stats.nb / stats.n
    stats = stats.sort_values("rate")
    groups = [[lvl] for lvl in stats.index]; n = stats.n.astype(int).tolist(); nb = stats.nb.astype(int).tolist()
    # merge below minimums with the rate-neighbour, then reduce to max_bins by least IV loss
    def merge(i):
        groups[i] += groups[i + 1]; del groups[i + 1]; _merge(n, nb, i)
    changed = True
    while changed and len(n) > 1:
        changed = False
        for i in range(len(n)):
            if n[i] < min_n or nb[i] < min_bads:
                j = 0 if i == 0 else i - 1 if i == len(n) - 1 else i - 1
                merge(j); changed = True; break
    while len(n) > max_bins:
        best, best_iv = 0, -1
        for i in range(len(n) - 1):
            n2, nb2 = list(n), list(nb); _merge(n2, nb2, i)
            iv2 = _woe_iv(np.array(n2), np.array(nb2))[1]
            if iv2 > best_iv: best, best_iv = i, iv2
        merge(best)
    woe, iv = _woe_iv(np.array(n), np.array(nb))
    level_to_bin = {}
    for b, grp in enumerate(groups):
        for lvl in grp:
            for raw, mapped in level_group.items():
                if mapped == lvl: level_to_bin[str(raw)] = b
    other_bin = next((b for b, grp in enumerate(groups) if OTHER_LABEL in grp), None)
    missing_bin = level_to_bin.get(MISSING_LABEL)
    labels = [", ".join(str(l) for l in grp)[:60] for grp in groups]
    return BinSpec(name=name, kind="categorical", groups=level_to_bin, other_bin=other_bin, labels=labels,
                   n=[int(v) for v in n], n_bad=[int(v) for v in nb], woe=[float(w) for w in woe],
                   missing_bin=missing_bin, missing_own_bin=missing_bin is not None, direction=0, iv=float(iv))


# --- Apply ---------------------------------------------------------------------------------

def bin_index(x: pd.Series, spec: BinSpec) -> np.ndarray:
    """Bin index for every row (missing and unseen levels handled per the spec)."""
    if spec.kind == "numeric":
        xv = x.to_numpy(dtype=float)
        idx = np.searchsorted(np.array(spec.edges[:-1]), xv, side="left")
        if spec.missing_bin is not None:
            idx = np.where(np.isnan(xv), spec.missing_bin, idx)
        else:
            idx = np.where(np.isnan(xv), 0, idx)
        return idx.astype(int)
    s = x.astype("object").where(x.notna(), MISSING_LABEL).astype(str)
    fallback = spec.other_bin if spec.other_bin is not None else (spec.missing_bin if spec.missing_bin is not None else 0)
    return s.map(spec.groups).fillna(fallback).astype(int).to_numpy()


def transform_woe(df: pd.DataFrame, specs: dict[str, BinSpec], suffix: str = "_woe") -> pd.DataFrame:
    """WOE-coded columns for every spec (column names ``<attribute><suffix>``)."""
    out = {}
    for name, spec in specs.items():
        woe = np.array(spec.woe)
        out[name + suffix] = woe[bin_index(df[name], spec)]
    return pd.DataFrame(out, index=df.index)


def fit_all(df: pd.DataFrame, names: list[str], target: str = "bad", directions: dict[str, int] | None = None,
            categorical: list[str] | None = None, **kw) -> dict[str, BinSpec]:
    """Fit a spec per attribute. ``directions`` (from the registry) fixes the monotone direction where
    given; others use 'auto'. ``categorical`` names use the categorical fitter."""
    y = df[target].astype(float)
    specs = {}
    for n in names:
        if categorical and n in categorical:
            specs[n] = fit_categorical(df[n], y, name=n, **{k: v for k, v in kw.items() if k in ("max_bins", "min_bin_share", "min_bads")})
        else:
            d = (directions or {}).get(n, "auto")
            specs[n] = fit_numeric(df[n], y, name=n, direction=d if d != 0 else "auto", **kw)
    return specs


def iv_summary(specs: dict[str, BinSpec]) -> pd.DataFrame:
    rows = [{"attribute": s.name, "iv": round(s.iv, 4), "bins": len(s.n), "monotone": s.monotone,
             "direction": s.direction, "missing_own_bin": s.missing_own_bin,
             "strength": "suspicious" if s.iv > 0.5 else "strong" if s.iv > 0.3 else "medium" if s.iv > 0.1
             else "weak" if s.iv > 0.02 else "useless"} for s in specs.values()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


# --- Stability ----------------------------------------------------------------------------

def csi(spec: BinSpec, x_current: pd.Series) -> tuple[float, pd.DataFrame]:
    """Characteristic stability index of ``x_current`` against the spec's development shares."""
    dev = spec.dev_shares()
    idx = bin_index(x_current, spec)
    cur = np.bincount(idx, minlength=len(dev)) / max(len(idx), 1)
    eps = 1e-4   # floor so an emptied bin reads as a large shift, not an infinite one
    parts = (cur - dev) * np.log((cur + eps) / (dev + eps))
    tbl = pd.DataFrame({"label": spec.labels, "dev_share": np.round(dev, 4), "cur_share": np.round(cur, 4),
                        "csi_part": np.round(parts, 5)})
    return float(parts.sum()), tbl


def csi_by_group(df: pd.DataFrame, specs: dict[str, BinSpec], group_col: str = "vintage_year") -> pd.DataFrame:
    """CSI per attribute and group (e.g. vintage year) against development shares."""
    rows = []
    for g, d in df.groupby(group_col):
        for n, s in specs.items():
            rows.append({group_col: g, "attribute": n, "csi": round(csi(s, d[n])[0], 4)})
    return pd.DataFrame(rows).pivot(index="attribute", columns=group_col, values="csi")


# --- Persistence ---------------------------------------------------------------------------

def save_specs(specs: dict[str, BinSpec], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {k: asdict(v) for k, v in specs.items()}
    for v in payload.values():
        v["edges"] = [None if not np.isfinite(e) else e for e in v["edges"]]
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_specs(path: str | Path) -> dict[str, BinSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for k, v in payload.items():
        v["edges"] = [np.inf if e is None else e for e in v["edges"]]
        out[k] = BinSpec(**v)
    return out
