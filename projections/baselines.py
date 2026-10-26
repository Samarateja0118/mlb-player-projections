"""Non-Bayesian baselines: league average and a Marcel-style weighted average."""
from __future__ import annotations

import pandas as pd

MARCEL_WEIGHTS = (5, 4, 3)  # most recent season first
MARCEL_REGRESSION = 1200  # opportunities of league-average performance added


def league_average(split) -> pd.Series:
    return pd.Series(split.league_rate, index=split.target.playerID.values)


def marcel(split, weights=MARCEL_WEIGHTS, regression=MARCEL_REGRESSION) -> pd.Series:
    """Weighted rate over the last len(weights) seasons, regressed toward league average.

    proj = (sum w_s * k_s + R * lg) / (sum w_s * n_s + R). No age adjustment.
    """
    t = split.target_year
    h = split.history.copy()
    h["w"] = h.yearID.map({t - 1 - i: w for i, w in enumerate(weights)}).fillna(0)
    agg = h.assign(wk=h.w * h.k, wn=h.w * h.n).groupby("playerID")[["wk", "wn"]].sum()
    proj = (agg.wk + regression * split.league_rate) / (agg.wn + regression)
    return proj.reindex(split.target.playerID.values)
