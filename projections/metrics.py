"""Scoring for rate projections against realized counts.

Each plate appearance (or batter faced) is a binary event, so a projected rate p
for a player with k events in n chances is scored on all n outcomes.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def log_loss(p, k, n) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    k, n = np.asarray(k, float), np.asarray(n, float)
    return float(-(k * np.log(p) + (n - k) * np.log1p(-p)).sum() / n.sum())


def brier(p, k, n) -> float:
    p, k, n = (np.asarray(x, float) for x in (p, k, n))
    return float((k * (1 - p) ** 2 + (n - k) * p**2).sum() / n.sum())


def weighted_rmse(p, k, n) -> float:
    """RMSE of projected vs realized rate, weighted by opportunities."""
    p, k, n = (np.asarray(x, float) for x in (p, k, n))
    return float(np.sqrt((n * (p - k / n) ** 2).sum() / n.sum()))


def per_player_log_loss(p, k, n) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    k, n = np.asarray(k, float), np.asarray(n, float)
    return -(k * np.log(p) + (n - k) * np.log1p(-p))


def paired_bootstrap(loss_a, loss_b, n_opp, reps: int = 2000, seed: int = 0):
    """Bootstrap players to get a 95% interval for mean per-chance loss(a) - loss(b).
    Negative means a is better."""
    rng = np.random.default_rng(seed)
    loss_a, loss_b, n_opp = (np.asarray(x, float) for x in (loss_a, loss_b, n_opp))
    idx = rng.integers(0, len(loss_a), size=(reps, len(loss_a)))
    diffs = (loss_a[idx].sum(1) - loss_b[idx].sum(1)) / n_opp[idx].sum(1)
    point = (loss_a.sum() - loss_b.sum()) / n_opp.sum()
    return float(point), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def interval_coverage(p_draws, k, n, level: float, seed: int = 0) -> float:
    """Share of players whose realized count falls inside the central `level`
    posterior predictive interval (rate uncertainty plus binomial noise)."""
    rng = np.random.default_rng(seed)
    p_draws = np.asarray(p_draws, float)
    sims = rng.binomial(np.asarray(n)[None, :], p_draws)
    lo, hi = np.quantile(sims, [(1 - level) / 2, (1 + level) / 2], axis=0)
    k = np.asarray(k)
    return float(((k >= lo) & (k <= hi)).mean())


def calibration_table(p, k, n, bins: int = 10):
    """Opportunity-weighted projected vs realized rate within quantile bins of p."""
    import pandas as pd

    df = pd.DataFrame({"p": p, "k": k, "n": n})
    df["bin"] = pd.qcut(df.p, bins, labels=False, duplicates="drop")
    g = df.groupby("bin")
    return pd.DataFrame({
        "projected": g.apply(lambda d: (d.p * d.n).sum() / d.n.sum(), include_groups=False),
        "realized": g.k.sum() / g.n.sum(),
        "players": g.size(),
    })
