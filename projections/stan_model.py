"""Fit the Stan version of the rate model (projections/stan/rates.stan) with PyStan."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import AGE_CENTER, Fit

STAN_FILE = Path(__file__).parent / "stan" / "rates.stan"


def stan_data(split) -> tuple[dict, np.ndarray]:
    h, t = split.history, split.target_year
    players = np.array(sorted(h.playerID.unique()))
    pidx = {p: i + 1 for i, p in enumerate(players)}
    first = int(h.yearID.min())
    target_age = split.target.set_index("playerID").age.reindex(players).to_numpy() - AGE_CENTER
    lr = split.league_rate
    d = {
        "N": len(h), "P": len(players), "S": t - first,
        "player": h.playerID.map(pidx).astype(int).tolist(),
        "season_idx": (h.yearID - first + 1).astype(int).tolist(),
        "n": h.n.astype(int).tolist(), "k": h.k.astype(int).tolist(),
        "age_c": (h.age - AGE_CENTER).astype(float).tolist(),
        "target_age_c": target_age.astype(float).tolist(),
        "league_logit": float(np.log(lr / (1 - lr))),
    }
    return d, players


def fit(split, draws: int = 500, tune: int = 500, chains: int = 4, seed: int = 1) -> Fit:
    import stan

    d, players = stan_data(split)
    posterior = stan.build(STAN_FILE.read_text(), data=d, random_seed=seed)
    sample = posterior.sample(num_chains=chains, num_samples=draws, num_warmup=tune)
    p_draws = sample["p_proj"].T  # (draws, players)
    return Fit(players, p_draws, sample)


def point(fit: Fit) -> pd.Series:
    return pd.Series(fit.p_draws.mean(axis=0), index=fit.players)
