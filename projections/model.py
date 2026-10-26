"""Hierarchical Bayesian rate model in PyMC.

For player i in season s (logit scale):

    logit p[i, s] = theta[i, s] + aging(age[i, s]) + season[s]
    theta[i, first] ~ Normal(mu, sigma_player)          # partial pooling toward the league
    theta[i, s]     = theta[i, s-1] + Normal(0, tau)      # talent drifts year to year
    aging(a)        = b1 * (a - 27) + b2 * (a - 27)^2
    k[i, s]         ~ Binomial(n[i, s], p[i, s])

The projection for the target season steps theta forward once more, applies the
aging curve at the player's new age, and reuses the last observed season effect.
Seasons a player missed stay in the random walk with no likelihood term.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit

AGE_CENTER = 27


@dataclass
class Fit:
    players: np.ndarray
    p_draws: np.ndarray  # (draws, players): projected rate for the target season
    idata: object


def fit(split, draws: int = 500, tune: int = 500, chains: int = 4, seed: int = 0) -> Fit:
    h, t = split.history, split.target_year
    players = np.array(sorted(h.playerID.unique()))
    pidx = {p: i for i, p in enumerate(players)}
    years = list(range(int(h.yearID.min()), t))
    S = len(years)
    i = h.playerID.map(pidx).to_numpy()
    s = (h.yearID - years[0]).to_numpy()
    age_c = (h.age - AGE_CENTER).to_numpy()

    target_age = (
        split.target.set_index("playerID").age.reindex(players).to_numpy() - AGE_CENTER
    )
    lg = np.log(split.league_rate / (1 - split.league_rate))

    with pm.Model() as m:
        mu = pm.Normal("mu", lg, 0.5)
        sigma_p = pm.HalfNormal("sigma_player", 1.0)
        tau = pm.HalfNormal("tau", 0.3)
        b = pm.Normal("b_age", 0, 0.05, shape=2)
        season = pm.ZeroSumNormal("season", 0.2, shape=S)
        z0 = pm.Normal("z0", 0, 1, shape=len(players))
        steps = pm.Normal("steps", 0, 1, shape=(len(players), S))  # S-1 drift steps + 1 projection step
        theta0 = mu + sigma_p * z0
        walk = theta0[:, None] + tau * pm.math.concatenate(
            [pm.math.zeros((len(players), 1)), pm.math.cumsum(steps[:, : S - 1], axis=1)], axis=1
        )
        aging = b[0] * age_c + b[1] * age_c**2
        logit_p = walk[i, s] + aging + season[s]
        pm.Binomial("k", n=h.n.to_numpy(), logit_p=logit_p, observed=h.k.to_numpy())
        theta_next = walk[:, S - 1] + tau * steps[:, S - 1]
        pm.Deterministic(
            "logit_proj",
            theta_next + b[0] * target_age + b[1] * target_age**2 + season[S - 1],
        )
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False,
                          target_accept=0.9)
    lp = idata.posterior["logit_proj"].stack(d=("chain", "draw")).transpose("d", ...).values
    return Fit(players, expit(lp), idata)


def point(fit: Fit) -> pd.Series:
    return pd.Series(fit.p_draws.mean(axis=0), index=fit.players)
