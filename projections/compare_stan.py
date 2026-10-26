"""Cross-check: fit the Stan and PyMC versions on the same split and compare.

    python -m projections.compare_stan --stat bat_k --years 2023 2024 2025
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import baselines, data, metrics, model, stan_model
from .backtest import OUT


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", default="bat_k")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    a = ap.parse_args(argv)
    sd = data.load(a.stat)
    rows = []
    for y in a.years:
        sp = data.split(sd, y)
        st = stan_model.fit(sp)
        pm_ = model.fit(sp)
        assert list(st.players) == list(pm_.players)
        tgt = sp.target.set_index("playerID").loc[st.players]
        k, n = tgt.k.to_numpy(), tgt.n.to_numpy()
        ps, pp = stan_model.point(st).to_numpy(), model.point(pm_).to_numpy()
        pmarcel = baselines.marcel(sp).reindex(st.players).to_numpy()
        row = {
            "stat": a.stat, "year": y, "players": len(k),
            "stan_logloss": metrics.log_loss(ps, k, n), "pymc_logloss": metrics.log_loss(pp, k, n),
            "marcel_logloss": metrics.log_loss(pmarcel, k, n),
            "stan_rmse": metrics.weighted_rmse(ps, k, n), "pymc_rmse": metrics.weighted_rmse(pp, k, n),
            "corr_stan_pymc": float(np.corrcoef(ps, pp)[0, 1]),
            "max_abs_diff": float(np.abs(ps - pp).max()),
            "stan_coverage80": metrics.interval_coverage(st.p_draws, k, n, 0.8),
        }
        print(json.dumps(row))
        rows.append(row)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"stan_vs_pymc_{a.stat}.csv", index=False)


if __name__ == "__main__":
    main()
