"""Backtest: project each target season from the three seasons before it and score.

    python -m projections.backtest --years 2023 2024 2025 --stats bat_k bat_bb bat_hr pit_k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import pandas as pd

from . import baselines, data, metrics, model

OUT = Path(__file__).resolve().parent.parent / "results"


def run_one(sd: data.SeasonData, year: int, draws: int, tune: int) -> tuple[dict, pd.DataFrame]:
    sp = data.split(sd, year)
    fit = model.fit(sp, draws=draws, tune=tune)
    tgt = sp.target.set_index("playerID").loc[fit.players]
    k, n = tgt.k.to_numpy(), tgt.n.to_numpy()
    preds = {
        "league": baselines.league_average(sp).reindex(fit.players).to_numpy(),
        "marcel": baselines.marcel(sp).reindex(fit.players).to_numpy(),
        "bayes": model.point(fit).to_numpy(),
    }
    row = {"stat": sd.stat, "year": year, "players": len(k), "league_rate": sp.league_rate}
    for name, p in preds.items():
        row[f"{name}_logloss"] = metrics.log_loss(p, k, n)
        row[f"{name}_brier"] = metrics.brier(p, k, n)
        row[f"{name}_rmse"] = metrics.weighted_rmse(p, k, n)
    row["coverage50"] = metrics.interval_coverage(fit.p_draws, k, n, 0.5)
    row["coverage80"] = metrics.interval_coverage(fit.p_draws, k, n, 0.8)
    summ = fit.idata.posterior
    rhat = az.rhat(fit.idata, var_names=["mu", "sigma_player", "tau", "b_age", "season"])
    row["rhat_max"] = float(max(float(rhat[v].max()) for v in rhat.data_vars))
    row["b_age"] = summ["b_age"].mean(("chain", "draw")).values.round(5).tolist()
    row["tau"] = float(summ["tau"].mean())
    row["sigma_player"] = float(summ["sigma_player"].mean())
    detail = pd.DataFrame({"stat": sd.stat, "year": year, "playerID": fit.players, "k": k, "n": n, **preds})
    return row, detail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--stats", nargs="+", default=list(data.STATS))
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=500)
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    rows, details = [], []
    for stat in a.stats:
        sd = data.load(stat)
        for y in a.years:
            r, d = run_one(sd, y, a.draws, a.tune)
            print(json.dumps(r))
            rows.append(r)
            details.append(d)
    pd.DataFrame(rows).to_csv(OUT / "backtest_summary.csv", index=False)
    det = pd.concat(details)
    det.to_csv(OUT / "backtest_players.csv", index=False)
    # pooled paired bootstrap per stat: bayes vs marcel, bayes vs league
    boot = []
    for stat, g in det.groupby("stat"):
        lb = metrics.per_player_log_loss(g.bayes, g.k, g.n)
        for other in ["marcel", "league"]:
            lo = metrics.per_player_log_loss(g[other], g.k, g.n)
            pt, l, h = metrics.paired_bootstrap(lb, lo, g.n)
            boot.append({"stat": stat, "comparison": f"bayes - {other}", "logloss_diff": pt, "ci95_lo": l, "ci95_hi": h,
                         "player_seasons": len(g)})
    pd.DataFrame(boot).to_csv(OUT / "bootstrap.csv", index=False)
    print(pd.DataFrame(boot).to_string())


if __name__ == "__main__":
    main()
