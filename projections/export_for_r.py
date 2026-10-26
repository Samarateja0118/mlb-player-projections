"""Write the exact backtest splits to CSV so the R replication (r/replicate.R) scores the same players.

    python -m projections.export_for_r
"""
from __future__ import annotations

import pandas as pd

from . import baselines, data
from .backtest import OUT


def main(stats=("bat_k", "bat_bb", "bat_hr", "pit_k"), years=(2023, 2024, 2025)):
    d = OUT / "r_input"
    d.mkdir(parents=True, exist_ok=True)
    for stat in stats:
        sd = data.load(stat)
        for y in years:
            sp = data.split(sd, y)
            sp.history.assign(league_rate=sp.league_rate).to_csv(d / f"{stat}_{y}_history.csv", index=False)
            tgt = sp.target.copy()
            tgt["marcel_py"] = baselines.marcel(sp).reindex(tgt.playerID).to_numpy()
            tgt.to_csv(d / f"{stat}_{y}_target.csv", index=False)


if __name__ == "__main__":
    main()
