"""Small synthetic fit: the model should recover a known talent spread and beat league average."""
import numpy as np
import pandas as pd
import pytest

from projections import data, metrics, model


@pytest.mark.slow
def test_model_recovers_player_differences():
    rng = np.random.default_rng(0)
    players = [f"p{i}" for i in range(60)]
    talent = rng.normal(-1.3, 0.35, len(players))
    rows = []
    for pid, t in zip(players, talent):
        for y in (2022, 2023, 2024, 2025):
            n = 400
            rows.append((pid, y, 27, rng.binomial(n, 1 / (1 + np.exp(-t))), n))
    sd = data.SeasonData("x", pd.DataFrame(rows, columns=["playerID", "yearID", "age", "k", "n"]))
    sp = data.split(sd, 2025)
    fit = model.fit(sp, draws=200, tune=200, chains=2)
    tgt = sp.target.set_index("playerID").loc[fit.players]
    p = model.point(fit).to_numpy()
    assert metrics.log_loss(p, tgt.k, tgt.n) < metrics.log_loss(np.full(len(p), sp.league_rate), tgt.k, tgt.n)
    true = pd.Series(1 / (1 + np.exp(-talent)), index=players).loc[fit.players].to_numpy()
    assert np.corrcoef(p, true)[0, 1] > 0.9
