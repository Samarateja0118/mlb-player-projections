import pandas as pd
import pytest

from projections import baselines, data


def _split(hist_rows, league=0.2):
    hist = pd.DataFrame(hist_rows, columns=["playerID", "yearID", "age", "k", "n"])
    tgt = pd.DataFrame({"playerID": sorted(hist.playerID.unique()), "yearID": 2025, "age": 30, "k": 0, "n": 100})
    return data.Split(2025, hist, tgt, league)


def test_marcel_weights_recent_seasons_more():
    sp = _split([("a", 2024, 29, 60, 200), ("a", 2023, 28, 20, 200), ("a", 2022, 27, 20, 200)])
    expected = (5 * 60 + 4 * 20 + 3 * 20 + 1200 * 0.2) / (5 * 200 + 4 * 200 + 3 * 200 + 1200)
    assert baselines.marcel(sp)["a"] == pytest.approx(expected)


def test_marcel_regresses_small_samples_to_league():
    sp = _split([("a", 2024, 29, 10, 10)])  # 100% rate in 10 chances
    proj = baselines.marcel(sp)["a"]
    assert 0.2 < proj < 0.25


def test_league_average_is_flat():
    sp = _split([("a", 2024, 29, 10, 100), ("b", 2024, 29, 30, 100)], league=0.21)
    assert baselines.league_average(sp).tolist() == [0.21, 0.21]
