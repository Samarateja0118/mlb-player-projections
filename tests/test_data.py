import pandas as pd

from projections import data


def _people():
    return pd.DataFrame({"playerID": ["a", "b"], "birthYear": [1995, 1995], "birthMonth": [6, 7]})


def test_age_uses_july_1_cutoff():
    ppl = _people()
    age = data.age_on_july_1(ppl.birthYear, ppl.birthMonth, pd.Series([2025, 2025]))
    assert age.tolist() == [30, 29]


def test_batting_stints_are_summed_and_pa_computed():
    bat = pd.DataFrame({
        "playerID": ["a", "a"], "yearID": [2024, 2024], "AB": [100, 50], "BB": [10, 5], "HBP": [1, 0],
        "SH": [0, 1], "SF": [2, 0], "SO": [20, 10], "HR": [3, 1],
    })
    out = data.batting_lines(bat)
    assert len(out) == 1
    assert out.PA.iloc[0] == 169
    assert out.SO.iloc[0] == 30


def _lines():
    rows = []
    for y in range(2020, 2026):
        rows += [("a", y, 28, 30, 150), ("b", y, 25, 20, 150)]
    rows += [("c", 2025, 24, 50, 200)]  # rookie in target year: no history
    rows += [("d", 2024, 30, 5, 40), ("d", 2025, 31, 25, 150)]  # too little history
    return data.SeasonData("x", pd.DataFrame(rows, columns=["playerID", "yearID", "age", "k", "n"]))


def test_split_never_uses_target_year_in_history():
    sp = data.split(_lines(), 2025)
    assert sp.history.yearID.max() == 2024
    assert sp.history.yearID.min() == 2022


def test_split_requires_history_and_target_volume():
    sp = data.split(_lines(), 2025)
    assert sorted(sp.target.playerID) == ["a", "b"]
    assert set(sp.history.playerID) == {"a", "b"}


def test_league_rate_is_prior_season_only():
    sp = data.split(_lines(), 2025)
    assert sp.league_rate == (30 + 20 + 5) / (150 + 150 + 40)
