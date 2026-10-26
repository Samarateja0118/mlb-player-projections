"""Load Lahman season lines and shape them into per-player, per-season rate data.

All rates are per plate appearance (batters) or per batter faced (pitchers).
Ages use the July 1 convention: a player's age for a season is their age on
July 1 of that year.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# stat name -> (table, event column, opportunity column)
STATS = {
    "bat_k": ("batting", "SO", "PA"),
    "bat_bb": ("batting", "BB", "PA"),
    "bat_hr": ("batting", "HR", "PA"),
    "pit_k": ("pitching", "SO", "BFP"),
}


@dataclass(frozen=True)
class SeasonData:
    """Season lines for one stat: columns playerID, yearID, age, k, n."""

    stat: str
    lines: pd.DataFrame


def age_on_july_1(birth_year: pd.Series, birth_month: pd.Series, season: pd.Series) -> pd.Series:
    """Age on July 1 of `season`; players born July or later haven't had their birthday yet."""
    return season - birth_year - (birth_month >= 7).astype(int)


def batting_lines(batting: pd.DataFrame) -> pd.DataFrame:
    """Sum stints (trades) into one row per player-season and compute PA."""
    cols = ["AB", "BB", "HBP", "SH", "SF", "SO", "HR"]
    df = batting.groupby(["playerID", "yearID"], as_index=False)[cols].sum(min_count=1).fillna(0)
    df["PA"] = df["AB"] + df["BB"] + df["HBP"] + df["SH"] + df["SF"]
    return df


def pitching_lines(pitching: pd.DataFrame) -> pd.DataFrame:
    """Sum stints into one row per player-season; BFP is batters faced."""
    return pitching.groupby(["playerID", "yearID"], as_index=False)[["SO", "BB", "HR", "BFP"]].sum(
        min_count=1
    ).fillna(0)


def build(stat: str, batting: pd.DataFrame, pitching: pd.DataFrame, people: pd.DataFrame) -> SeasonData:
    table, event, opp = STATS[stat]
    lines = batting_lines(batting) if table == "batting" else pitching_lines(pitching)
    lines = lines[lines[opp] > 0]
    ppl = people[["playerID", "birthYear", "birthMonth"]].dropna()
    lines = lines.merge(ppl, on="playerID", how="inner")
    lines["age"] = age_on_july_1(lines["birthYear"].astype(int), lines["birthMonth"].astype(int), lines["yearID"])
    out = lines.rename(columns={event: "k", opp: "n"})[["playerID", "yearID", "age", "k", "n"]]
    return SeasonData(stat, out.astype({"k": int, "n": int, "age": int}).reset_index(drop=True))


def load(stat: str) -> SeasonData:
    import pylahman

    return build(stat, pylahman.Batting(), pylahman.Pitching(), pylahman.People())


@dataclass(frozen=True)
class Split:
    """History seasons (strictly before target) and the target-season outcomes to score."""

    target_year: int
    history: pd.DataFrame
    target: pd.DataFrame
    league_rate: float


def split(data: SeasonData, target_year: int, years: int = 3, min_hist: int = 100, min_target: int = 100) -> Split:
    """Players with >= min_hist opportunities over the `years` seasons before target_year
    and >= min_target in the target season. Nothing from target_year enters history."""
    lines = data.lines
    hist = lines[(lines.yearID >= target_year - years) & (lines.yearID < target_year)]
    tot = hist.groupby("playerID")["n"].sum()
    keep = set(tot[tot >= min_hist].index)
    tgt = lines[(lines.yearID == target_year) & (lines.n >= min_target) & lines.playerID.isin(keep)]
    keep &= set(tgt.playerID)
    hist = hist[hist.playerID.isin(keep)].reset_index(drop=True)
    last = lines[lines.yearID == target_year - 1]
    league = last.k.sum() / last.n.sum()
    return Split(target_year, hist, tgt.reset_index(drop=True), float(league))
