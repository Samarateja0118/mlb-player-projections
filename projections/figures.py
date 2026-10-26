"""Figures from backtest outputs: calibration by stat and log-loss skill vs league average.

    python -m projections.figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics
from .backtest import OUT

LABELS = {"bat_k": "Hitter K%", "bat_bb": "Hitter BB%", "bat_hr": "Hitter HR%", "pit_k": "Pitcher K%"}


def calibration(det: pd.DataFrame):
    stats = [s for s in LABELS if s in set(det.stat)]
    fig, axes = plt.subplots(1, len(stats), figsize=(4 * len(stats), 4))
    for ax, stat in zip(axes if len(stats) > 1 else [axes], stats):
        g = det[det.stat == stat]
        for name, style in [("bayes", "o-"), ("marcel", "s--")]:
            tab = metrics.calibration_table(g[name], g.k, g.n, bins=10)
            ax.plot(tab.projected, tab.realized, style, label=name, ms=4)
        lo, hi = g.bayes.min(), g.bayes.max()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=1)
        ax.set_title(LABELS[stat])
        ax.set_xlabel("projected rate (decile mean)")
        ax.set_ylabel("realized rate")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=120)


def skill(summary: pd.DataFrame):
    s = summary.copy()
    for m in ["marcel", "bayes"]:
        s[f"{m}_skill"] = 1 - s[f"{m}_logloss"] / s["league_logloss"]
    print(s[["stat", "year", "players", "marcel_skill", "bayes_skill", "coverage50", "coverage80", "rhat_max"]].round(4)
          .to_string(index=False))
    return s


def main():
    det = pd.read_csv(OUT / "backtest_players.csv")
    calibration(det)
    skill(pd.read_csv(OUT / "backtest_summary.csv"))


if __name__ == "__main__":
    main()
