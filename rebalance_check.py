"""Rebalancing cost check, added after the first version (see UPGRADE_PLAN.md). Not in PREREGISTRATION.md.

The portfolio holds an equal weight in each eligible pair every day. Keeping the weights equal means moving
money between pairs, and the first version does not cost that. This script computes the cost for ALL, COINT and
COINT_BH and writes results/rebalance_cost_sensitivity.csv. It does not change any other result.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import pairs
import run_walkforward as rw

ROOT = Path(__file__).resolve().parent
VARIANTS = ["ALL", "COINT", "COINT_BH"]


def cost_series(sims, elig, pair_list, windows, variant):
    """Daily rebalancing cost, window by window, concatenated in order."""
    parts = []
    for k in range(len(windows)):
        chosen = [(y, x) for y, x, _ in pair_list if elig[(y, x, k)][variant]]
        rets = {rw.pair_label(y, x): sims[(y, x, k)].net for y, x in chosen}
        poss = {rw.pair_label(y, x): sims[(y, x, k)].position for y, x in chosen}
        parts.append(pairs.rebalance_costs(rets, poss, list(rets), config.TRADING_DAYS))
    return np.concatenate(parts)


def main() -> None:
    t0 = time.time()
    prices = pd.read_csv(ROOT / config.PRICES_PATH, index_col=0, parse_dates=True)
    pair_list = list(config.PAIRS)
    windows = pairs.make_windows(len(prices))
    prep = rw.prepare(prices, pair_list, windows)
    elig = rw.eligibility(prep, pair_list, windows)
    sims = rw.simulate_all(prep, windows, config.ENTRY_Z, config.COST_BPS)
    summary = pd.read_csv(ROOT / "results/summary.csv")
    rows = []
    for v in VARIANTS:
        net = rw.portfolio_series(sims, elig, pair_list, windows, v, "net")
        cost = cost_series(sims, elig, pair_list, windows, v)
        after = net - cost
        published = float(summary[(summary.variant == v) & (summary.basis == "net")].sharpe.iloc[0])
        assert abs(pairs.sharpe(net) - published) < 1e-9, f"net Sharpe of {v} differs from summary.csv"
        rows.append({
            "variant": v,
            "net_sharpe": pairs.sharpe(net), "net_sharpe_with_rebalance_cost": pairs.sharpe(after),
            "net_ann_return": pairs.annualized_return(net),
            "net_ann_return_with_rebalance_cost": pairs.annualized_return(after),
            "max_drawdown": pairs.max_drawdown(net), "max_drawdown_with_rebalance_cost": pairs.max_drawdown(after),
            "mean_daily_rebalance_cost_bps": float(cost.mean() * 1e4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/rebalance_cost_sensitivity.csv", index=False)
    print(out.to_string(index=False))
    print("run time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
