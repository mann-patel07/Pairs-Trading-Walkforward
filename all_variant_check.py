"""Sensitivity check on the ALL variant, added after a review. Not in PREREGISTRATION.md.

The README says ALL has a higher net Sharpe ratio than COINT. This script checks how much that
depends on pairs with a small hedge ratio and on the single best pair-window, and writes
results/all_variant_sensitivity.csv.
"""
import copy
from pathlib import Path

import pandas as pd

import config
import pairs
import run_walkforward as rw

ROOT = Path(__file__).resolve().parent


def main() -> None:
    prices = pd.read_csv(ROOT / config.PRICES_PATH, index_col=0, parse_dates=True)
    pair_list = list(config.PAIRS)
    windows = pairs.make_windows(len(prices))
    prep = rw.prepare(prices, pair_list, windows)
    elig = rw.eligibility(prep, pair_list, windows)
    sims = rw.simulate_all(prep, windows, config.ENTRY_Z, config.COST_BPS)

    def row(case, e, variant):
        s = rw.portfolio_series(sims, e, pair_list, windows, variant, "net")
        return {"case": case, "variant": variant, "eligible_pair_windows": sum(e[k][variant] for k in e),
                "net_sharpe": pairs.sharpe(s), "net_ann_return": pairs.annualized_return(s)}

    rows = [row("as published", elig, "COINT"), row("as published", elig, "ALL")]
    for floor in (0.1, 0.3, 0.5):
        e = copy.deepcopy(elig)
        for key in e:
            if prep[key]["fit"].beta < floor:
                e[key]["ALL"] = False
        rows.append(row(f"ALL with beta at least {floor}", e, "ALL"))
    pw = pd.read_csv(ROOT / "results/pair_windows.csv")
    best = pw[pw.eligible_ALL].sort_values("net_return").iloc[-1]
    e = copy.deepcopy(elig)
    for (y, x, k) in e:
        if rw.pair_label(y, x) == best["pair"] and k == best["window"]:
            e[(y, x, k)]["ALL"] = False
    rows.append(row(f"ALL without its best pair-window ({best['pair']}, window {best['window']})", e, "ALL"))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/all_variant_sensitivity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
