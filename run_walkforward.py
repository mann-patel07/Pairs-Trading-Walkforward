"""Run the pre-registered walk-forward study and write everything to results/.

Usage: python run_walkforward.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
import pairs

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VARIANTS = ["ALL", "COINT", "COINT_BH"]
VMA_NOTEBOOK_TABLE = RESULTS / "vma_case_study_windows.csv"
TOLERANCE = 1e-9


def pair_label(y: str, x: str) -> str:
    """Display name for a pair."""
    return f"{y}/{x}"


def prepare(prices: pd.DataFrame, pair_list, windows):
    """Fit every pair in every window once and keep the frozen z-scores and returns."""
    prep = {}
    for k, w in enumerate(windows):
        ts, te = w.trade_start, w.trade_end
        for y, x, _ in pair_list:
            py = prices[y].to_numpy()
            px = prices[x].to_numpy()
            fit = pairs.fit_window(py, px, w)
            adf = pairs.oos_adf_pvalue(py, px, w)
            if fit.beta > 0.0:
                z = pairs.zscore(np.log(py[ts:te]), np.log(px[ts:te]), fit.alpha, fit.beta, fit.sigma)
                r_y = py[ts:te] / py[ts - 1 : te - 1] - 1.0
                r_x = px[ts:te] / px[ts - 1 : te - 1] - 1.0
            else:
                z = r_y = r_x = None
            prep[(y, x, k)] = {"fit": fit, "z": z, "r_y": r_y, "r_x": r_x, "adf": adf}
    return prep


def simulate_all(prep, windows, entry_z: float, cost_bps: float):
    """Simulate every pair-window under one setting of entry threshold and cost."""
    sims = {}
    n = config.TRADING_DAYS
    for (y, x, k), p in prep.items():
        fit = p["fit"]
        if fit.beta > 0.0:
            sims[(y, x, k)] = pairs.simulate_trading(
                p["z"], p["r_y"], p["r_x"], fit.beta,
                entry_z=entry_z, exit_z=config.EXIT_Z, stop_z=config.STOP_Z, cost_bps=cost_bps,
            )
        else:
            zeros = np.zeros(n)
            izeros = np.zeros(n, dtype=int)
            sims[(y, x, k)] = pairs.TradingResult(izeros, izeros, zeros, zeros, zeros, zeros, [])
    return sims


def eligibility(prep, pair_list, windows):
    """Eligibility flags per (pair, window) for the three variants, plus BH rejections."""
    elig = {}
    for k in range(len(windows)):
        pvals = [prep[(y, x, k)]["fit"].pvalue for y, x, _ in pair_list]
        bh = pairs.bh_rejections(pvals)
        for i, (y, x, _) in enumerate(pair_list):
            f = prep[(y, x, k)]["fit"]
            elig[(y, x, k)] = {
                "bh_reject": bool(bh[i]),
                "ALL": pairs.eligible_all(f.beta),
                "COINT": pairs.eligible_coint(f.beta, f.pvalue, f.half_life),
                "COINT_BH": pairs.eligible_coint_bh(f.beta, bh[i], f.half_life),
            }
    return elig


def portfolio_series(sims, elig, pair_list, windows, variant: str, field: str):
    """Concatenated daily portfolio return for one variant, using net or gross returns."""
    parts = []
    for k in range(len(windows)):
        chosen = [(y, x) for y, x, _ in pair_list if elig[(y, x, k)][variant]]
        series = {pair_label(y, x): getattr(sims[(y, x, k)], field) for y, x in chosen}
        parts.append(pairs.portfolio_window_return(series, list(series), config.TRADING_DAYS))
    return np.concatenate(parts)


def gross_series(sims, elig, pair_list, windows, variant: str):
    """Portfolio return before trade costs and borrow."""
    return portfolio_series(sims, elig, pair_list, windows, variant, "gross")


def open_position_share(sims, elig, pair_list, windows, variant: str) -> float:
    """Share of days on which at least one eligible pair holds a position."""
    flags = []
    for k in range(len(windows)):
        any_open = np.zeros(config.TRADING_DAYS, dtype=bool)
        for y, x, _ in pair_list:
            if elig[(y, x, k)][variant]:
                any_open |= sims[(y, x, k)].position != 0
        flags.append(any_open)
    return float(np.concatenate(flags).mean())


def compound(r: np.ndarray) -> float:
    """Total compounded return of a daily series."""
    return float(np.prod(1.0 + r) - 1.0)


def check_vma_regression(prices: pd.DataFrame, windows, pair_windows: pd.DataFrame) -> None:
    """Recompute V/MA with evaluate_pair_window and match it to the notebook table and the CSV."""
    if not VMA_NOTEBOOK_TABLE.exists():
        raise SystemExit(f"STOP: {VMA_NOTEBOOK_TABLE} is missing. Run the V/MA notebook first.")
    nb = pd.read_csv(VMA_NOTEBOOK_TABLE)
    rows = pair_windows[pair_windows["pair"] == "V/MA"].sort_values("window").reset_index(drop=True)
    py, px = prices["V"].to_numpy(), prices["MA"].to_numpy()
    if len(nb) != len(windows) or len(rows) != len(windows):
        raise SystemExit("STOP: V/MA row count differs between notebook, CSV, and windows.")
    for k, w in enumerate(windows):
        r = pairs.evaluate_pair_window(py, px, w)
        f = r.fit
        fresh = {
            "eg_pvalue": f.pvalue,
            "half_life": f.half_life,
            "beta": f.beta,
            "n_completed_trades": float(len(r.result.trades)),
            "net_return": float(np.prod(1.0 + r.result.net) - 1.0),
            "eligible_COINT": float(pairs.eligible_coint(f.beta, f.pvalue, f.half_life)),
        }
        nb_row = nb.iloc[k]
        csv_row = rows.iloc[k]
        nb_vals = {
            "eg_pvalue": nb_row["eg_pvalue"],
            "half_life": nb_row["half_life"],
            "beta": nb_row["beta"],
            "n_completed_trades": float(nb_row["completed_trades"]),
            "net_return": nb_row["net_return"],
            "eligible_COINT": float(nb_row["eligible_COINT"] == "yes"),
        }
        for key, value in fresh.items():
            for label, other in (("notebook table", nb_vals[key]), ("pair_windows.csv", float(csv_row[key]))):
                same = value == other or abs(value - other) <= TOLERANCE
                if not same:
                    raise SystemExit(
                        f"STOP: V/MA regression check failed, window {k}, {key}: "
                        f"recomputed {value} vs {label} {other}"
                    )


def main() -> int:
    """Run the full study and write all result files."""
    os.chdir(ROOT)
    RESULTS.mkdir(exist_ok=True)
    prices = pd.read_csv(config.PRICES_PATH, index_col=0, parse_dates=True)
    dates = prices.index
    pair_list = list(config.PAIRS)
    windows = pairs.make_windows(len(prices))
    prep = prepare(prices, pair_list, windows)
    elig = eligibility(prep, pair_list, windows)
    sims = simulate_all(prep, windows, config.ENTRY_Z, config.COST_BPS)

    pw_rows, trade_rows = [], []
    for k, w in enumerate(windows):
        for y, x, sector in pair_list:
            p = prep[(y, x, k)]
            f, sim, e = p["fit"], sims[(y, x, k)], elig[(y, x, k)]
            pw_rows.append({
                "pair": pair_label(y, x),
                "sector": sector,
                "window": k,
                "formation_start": dates[w.form_start].date(),
                "formation_end": dates[w.form_end - 1].date(),
                "trading_start": dates[w.trade_start].date(),
                "trading_end": dates[w.trade_end - 1].date(),
                "alpha": f.alpha, "beta": f.beta, "sigma": f.sigma,
                "eg_pvalue": f.pvalue,
                "bh_reject": e["bh_reject"],
                "half_life": f.half_life,
                "eligible_ALL": e["ALL"], "eligible_COINT": e["COINT"], "eligible_COINT_BH": e["COINT_BH"],
                "n_completed_trades": len(sim.trades),
                "n_stops": sum(t.reason == "stop" for t in sim.trades),
                "gross_return": compound(sim.gross),
                "net_return": compound(sim.net),
                "days_in_position": int((sim.position != 0).sum()),
                "oos_adf_pvalue": p["adf"],
            })
            trade_dates = dates[w.trade_start : w.trade_end]
            for t in sim.trades:
                trade_rows.append({
                    "pair": pair_label(y, x),
                    "window": k,
                    "direction": "long_spread" if t.direction == 1 else "short_spread",
                    "entry_date": trade_dates[t.entry_idx].date(),
                    "exit_date": trade_dates[t.exit_idx].date(),
                    "entry_z": t.entry_z,
                    "exit_z": t.exit_z,
                    "exit_reason": t.reason,
                    "holding_days": t.exit_idx - t.entry_idx,
                    "net_return": pairs.trade_net_return(sim, t),
                    "eligible_COINT": e["COINT"],
                    "eligible_COINT_BH": e["COINT_BH"],
                })
    pair_windows = pd.DataFrame(pw_rows)
    pair_windows.to_csv(RESULTS / "pair_windows.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(RESULTS / "trades.csv", index=False)

    port_dates = dates[windows[0].trade_start : windows[-1].trade_end]
    spy = prices[config.BENCHMARK].pct_change().reindex(port_dates).to_numpy()
    net = {v: portfolio_series(sims, elig, pair_list, windows, v, "net") for v in VARIANTS}
    gross = {v: gross_series(sims, elig, pair_list, windows, v) for v in VARIANTS}
    out = pd.DataFrame({"date": port_dates.date})
    for v in VARIANTS:
        out[v] = net[v]
    out[config.BENCHMARK] = spy
    out.to_csv(RESULTS / "portfolio_returns.csv", index=False)

    summary_rows = []
    for v in VARIANTS:
        n_pair_windows = sum(elig[key][v] for key in elig)
        n_trades = sum(len(sims[key].trades) for key in elig if elig[key][v])
        share = open_position_share(sims, elig, pair_list, windows, v)
        for basis, series in (("gross", gross[v]), ("net", net[v])):
            slope, corr = pairs.ols_slope_and_corr(series, spy)
            summary_rows.append({
                "variant": v, "basis": basis,
                "ann_return": pairs.annualized_return(series),
                "ann_volatility": pairs.annualized_vol(series),
                "sharpe": pairs.sharpe(series),
                "max_drawdown": pairs.max_drawdown(series),
                "pair_windows_traded": n_pair_windows,
                "completed_trades": n_trades,
                "pct_days_open_position": 100.0 * share,
                "beta_to_spy": slope,
                "corr_to_spy": corr,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    cost_rows = []
    for bps in config.COST_GRID_BPS:
        s = simulate_all(prep, windows, config.ENTRY_Z, bps)
        for v in VARIANTS:
            r = portfolio_series(s, elig, pair_list, windows, v, "net")
            cost_rows.append({"variant": v, "cost_bps": bps,
                              "net_sharpe": pairs.sharpe(r), "net_ann_return": pairs.annualized_return(r)})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(RESULTS / "cost_sensitivity.csv", index=False)

    entry_rows = []
    for ez in config.ENTRY_GRID:
        s = simulate_all(prep, windows, ez, config.COST_BPS)
        for v in VARIANTS:
            r = portfolio_series(s, elig, pair_list, windows, v, "net")
            entry_rows.append({"variant": v, "entry_z": ez,
                               "net_sharpe": pairs.sharpe(r), "net_ann_return": pairs.annualized_return(r)})
    entry_df = pd.DataFrame(entry_rows)
    entry_df.to_csv(RESULTS / "entry_sensitivity.csv", index=False)

    for v in VARIANTS:
        base = summary[(summary.variant == v) & (summary.basis == "net")].iloc[0]
        c = cost_df[(cost_df.variant == v) & (cost_df.cost_bps == config.COST_BPS)].iloc[0]
        e = entry_df[(entry_df.variant == v) & (entry_df.entry_z == config.ENTRY_Z)].iloc[0]
        for other in (c, e):
            assert np.isclose(other.net_sharpe, base.sharpe, equal_nan=True, atol=TOLERANCE), \
                f"sensitivity row for {v} does not match the primary run"

    check_vma_regression(prices, windows, pair_windows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("V/MA regression check passed (recomputation matches notebook table and pair_windows.csv).")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
