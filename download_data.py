"""Download adjusted close prices once and cache them in data/prices.csv.

Usage: python download_data.py --refresh
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

ROOT = Path(__file__).resolve().parent


def fetch_adj_close(ticker: str) -> pd.Series:
    """Download one ticker and return its adjusted close series (empty on failure)."""
    df = yf.download(
        ticker,
        start=config.START_DATE,
        end=config.END_DATE_EXCLUSIVE,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna().astype(float)
    s.name = ticker
    return s


def main() -> int:
    """Run the download, apply the pre-registered stop rules, and write the cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    args = parser.parse_args()
    prices_file = ROOT / config.PRICES_PATH
    if prices_file.exists() and not args.refresh:
        print(f"{prices_file} already exists. Use --refresh to re-download.")
        return 0

    series = {}
    failed = {}
    tickers = [config.BENCHMARK] + [t for p in config.PAIRS for t in p[:2]]
    cutoff = pd.Timestamp(config.LATEST_FIRST_VALID_DATE)
    for t in tickers:
        try:
            s = fetch_adj_close(t)
        except Exception as exc:
            failed[t] = f"download raised: {exc}"
            continue
        if s.empty:
            failed[t] = "empty frame"
        elif s.index[0] > cutoff:
            failed[t] = f"first valid date {s.index[0].date()} is after {cutoff.date()}"
        else:
            series[t] = s
    for t, why in failed.items():
        print(f"WARNING: {t}: {why}")

    if config.BENCHMARK in failed:
        print("STOP: SPY failed to download.")
        return 1

    kept_pairs, dropped_pairs = [], []
    for y, x, _ in config.PAIRS:
        if y in series and x in series:
            kept_pairs.append([y, x])
        else:
            dropped_pairs.append([y, x])
            print(f"WARNING: dropping pair ({y}, {x})")
    if len(dropped_pairs) > config.MAX_PAIRS_DROPPED:
        print(f"STOP: {len(dropped_pairs)} pairs dropped, more than {config.MAX_PAIRS_DROPPED}.")
        return 1

    kept = [config.BENCHMARK] + [t for p in kept_pairs for t in p]
    wide = pd.concat([series[t] for t in kept], axis=1, join="inner")
    spy_days = len(series[config.BENCHMARK])
    lost = spy_days - len(wide)
    if lost > config.MAX_CALENDAR_LOSS_DAYS:
        counts = {t: spy_days - len(series[t].index.intersection(series[config.BENCHMARK].index)) for t in kept}
        print(f"STOP: inner join removed {lost} days versus SPY. Missing days per ticker: {counts}")
        return 1

    prices_file.parent.mkdir(exist_ok=True)
    wide.index.name = "Date"
    wide.to_csv(prices_file)
    meta = {
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": yf.__version__,
        "first_date": str(wide.index[0].date()),
        "last_date": str(wide.index[-1].date()),
        "n_trading_days": int(len(wide)),
        "spy_calendar_days": int(spy_days),
        "tickers_kept": kept,
        "pairs_kept": kept_pairs,
        "pairs_dropped": dropped_pairs,
        "failures": failed,
    }
    (ROOT / config.METADATA_PATH).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
