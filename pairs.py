"""Library for the pairs trading walk-forward study.

All functions are pure: inputs in, outputs out, no printing, no global state.
Default arguments come from config.py (the pre-registered constants).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

import config


class Window(NamedTuple):
    """One walk-forward window. Index ranges are half-open: [start, end)."""

    k: int
    form_start: int
    form_end: int
    trade_start: int
    trade_end: int


@dataclass(frozen=True)
class FormationFit:
    """Parameters estimated from formation data only."""

    alpha: float
    beta: float
    sigma: float
    pvalue: float
    half_life: float


@dataclass(frozen=True)
class Trade:
    """A completed trade. Indices are positions inside the trading window."""

    direction: int          # +1 long spread (long Y, short X), -1 short spread
    entry_idx: int          # day the signal fired (at the close)
    exit_idx: int           # day the exit signal fired (at the close)
    reason: str             # "revert", "stop", or "window_end"
    entry_z: float
    exit_z: float


@dataclass(frozen=True)
class TradingResult:
    """Daily arrays for one pair in one trading window."""

    states: np.ndarray
    position: np.ndarray
    gross: np.ndarray
    trade_cost: np.ndarray
    borrow: np.ndarray
    net: np.ndarray
    trades: List[Trade]


@dataclass(frozen=True)
class PairWindowResult:
    """Formation fit plus the trading-window result for one pair and one window."""

    fit: FormationFit
    traded: bool
    z: np.ndarray
    result: TradingResult


def make_windows(
    n_days: int,
    formation_days: int = config.FORMATION_DAYS,
    trading_days: int = config.TRADING_DAYS,
    step_days: int = config.STEP_DAYS,
) -> List[Window]:
    """Return all walk-forward windows whose trading period has the full length."""
    windows: List[Window] = []
    k = 0
    while k * step_days + formation_days + trading_days <= n_days:
        fs = k * step_days
        ts = fs + formation_days
        windows.append(Window(k, fs, ts, ts, ts + trading_days))
        k += 1
    return windows


def ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS of y on a constant and x. Returns (intercept, slope, residuals)."""
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(coef[0]), float(coef[1]), resid


def half_life(resid: np.ndarray) -> float:
    """Half-life of mean reversion from delta_e[t] = a + b * e[t-1]; inf if not reverting."""
    lagged = resid[:-1]
    delta = np.diff(resid)
    _, b, _ = ols_line(lagged, delta)
    if -1.0 < b < 0.0:
        return float(-math.log(2.0) / math.log(1.0 + b))
    return float("inf")


def formation_fit(log_y: np.ndarray, log_x: np.ndarray) -> FormationFit:
    """Fit log(Y) = alpha + beta * log(X) + e on formation data and test for cointegration."""
    alpha, beta, resid = ols_line(log_x, log_y)
    sigma = float(np.std(resid, ddof=1))
    pvalue = float(coint(log_y, log_x, trend="c", autolag="aic")[1])
    return FormationFit(alpha, beta, sigma, pvalue, half_life(resid))


def fit_window(prices_y: np.ndarray, prices_x: np.ndarray, window: Window) -> FormationFit:
    """Formation fit for a window. Only the formation slice of the prices is read."""
    sl = slice(window.form_start, window.form_end)
    return formation_fit(np.log(prices_y[sl]), np.log(prices_x[sl]))


def zscore(
    log_y: np.ndarray, log_x: np.ndarray, alpha: float, beta: float, sigma: float
) -> np.ndarray:
    """z[t] = (log Y - alpha - beta * log X) / sigma with frozen formation parameters."""
    return (log_y - alpha - beta * log_x) / sigma


def run_state_machine(
    z: np.ndarray,
    entry_z: float = config.ENTRY_Z,
    exit_z: float = config.EXIT_Z,
    stop_z: float = config.STOP_Z,
) -> tuple[np.ndarray, List[Trade]]:
    """Signal states decided at each close, plus the list of completed trades."""
    n = len(z)
    states = np.zeros(n, dtype=int)
    trades: List[Trade] = []
    state = 0
    stopped = False
    open_idx = -1
    for t in range(n):
        last = t == n - 1
        new = state
        reason = ""
        if stopped:
            new = 0
        elif state == 0:
            if -stop_z < z[t] <= -entry_z:
                new = 1
            elif entry_z <= z[t] < stop_z:
                new = -1
        elif state == 1:
            if z[t] <= -stop_z:
                new, stopped, reason = 0, True, "stop"
            elif z[t] >= exit_z:
                new, reason = 0, "revert"
        else:
            if z[t] >= stop_z:
                new, stopped, reason = 0, True, "stop"
            elif z[t] <= exit_z:
                new, reason = 0, "revert"
        if last and new != 0:
            if state == 0:
                new = 0
            else:
                new, reason = 0, "window_end"
        if state == 0 and new != 0:
            open_idx = t
        elif state != 0 and new == 0:
            trades.append(Trade(state, open_idx, t, reason, float(z[open_idx]), float(z[t])))
        states[t] = new
        state = new
    return states, trades


def lag_positions(states: np.ndarray) -> np.ndarray:
    """position[t] = state[t-1], and position[0] = 0."""
    position = np.zeros_like(states)
    position[1:] = states[:-1]
    return position


def gross_returns(
    position: np.ndarray, r_y: np.ndarray, r_x: np.ndarray, beta: float
) -> np.ndarray:
    """Daily gross return with unit gross exposure: pos * (rY - beta * rX) / (1 + beta)."""
    return position * (r_y - beta * r_x) / (1.0 + beta)


def trade_costs(position: np.ndarray, cost_bps: float = config.COST_BPS) -> np.ndarray:
    """Cost of position changes, plus the closing cost on the last day of the window."""
    prev = np.concatenate([[0], position[:-1]])
    cost = np.abs(position - prev) * cost_bps / 10000.0
    cost[-1] += abs(position[-1]) * cost_bps / 10000.0
    return cost


def borrow_costs(
    position: np.ndarray,
    beta: float,
    borrow_annual: float = config.BORROW_ANNUAL,
    days_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Daily borrow fee on the short leg while a position is open."""
    short_fraction = np.where(position == 1, beta / (1.0 + beta), 1.0 / (1.0 + beta))
    return np.where(position != 0, borrow_annual / days_per_year * short_fraction, 0.0)


def simulate_trading(
    z: np.ndarray,
    r_y: np.ndarray,
    r_x: np.ndarray,
    beta: float,
    entry_z: float = config.ENTRY_Z,
    exit_z: float = config.EXIT_Z,
    stop_z: float = config.STOP_Z,
    cost_bps: float = config.COST_BPS,
) -> TradingResult:
    """Run signals, lag them one day, and compute gross, cost, borrow, and net returns."""
    states, trades = run_state_machine(z, entry_z, exit_z, stop_z)
    position = lag_positions(states)
    gross = gross_returns(position, r_y, r_x, beta)
    cost = trade_costs(position, cost_bps)
    borrow = borrow_costs(position, beta)
    return TradingResult(states, position, gross, cost, borrow, gross - cost - borrow, trades)


def trade_net_return(result: TradingResult, trade: Trade) -> float:
    """Compounded net return of one trade, including its closing cost day."""
    last = len(result.net) - 1
    end = min(trade.exit_idx + 1, last)
    return float(np.prod(1.0 + result.net[trade.entry_idx + 1 : end + 1]) - 1.0)


def evaluate_pair_window(
    prices_y: np.ndarray,
    prices_x: np.ndarray,
    window: Window,
    entry_z: float = config.ENTRY_Z,
    exit_z: float = config.EXIT_Z,
    stop_z: float = config.STOP_Z,
    cost_bps: float = config.COST_BPS,
) -> PairWindowResult:
    """Fit on the formation slice, then trade the trading slice with frozen parameters."""
    fit = fit_window(prices_y, prices_x, window)
    n = window.trade_end - window.trade_start
    ts, te = window.trade_start, window.trade_end
    if fit.beta <= 0.0:
        zeros = np.zeros(n)
        empty = TradingResult(
            np.zeros(n, dtype=int), np.zeros(n, dtype=int), zeros, zeros, zeros, zeros, []
        )
        return PairWindowResult(fit, False, np.full(n, np.nan), empty)
    py, px = prices_y[ts - 1 : te], prices_x[ts - 1 : te]
    r_y = py[1:] / py[:-1] - 1.0
    r_x = px[1:] / px[:-1] - 1.0
    z = zscore(np.log(prices_y[ts:te]), np.log(prices_x[ts:te]), fit.alpha, fit.beta, fit.sigma)
    result = simulate_trading(z, r_y, r_x, fit.beta, entry_z, exit_z, stop_z, cost_bps)
    return PairWindowResult(fit, True, z, result)


def oos_adf_pvalue(prices_y: np.ndarray, prices_x: np.ndarray, window: Window) -> float:
    """ADF p-value of the trading-window spread built with frozen formation parameters."""
    fit = fit_window(prices_y, prices_x, window)
    sl = slice(window.trade_start, window.trade_end)
    s = np.log(prices_y[sl]) - fit.alpha - fit.beta * np.log(prices_x[sl])
    return float(adfuller(s, autolag="AIC", result_object=True).pvalue)


def eligible_all(beta: float) -> bool:
    """ALL variant: every pair with a positive hedge ratio."""
    return beta > 0.0


def eligible_coint(
    beta: float,
    pvalue: float,
    half_life_days: float,
    alpha: float = config.COINT_ALPHA,
    min_hl: float = config.MIN_HALF_LIFE,
    max_hl: float = config.MAX_HALF_LIFE,
) -> bool:
    """COINT variant: beta > 0, raw p-value below alpha, half-life inside the allowed band."""
    return beta > 0.0 and pvalue < alpha and min_hl <= half_life_days <= max_hl


def bh_rejections(pvalues: Sequence[float], fdr: float = config.BH_FDR) -> np.ndarray:
    """Benjamini-Hochberg rejections across all pairs tested in one formation window."""
    return np.asarray(multipletests(pvalues, alpha=fdr, method="fdr_bh")[0], dtype=bool)


def eligible_coint_bh(
    beta: float,
    bh_reject: bool,
    half_life_days: float,
    min_hl: float = config.MIN_HALF_LIFE,
    max_hl: float = config.MAX_HALF_LIFE,
) -> bool:
    """COINT_BH variant: like COINT, but the p-value rule is the BH rejection."""
    return beta > 0.0 and bool(bh_reject) and min_hl <= half_life_days <= max_hl


def portfolio_window_return(
    returns: Dict[str, np.ndarray], eligible: Sequence[str], n_days: int
) -> np.ndarray:
    """Equal-weight average of eligible pairs' daily returns; zeros if none are eligible."""
    if len(eligible) == 0:
        return np.zeros(n_days)
    return np.mean([returns[name] for name in eligible], axis=0)


def equity_curve(returns: np.ndarray) -> np.ndarray:
    """Cumulative equity starting at 1 (the first element is the value after day one)."""
    return np.cumprod(1.0 + np.asarray(returns, dtype=float))


def annualized_return(
    returns: np.ndarray, days_per_year: int = config.TRADING_DAYS_PER_YEAR
) -> float:
    """(prod(1 + r)) ** (252 / n) - 1."""
    r = np.asarray(returns, dtype=float)
    return float(np.prod(1.0 + r) ** (days_per_year / len(r)) - 1.0)


def annualized_vol(
    returns: np.ndarray, days_per_year: int = config.TRADING_DAYS_PER_YEAR
) -> float:
    """Sample standard deviation times sqrt(252)."""
    return float(np.std(returns, ddof=1) * math.sqrt(days_per_year))


def sharpe(returns: np.ndarray, days_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    """Excess-return Sharpe: mean / std (ddof=1) * sqrt(252), with no risk-free subtraction."""
    r = np.asarray(returns, dtype=float)
    sd = np.std(r, ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(np.mean(r) / sd * math.sqrt(days_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough fall of the equity curve (a negative number or zero)."""
    eq = np.concatenate([[1.0], equity_curve(returns)])
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))


def ols_slope_and_corr(
    y: np.ndarray, x: np.ndarray
) -> tuple[float, float]:
    """OLS slope of y on x and the Pearson correlation of the two series."""
    _, slope, _ = ols_line(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return slope, float(np.corrcoef(x, y)[0, 1])


def circular_block_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of one circular block bootstrap resample of length n."""
    n_blocks = -(-n // block_length)
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_length)[None, :]) % n
    return idx.ravel()[:n]


def block_bootstrap_sharpe(
    returns: np.ndarray,
    reps: int = config.BOOTSTRAP_REPS,
    block_length: int = config.BLOCK_LENGTH,
    seed: Optional[int] = config.RANDOM_SEED,
) -> np.ndarray:
    """Annualized Sharpe of each circular block bootstrap resample."""
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        out[i] = sharpe(r[circular_block_bootstrap_indices(len(r), block_length, rng)])
    return out


def pair_bootstrap_means(
    pair_means: np.ndarray,
    reps: int = config.BOOTSTRAP_REPS,
    seed: Optional[int] = config.RANDOM_SEED,
) -> np.ndarray:
    """Means of resamples of per-pair mean returns, drawing pairs with replacement."""
    values = np.asarray(pair_means, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    return values[idx].mean(axis=1)
