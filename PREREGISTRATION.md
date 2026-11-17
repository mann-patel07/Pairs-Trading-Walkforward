# Pre-registration

Written before any price data was downloaded. These choices were fixed before any results were computed and were not changed afterward.

## Research question

Does Engle-Granger cointegration measured in a formation window predict profitable mean reversion in the following out-of-sample trading window, for economically matched US large-cap pairs, after costs?

## Universe

Thirty pairs, chosen by business similarity before I looked at any price statistics. In each pair the first ticker is Y (dependent variable) and the second is X.

| Pair (Y, X) | Why it was chosen |
|---|---|
| V, MA | payments networks |
| KO, PEP | beverages |
| XOM, CVX | integrated oil |
| COP, EOG | exploration and production |
| JPM, BAC | money-center banks |
| GS, MS | investment banks |
| USB, PNC | regional banks |
| AXP, COF | consumer credit |
| HD, LOW | home improvement retail |
| WMT, TGT | big-box retail |
| PG, CL | household products |
| MCD, YUM | quick-service restaurants |
| MO, PM | tobacco |
| UPS, FDX | parcel delivery |
| UNP, CSX | railroads |
| DAL, UAL | airlines |
| GM, F | automakers |
| CAT, DE | heavy machinery |
| LMT, NOC | defense |
| MRK, PFE | pharmaceuticals |
| ABT, MDT | medical devices |
| DUK, SO | regulated utilities |
| T, VZ | telecom |
| SPGI, MCO | credit rating agencies |
| CME, ICE | exchanges |
| AMAT, LRCX | semiconductor equipment |
| AMT, CCI | cell towers |
| PSA, EXR | self-storage REITs |
| SHW, PPG | paints and coatings |
| ADP, PAYX | payroll services |

The benchmark is SPY. If a ticker fails to download or starts after 2015-01-10, its pair is dropped and nothing is substituted. If SPY fails or more than 5 pairs are dropped, the study stops.

## Constants

All of these are in `config.py`.

| Constant | Value |
|---|---|
| START_DATE | 2015-01-01 |
| END_DATE_EXCLUSIVE | 2026-11-01 (last trading day used is 2026-10-30) |
| FORMATION_DAYS | 504 |
| TRADING_DAYS | 126 |
| STEP_DAYS | 126 |
| ENTRY_Z | 2.0 |
| EXIT_Z | 0.0 |
| STOP_Z | 4.0 |
| COINT_ALPHA | 0.05 |
| BH_FDR | 0.10 |
| MIN_HALF_LIFE | 1.0 trading days |
| MAX_HALF_LIFE | 63.0 trading days |
| COST_BPS | 5.0 (illustrative one-way cost per unit of gross notional traded) |
| BORROW_ANNUAL | 0.005 (illustrative annual borrow fee on the short leg) |
| TRADING_DAYS_PER_YEAR | 252 |
| BOOTSTRAP_REPS | 10000 |
| BLOCK_LENGTH | 21 |
| RANDOM_SEED | 42 |
| COST_GRID_BPS | 0, 2, 5, 10 (sensitivity only) |
| ENTRY_GRID | 1.5, 2.0, 2.5 (sensitivity only) |
| BENCHMARK | SPY |

## Trading rules

**Windows.** On the common trading-day index with N days, window k (k = 0, 1, 2, ...) has formation days `[k*STEP_DAYS, k*STEP_DAYS + FORMATION_DAYS)` and trading days `[k*STEP_DAYS + FORMATION_DAYS, k*STEP_DAYS + FORMATION_DAYS + TRADING_DAYS)`. Only windows whose trading period has the full TRADING_DAYS are kept. The final partial window is dropped.

**Formation fit (formation data only).** With natural-log prices, fit OLS `log(Y) = alpha + beta * log(X) + e`. Record alpha, beta, and sigma (sample standard deviation of the residuals, ddof=1).
- Engle-Granger p-value: `statsmodels.tsa.stattools.coint(logY, logX, trend="c", autolag="aic")`.
- Half-life: regress `delta_e[t] = a + b * e[t-1]`. If `-1 < b < 0`, half_life = `-ln(2) / ln(1 + b)`. Otherwise it is infinite.

**Spread and z-score.** In the trading window, `s[t] = log(Y[t]) - alpha - beta * log(X[t])` and `z[t] = s[t] / sigma`. Alpha, beta, and sigma come only from the formation fit and stay frozen for the whole trading window.

**Signal state machine.** State starts at 0 at the start of every trading window, with `stopped` False. Each day, using z[t] at the close:
- If stopped, state is 0.
- If state is 0: when `-STOP_Z < z <= -ENTRY_Z`, state becomes +1 (long Y, short X). When `ENTRY_Z <= z < STOP_Z`, state becomes -1. Otherwise it stays 0.
- If state is +1: when `z <= -STOP_Z`, close, set stopped, reason "stop". Else when `z >= EXIT_Z`, close, reason "revert".
- If state is -1: when `z >= STOP_Z`, close, set stopped, reason "stop". Else when `z <= EXIT_Z`, close, reason "revert".
- A day on which a position is closed cannot also open a new one.
- On the last day of the trading window, state is forced to 0. An open position gets reason "window_end".
- After a stop there is no re-entry for the rest of that trading window.

**Position and no look-ahead.** `position[t] = state[t-1]`. The position on the first trading day is 0.

**Daily return.** `gross[t] = position[t] * (rY[t] - beta * rX[t]) / (1 + beta)`, using simple daily returns. Gross exposure is normalized to 1 unit of capital. Rebalancing costs between entry and exit are ignored (a stated limitation).

**Costs.** `trade_cost[t] = abs(position[t] - position[t-1]) * COST_BPS / 10000`. Borrow on days with a position: `(BORROW_ANNUAL / TRADING_DAYS_PER_YEAR) * short_fraction`, where short_fraction is `beta / (1 + beta)` if position is +1 and `1 / (1 + beta)` if position is -1. `net = gross - trade_cost - borrow`. The closing trade cost is charged on the last day of the window so every window ends fully costed.

**Excess returns.** Long-short returns are treated as excess returns. No cash yield is added. Sharpe = mean(daily) / std(daily, ddof=1) * sqrt(252).

## Strategy variants

A pair is skipped for a window if beta <= 0.
- `ALL`: trade every pair with beta > 0.
- `COINT` (primary): beta > 0, Engle-Granger p < COINT_ALPHA, and MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE.
- `COINT_BH`: as COINT, but the raw p-value rule is replaced by Benjamini-Hochberg across all pairs tested in that formation window (`multipletests(..., alpha=BH_FDR, method="fdr_bh")`).

Capital is split equally across the pairs eligible under the variant. Portfolio daily return is the simple average of the eligible pairs' daily net returns. If no pair is eligible, the window return is 0.

## Primary metric

Net annualized Sharpe ratio of the COINT variant, at ENTRY_Z = 2.0 and COST_BPS = 5.0. The cost and entry grids are robustness tables only.

## Statement

These choices were fixed before any results were computed and were not changed afterward.
