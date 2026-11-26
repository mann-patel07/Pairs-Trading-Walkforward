"""Unit tests for pairs.py. All randomness uses fixed seeds."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pairs
from pairs import Window


def make_cointegrated(n, beta=1.5, phi=0.9, eps_sd=0.01, seed=0, alpha=0.3):
    """Log prices with a known hedge ratio and a stationary AR(1) spread."""
    rng = np.random.default_rng(seed)
    log_x = 4.0 + np.cumsum(rng.normal(0.0, 0.01, n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0.0, eps_sd)
    return alpha + beta * log_x + e, log_x, e


def price_pair(seed=1):
    """Price arrays long enough for one window (504 formation + 126 trading days)."""
    log_y, log_x, _ = make_cointegrated(630, seed=seed, eps_sd=0.02)
    return np.exp(log_y), np.exp(log_x)


WINDOW0 = Window(0, 0, 504, 504, 630)


def test_no_lookahead():
    py, px = price_pair()
    base = pairs.evaluate_pair_window(py, px, WINDOW0).result
    assert len(base.trades) > 0
    rng = np.random.default_rng(5)
    for t in [0, 10, 40, 80, 110, 124]:
        py2, px2 = py.copy(), px.copy()
        cut = WINDOW0.trade_start + t + 1
        py2[cut:] *= rng.uniform(0.5, 1.5, len(py2) - cut)
        px2[cut:] *= rng.uniform(0.5, 1.5, len(px2) - cut)
        alt = pairs.evaluate_pair_window(py2, px2, WINDOW0).result
        assert np.array_equal(base.position[: t + 1], alt.position[: t + 1])


def test_formation_isolation():
    py, px = price_pair()
    base = pairs.fit_window(py, px, WINDOW0)
    rng = np.random.default_rng(6)
    py2, px2 = py.copy(), px.copy()
    py2[504:] *= rng.uniform(0.3, 3.0, 126)
    px2[504:] *= rng.uniform(0.3, 3.0, 126)
    alt = pairs.fit_window(py2, px2, WINDOW0)
    assert base == alt
    full = pairs.evaluate_pair_window(py2, px2, WINDOW0).fit
    assert full == base


def test_short_leg_sign():
    r_y = np.array([0.01, 0.01])
    r_x = np.zeros(2)
    long_ret = pairs.gross_returns(np.array([1, 1]), r_y, r_x, 1.0)
    short_ret = pairs.gross_returns(np.array([-1, -1]), r_y, r_x, 1.0)
    assert np.all(long_ret > 0)
    assert np.allclose(short_ret, -long_ret)
    assert np.allclose(long_ret, 0.005)


def reverting_z():
    z = np.zeros(20)
    z[5] = -2.5
    z[6] = -1.0
    z[7] = 0.2
    return z


def test_cost_entry_and_exit():
    z = reverting_z()
    res = pairs.simulate_trading(z, np.zeros(20), np.zeros(20), 1.0)
    assert len(res.trades) == 1
    assert res.trades[0].reason == "revert"
    assert res.trade_cost.sum() == pytest.approx(2 * config.COST_BPS / 10000)


def test_stop_blocks_reentry():
    z = np.zeros(30)
    z[5], z[6], z[7] = -2.5, -4.5, -2.5
    z[10] = -2.5
    z[15] = 2.5
    res = pairs.simulate_trading(z, np.zeros(30), np.zeros(30), 1.0)
    assert res.trades[0].reason == "stop"
    assert len(res.trades) == 1
    assert np.all(res.states[6:] == 0)
    assert np.all(res.position[7:] == 0)


def test_forced_close_at_window_end():
    z = np.zeros(20)
    z[5:] = -2.5
    res = pairs.simulate_trading(z, np.zeros(20), np.zeros(20), 1.0)
    assert res.states[-1] == 0
    assert res.position[-1] == 1
    assert res.trades[-1].reason == "window_end"
    assert res.trade_cost[-1] == pytest.approx(config.COST_BPS / 10000)
    assert res.trade_cost.sum() == pytest.approx(2 * config.COST_BPS / 10000)


def test_entry_on_last_day_is_not_taken():
    z = np.zeros(10)
    z[-1] = -2.5
    res = pairs.simulate_trading(z, np.zeros(10), np.zeros(10), 1.0)
    assert len(res.trades) == 0
    assert res.trade_cost.sum() == 0.0


def test_no_same_day_flip():
    rng = np.random.default_rng(11)
    for _ in range(200):
        z = np.cumsum(rng.normal(0, 1.0, 126))
        states, trades = pairs.run_state_machine(z)
        entries = {tr.entry_idx for tr in trades}
        exits = {tr.exit_idx for tr in trades}
        assert not (entries & exits)
        assert np.all(states[1:] * states[:-1] != -1)
        assert states[-1] == 0


def test_synthetic_cointegrated_pair():
    log_y, log_x, _ = make_cointegrated(504, beta=1.5, phi=0.8, seed=3)
    fit = pairs.formation_fit(log_y, log_x)
    assert abs(fit.beta - 1.5) < 0.05
    assert fit.pvalue < 0.05


def test_half_life_recovery():
    true_hl = 10.0
    phi = 2.0 ** (-1.0 / true_hl)
    rng = np.random.default_rng(4)
    e = np.zeros(3000)
    for t in range(1, len(e)):
        e[t] = phi * e[t - 1] + rng.normal()
    est = pairs.half_life(e)
    assert abs(est - true_hl) / true_hl < 0.25


def test_half_life_infinite_when_not_reverting():
    explosive = 1.01 ** np.arange(300)
    assert math.isinf(pairs.half_life(explosive))


def test_random_walk_rejection_rate():
    rng = np.random.default_rng(12)
    reject = 0
    for _ in range(400):
        a = np.cumsum(rng.normal(size=504))
        b = np.cumsum(rng.normal(size=504))
        if pairs.formation_fit(a, b).pvalue < 0.05:
            reject += 1
    rate = reject / 400
    assert 0.02 <= rate <= 0.09


def test_windows_do_not_overlap_and_are_contiguous():
    wins = pairs.make_windows(2975)
    assert len(wins) == 19
    for w in wins:
        assert w.form_end <= w.trade_start
        assert w.form_end - w.form_start == config.FORMATION_DAYS
        assert w.trade_end - w.trade_start == config.TRADING_DAYS
        assert w.trade_end <= 2975
    for a, b in zip(wins[:-1], wins[1:]):
        assert b.trade_start == a.trade_end
    assert 2975 - wins[-1].trade_end < config.TRADING_DAYS


def test_bh_is_stricter_than_raw():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.5])
    bh = pairs.bh_rejections(p, 0.10)
    assert bh.sum() <= (p < 0.10).sum()
    assert bh[0]


def test_metrics_simple_cases():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert pairs.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert pairs.annualized_return(np.full(252, 0.0)) == 0.0
    assert pairs.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))


def test_lag_two_position_is_the_state_two_days_earlier():
    rng = np.random.default_rng(21)
    z = np.cumsum(rng.normal(0, 1.0, 126))
    res = pairs.simulate_trading(z, np.zeros(126), np.zeros(126), 1.0, lag=2)
    assert np.all(res.position[:2] == 0)
    assert np.array_equal(res.position[2:], res.states[:-2])
    for k in (20, 60, 100):
        z2 = z.copy()
        z2[k:] = rng.normal(0, 3.0, 126 - k)
        alt = pairs.simulate_trading(z2, np.zeros(126), np.zeros(126), 1.0, lag=2)
        # z from day k on reaches the state on day k and the position on day k + 2,
        # so every position up to and including day k + 1 stays the same
        assert np.array_equal(res.position[: k + 2], alt.position[: k + 2])


def test_lag_below_one_raises():
    with pytest.raises(ValueError):
        pairs.lag_positions(np.zeros(10, dtype=int), lag=0)
    with pytest.raises(ValueError):
        pairs.simulate_trading(np.zeros(10), np.zeros(10), np.zeros(10), 1.0, lag=0)


def test_trade_net_returns_compound_to_the_window_return():
    n = 40
    z = np.zeros(n)
    z[5], z[10] = -2.5, 0.5      # long spread, reverts
    z[15], z[20] = 2.5, -0.5     # short spread, reverts
    rng = np.random.default_rng(31)
    r_y = rng.normal(0.0, 0.01, n)
    r_x = rng.normal(0.0, 0.01, n)
    for lag in (1, 2):
        res = pairs.simulate_trading(z, r_y, r_x, 1.2, lag=lag)
        assert len(res.trades) == 2
        assert all(tr.reason == "revert" for tr in res.trades)
        by_trade = np.prod([1.0 + pairs.trade_net_return(res, tr, lag) for tr in res.trades])
        whole = np.prod(1.0 + res.net)
        assert by_trade == pytest.approx(whole, abs=1e-12, rel=0)
        assert by_trade != pytest.approx(1.0, abs=1e-6)


def test_portfolio_window_return_with_no_eligible_pairs_is_zero():
    out = pairs.portfolio_window_return({}, [], 126)
    assert out.shape == (126,) and not out.any()


def test_portfolio_window_return_is_the_elementwise_mean_of_two_pairs():
    a = np.array([0.01, -0.02, 0.03])
    b = np.array([0.03, 0.00, -0.01])
    out = pairs.portfolio_window_return({"A": a, "B": b}, ["A", "B"], 3)
    assert np.allclose(out, [0.02, -0.01, 0.01], atol=1e-15, rtol=0)


def test_portfolio_series_concatenates_windows_in_order():
    import types
    import run_walkforward as rw
    rng = np.random.default_rng(4)
    n = config.TRADING_DAYS
    pair_list = [("A", "B", "s"), ("C", "D", "s")]
    sims, expected = {}, []
    for k in range(2):
        net = {p: rng.normal(0, 0.01, n) for p in ("AB", "CD")}
        sims[("A", "B", k)] = types.SimpleNamespace(net=net["AB"])
        sims[("C", "D", k)] = types.SimpleNamespace(net=net["CD"])
        expected.append((net["AB"] + net["CD"]) / 2 if k == 0 else net["CD"])
    elig = {("A", "B", 0): {"X": True}, ("C", "D", 0): {"X": True},
            ("A", "B", 1): {"X": False}, ("C", "D", 1): {"X": True}}
    out = rw.portfolio_series(sims, elig, pair_list, [0, 1], "X", "net")
    assert np.allclose(out, np.concatenate(expected), atol=1e-15, rtol=0)


def test_rebalance_cost_is_zero_when_every_pair_earns_the_same_return():
    r = np.full(5, 0.01)
    pos = np.ones(5, dtype=int)
    out = pairs.rebalance_costs({"A": r, "B": r}, {"A": pos, "B": pos}, ["A", "B"], 5)
    assert np.allclose(out, 0.0, atol=1e-18, rtol=0)


def test_rebalance_cost_is_zero_for_a_flat_pair_and_matches_the_hand_value():
    out = pairs.rebalance_costs({"A": np.array([0.02]), "B": np.array([0.0])},
                                {"A": np.array([1]), "B": np.array([0])}, ["A", "B"], 1, cost_bps=5.0)
    assert out[0] == pytest.approx(5.0 / 1e4 * 0.5 * abs(1 - 1.02 / 1.01), abs=1e-18, rel=0)
    both_flat = pairs.rebalance_costs({"A": np.array([0.02]), "B": np.array([0.0])},
                                      {"A": np.array([0]), "B": np.array([0])}, ["A", "B"], 1)
    assert both_flat[0] == 0.0


def test_rebalance_cost_with_no_eligible_pairs_is_zero():
    out = pairs.rebalance_costs({}, {}, [], 7)
    assert out.shape == (7,) and not out.any()


def test_the_spy_benchmark_is_aligned_to_the_portfolio_dates():
    import pandas as pd
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prices = pd.read_csv(os.path.join(root, config.PRICES_PATH), index_col=0, parse_dates=True)
    windows = pairs.make_windows(len(prices))
    port_dates = prices.index[windows[0].trade_start: windows[-1].trade_end]
    spy = prices[config.BENCHMARK].pct_change().reindex(port_dates)
    assert len(spy) == len(windows) * config.TRADING_DAYS
    assert not spy.isna().any()
