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
