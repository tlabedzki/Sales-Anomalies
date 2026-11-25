import numpy as np
import pandas as pd
import pytest

from sales_anomalies import baseline as b


def weekly_counts(seed=3, days=900, effect=0.3):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=days, freq="D")
    rate = 50 * np.exp(effect * (index.dayofweek == 5))
    return pd.Series(rng.poisson(rate), index=index), index


def test_poisson_fit_recovers_a_weekday_effect():
    y, days = weekly_counts()
    x = b.design(days)
    beta = b.fit_poisson(y.to_numpy(float), x, np.zeros(len(days)))
    saturday = 2 + 4  # intercept, trend, then weekdays 1..6 (Saturday is weekday 5)
    assert beta[saturday] == pytest.approx(0.3, abs=0.06)


def test_online_prediction_never_sees_the_day_it_judges():
    y, days = weekly_counts()
    x = b.design(days)
    first = pd.Timestamp("2023-01-01")
    before = b.online_poisson(y, x, first).expected
    changed = y.copy()
    changed[changed.index >= pd.Timestamp("2023-03-15")] *= 10   # a huge jump later on
    after = b.online_poisson(changed, x, first).expected
    window = (days >= first) & (days < pd.Timestamp("2023-03-15"))
    assert np.allclose(before[window], after[window])


def test_binomial_fit_recovers_the_share():
    rng = np.random.default_rng(5)
    trials = rng.integers(500, 1500, 800).astype(float)
    successes = rng.binomial(trials.astype(int), 0.08).astype(float)
    beta = b.fit_binomial(successes, trials, np.ones((800, 1)))
    assert 1 / (1 + np.exp(-beta[0])) == pytest.approx(0.08, abs=0.005)


def test_weekly_inflation_is_one_for_independent_days_and_more_when_they_run_together():
    rng = np.random.default_rng(7)
    independent = rng.standard_normal(2000)
    assert b.weekly_inflation(independent) == pytest.approx(1.0, abs=0.1)
    running = np.convolve(rng.standard_normal(2006), np.ones(7) / np.sqrt(7), "valid")
    assert b.weekly_inflation(running) > 1.8


def test_level_correction_follows_a_lasting_drift_and_stays_one_without_it():
    observed = np.full(300, 110.0)
    fitted = np.full(300, 100.0)
    drift = b.level_correction(observed, fitted, np.arange(100, 110), phi=1.0)
    assert drift == pytest.approx(np.full(10, 1.1), abs=0.01)
    steady = b.level_correction(fitted, fitted, np.arange(100, 110), phi=1.0)
    assert np.allclose(steady, 1.0)
    # The 8 days before a judged day are left out: a jump there does not move it.
    jump = fitted.copy()
    jump[95:100] = 1000
    assert np.allclose(b.level_correction(jump, fitted, np.array([100]), phi=1.0), 1.0)


def test_reference_skips_the_last_eight_days():
    sums = pd.Series(10.0, index=range(120))
    counts = pd.Series(1.0, index=range(120))
    sums.iloc[110:] = 1000.0
    assert b.reference(sums, counts).iloc[115] == pytest.approx(10.0)
