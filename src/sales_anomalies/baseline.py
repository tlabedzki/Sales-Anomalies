"""
What a normal day would look like: expected counts from a regression fitted only on the past.

Counts (order lines, returns, sessions) follow a Poisson regression and conversion a binomial (logistic) one, both
with weekday, month, a linear trend, the shop's special days and the campaigns running in the channel. The model is
refitted at the start of every month on the two years before it and predicts that month only, so a day is always
judged by a model that has not seen it. This is the idea of the Farrington algorithm used in disease surveillance
(Farrington et al., 1996): a quasi-Poisson regression on past counts, robust to past outbreaks. Like there, days
the first fit finds far off (|Pearson residual| > 4) are dropped before the final fit, and the dispersion (how much
more the counts vary than a Poisson count) widens the band.

Order lines of a category in one channel (a cell) are sparse, so a cell is modelled as its category's expected
lines times the cell's share, which depends on trend and the channel's campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sales_anomalies.calendar import EVENT_ORDER, events

TRAINING_DAYS = 730
OUTLIER_RESIDUAL = 4.0
RIDGE = 1.0


PEAK_EVENTS = ("black_friday", "cyber_monday", "black_week", "singles_day")


def design(days: pd.DatetimeIndex, campaigns: pd.DataFrame | None = None, calendar: bool = True) -> np.ndarray:
    """
    Columns: intercept, trend (years); with calendar: 6 weekdays, 11 months and every special day, without it only
    the four promotion peaks (a cell's share of its category shifts on Black Friday); then campaigns.
    """
    columns = [np.ones(len(days)), (days - days[0]).days.to_numpy() / 365.25]
    names = events(days).to_numpy()
    if calendar:
        weekday, month = days.dayofweek.to_numpy(), days.month.to_numpy()
        columns += [(weekday == d).astype(float) for d in range(1, 7)]
        columns += [(month == m).astype(float) for m in range(2, 13)]
        columns += [(names == name).astype(float) for name in EVENT_ORDER]
    else:
        columns += [(names == name).astype(float) for name in PEAK_EVENTS]
    if campaigns is not None:
        # Running or not: a count would extrapolate (four social bursts at once never happened before).
        columns += [(campaigns[c].to_numpy() > 0).astype(float) for c in campaigns.columns]
    return np.column_stack(columns)


def _solve(x: np.ndarray, w: np.ndarray, z: np.ndarray) -> np.ndarray:
    # The penalty is RIDGE average days of data, so it means the same for counts and for amounts in PLN. It keeps
    # an effect seen once or twice (Easter in a small category with no sale that day) from running to zero sales.
    penalty = np.full(x.shape[1], RIDGE * max(float(np.mean(w)), 1e-9))
    penalty[0] = 0.0
    xtw = x.T * w
    return np.linalg.solve(xtw @ x + np.diag(penalty), xtw @ z)


def fit_poisson(y: np.ndarray, x: np.ndarray, offset: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Ridge Poisson regression by iteratively reweighted least squares; returns the coefficients."""
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-3) / max(np.exp(offset).mean(), 1e-9))
    for _ in range(iterations):
        eta = np.clip(x @ beta + offset, -20, 20)
        mu = np.exp(eta)
        z = eta - offset + (y - mu) / np.maximum(mu, 1e-9)
        new = _solve(x, mu, z)
        if np.max(np.abs(new - beta)) < 1e-8:
            return new
        beta = new
    return beta


def fit_binomial(successes: np.ndarray, trials: np.ndarray, x: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Ridge logistic regression of successes out of trials (IRLS); rows with no trials carry no weight."""
    share = successes.sum() / max(trials.sum(), 1)
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(max(share, 1e-6) / max(1 - share, 1e-6))
    for _ in range(iterations):
        eta = np.clip(x @ beta, -20, 20)
        p = 1 / (1 + np.exp(-eta))
        w = trials * p * (1 - p)
        z = eta + (successes - trials * p) / np.maximum(w, 1e-9)
        new = _solve(x, w, np.where(w > 0, z, eta))
        if np.max(np.abs(new - beta)) < 1e-8:
            return new
        beta = new
    return beta


def dispersion(observed: np.ndarray, expected: np.ndarray, variance: np.ndarray, parameters: int) -> float:
    """Pearson dispersion, at least 1 (counts never vary less than the model allows)."""
    keep = variance > 0
    statistic = np.sum((observed[keep] - expected[keep]) ** 2 / variance[keep])
    return max(1.0, statistic / max(keep.sum() - parameters, 1))


def robust_sd(values: np.ndarray) -> float:
    """Standard deviation from the median absolute deviation (a few outliers do not move it)."""
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def weekly_inflation(standardized: np.ndarray) -> float:
    """
    How much more the 7-day sums of the standardized training residuals spread than independent days would:
    the robust sd of the 7-day sums over sqrt(7), divided by the robust sd of single days, at least 1. Days that
    run together (a month that sells a little more than its history, a strong campaign) make it larger.
    """
    if len(standardized) < 28:
        return 1.0
    weekly = np.convolve(standardized, np.ones(7), "valid") / np.sqrt(7)
    daily = robust_sd(standardized)
    return max(1.0, robust_sd(weekly) / daily) if daily > 0 else 1.0


@dataclass
class Expected:
    """Expected value, dispersion and weekly inflation of a series for every day of the detection period."""

    expected: pd.Series
    dispersion: pd.Series
    inflation: pd.Series


LEVEL_WEEKS = 8      # weeks of the recent level
LEVEL_GAP = 8        # days between the level window and the day judged: an anomaly shorter than this is not absorbed
LEVEL_PRIOR_SD = 0.1  # how far the level is expected to drift (10%); a noisy small series stays near no correction


def level_correction(observed: np.ndarray, fitted: np.ndarray, day_index: np.ndarray, phi: float) -> np.ndarray:
    """
    Recent level of a series against its model: for each judged day, the median of the 8 weekly ratios
    observed / fitted in the weeks that end LEVEL_GAP days before it, shrunk towards 1 by how noisy the weeks are.
    Prices and the product mix drift (most products change their list price at some point), and a model fitted on
    two years cannot see that; the median of whole weeks follows the drift and ignores a week-long anomaly.
    """
    observed_sum = np.concatenate([[0.0], np.cumsum(np.maximum(observed, 0.0))])
    fitted_sum = np.concatenate([[0.0], np.cumsum(fitted)])
    ends = day_index[:, None] - LEVEL_GAP - 7 * np.arange(LEVEL_WEEKS)[None, :]   # last day of each week
    begins = ends - 6
    valid = begins >= 0
    safe_end, safe_begin = np.where(valid, ends, 0), np.where(valid, begins, 0)
    expected = fitted_sum[safe_end + 1] - fitted_sum[safe_begin]
    actual = observed_sum[safe_end + 1] - observed_sum[safe_begin]
    valid &= expected > 0
    ratio = np.log(np.clip(np.where(valid, actual / np.where(valid, expected, 1.0), 1.0), 0.2, 5.0))
    count = valid.sum(axis=1)
    with np.errstate(all="ignore"):
        log_ratio = np.nanmedian(np.where(valid, ratio, np.nan), axis=1)
        size = np.nanmean(np.where(valid, expected, np.nan), axis=1)
    noise = (np.pi / 2) * phi / np.maximum(np.nan_to_num(size), 1e-9) / np.maximum(count, 1)
    weight = LEVEL_PRIOR_SD ** 2 / (LEVEL_PRIOR_SD ** 2 + noise)
    correction = np.exp(np.clip(weight * np.nan_to_num(log_ratio), np.log(0.67), np.log(1.5)))
    return np.where(count >= 4, correction, 1.0)


def month_starts(days: pd.DatetimeIndex, first: pd.Timestamp) -> list[pd.Timestamp]:
    return [m for m in pd.date_range(first, days[-1], freq="MS")]


def online_poisson(y: pd.Series, x: np.ndarray, first: pd.Timestamp, offset: np.ndarray | None = None,
                   offset_future: np.ndarray | None = None, level: bool = True) -> Expected:
    """
    For every month from first on: fit on the TRAINING_DAYS before it, predict the month. `offset` is an exposure
    known in the past (for a cell: its category's actual lines); `offset_future` replaces it in the prediction (the
    category's expected lines), so the prediction never uses the day it judges.
    """
    days = y.index
    offset = np.zeros(len(days)) if offset is None else np.log(np.maximum(offset, 0.5))
    future = offset if offset_future is None else np.log(np.maximum(offset_future, 1e-9))
    values = y.to_numpy(dtype=float)
    fitted_values = np.maximum(values, 0.0)  # a day of a cell can lose money; the fit needs y >= 0
    expected = pd.Series(np.nan, index=days)
    spread = pd.Series(np.nan, index=days)
    inflation = pd.Series(np.nan, index=days)
    for start in month_starts(days, first):
        train = (days < start) & (days >= start - pd.Timedelta(days=TRAINING_DAYS))
        test = (days >= start) & (days < start + pd.offsets.MonthBegin(1))
        xt, yt, ot = x[train], fitted_values[train], offset[train]
        beta = fit_poisson(yt, xt, ot)
        mu = np.exp(np.clip(xt @ beta + ot, -20, 20))
        residual = (values[train] - mu) / np.sqrt(np.maximum(mu, 1e-9))
        scale = np.sqrt(dispersion(values[train], mu, mu, x.shape[1]))
        keep = np.abs(residual) <= OUTLIER_RESIDUAL * scale
        if keep.sum() > x.shape[1] and keep.mean() > 0.9:
            beta = fit_poisson(yt[keep], xt[keep], ot[keep])
            mu = np.exp(np.clip(xt @ beta + ot, -20, 20))
        # The spread counts every training day: sparse series (returns of a small category) swing on a few
        # expensive days, and cutting them would shrink the band and flood the shop with alerts. An outage in the
        # history widens the band a little, which errs on the side of fewer false alarms.
        phi = dispersion(values[train], mu, mu, x.shape[1])
        # The recent level uses the actual exposure of past days (never the day judged, see LEVEL_GAP).
        fitted_all = np.exp(np.clip(x @ beta + offset, -20, 20))
        test_index = np.flatnonzero(test)
        recent = level_correction(values, fitted_all, test_index, phi) if level else 1.0
        expected[test] = np.exp(np.clip(x[test] @ beta + future[test], -20, 20)) * recent
        spread[test] = phi
        inflation[test] = weekly_inflation((values[train] - mu) / np.sqrt(np.maximum(phi * mu, 1e-9)))
    return Expected(expected, spread, inflation)


def online_binomial(successes: pd.Series, trials: pd.Series, x: np.ndarray, first: pd.Timestamp) -> Expected:
    """Expected share (conversion) for every day from first on, with the dispersion of the fit."""
    days = successes.index
    s, n = successes.to_numpy(dtype=float), trials.to_numpy(dtype=float)
    expected = pd.Series(np.nan, index=days)
    spread = pd.Series(np.nan, index=days)
    inflation = pd.Series(np.nan, index=days)
    for start in month_starts(days, first):
        train = (days < start) & (days >= start - pd.Timedelta(days=TRAINING_DAYS))
        test = (days >= start) & (days < start + pd.offsets.MonthBegin(1))
        xt, st, nt = x[train], s[train], n[train]
        beta = fit_binomial(st, nt, xt)
        p = 1 / (1 + np.exp(-(xt @ beta)))
        variance = nt * p * (1 - p)
        residual = (st - nt * p) / np.sqrt(np.maximum(variance, 1e-9))
        scale = np.sqrt(dispersion(st, nt * p, variance, x.shape[1]))
        keep = np.abs(residual) <= OUTLIER_RESIDUAL * scale
        if keep.mean() > 0.9:
            beta = fit_binomial(st[keep], nt[keep], xt[keep])
            p = 1 / (1 + np.exp(-(xt @ beta)))
            variance = nt * p * (1 - p)
        phi = dispersion(st, nt * p, variance, x.shape[1])
        expected[test] = 1 / (1 + np.exp(-(x[test] @ beta)))
        spread[test] = phi
        inflation[test] = weekly_inflation((st - nt * p) / np.sqrt(np.maximum(phi * variance, 1e-9)))
    return Expected(expected, spread, inflation)


def reference(sums: pd.Series, counts: pd.Series, weeks: int = 8, gap: int = 8) -> pd.Series:
    """
    Typical value per unit (per order line, per return) on every day: the median of the weekly means over the
    `weeks` weeks that end `gap` days before the day, so a week-long anomaly does not move its own reference.
    """
    weekly_sum = sums.rolling(7, min_periods=7).sum()
    weekly_n = counts.rolling(7, min_periods=7).sum()
    means = []
    for week in range(weeks):
        lag = gap + 7 * week
        n = weekly_n.shift(lag)
        means.append((weekly_sum.shift(lag) / n).where(n > 0))
    return pd.concat(means, axis=1).median(axis=1)
