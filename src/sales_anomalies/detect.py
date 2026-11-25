"""
The daily check: for every segment and metric, how far the day (and the last 7 days) is from what was expected.

Metrics, in PLN without VAT:
    sales       sales of the day (list price after discount)
    margin      contribution margin of the lines sold that day (after discount, purchase cost, shipping, payment fee
                and commission); returns are left out: in clothing a quarter of the lines come back and the cost of
                the returns booked on a day would bury a margin problem in their noise
    returns     cost of the returns booked that day (shop, channels and categories)
    costs       cost of serving the day's orders: shipping, payment fees and affiliate commission (shop and
                channels); small next to the margin, so a leak here hides in the margin but shows on its own
    conversion  orders / sessions (shop and channels; sessions carry no category); its impact is the lost or extra
                sales: (orders - expected orders) x the typical order value

The expected value comes from baseline.py; the z-score is (observed - expected) / sqrt(variance) with the
quasi-Poisson variance (dispersion x expected) for amounts and the binomial variance for conversion. Amounts are
scored on their clipped versions (facts.py) and reported in real PLN. The 7-day
score adds the last 7 days, so a smaller change that lasts a week is caught too. A flag needs both: a z-score past
the threshold of its group (level and density, metric, direction, scale; chosen on another seed, see
calibrate.py) and an
impact past the materiality in PLN, so tiny segments do not raise alerts about a few złoty.

Flagged days become episodes (consecutive days of one segment, metric and direction). Then the most specific
segment that carries the change is kept: a category whose single cell carries 70% of its change gives way to that
cell; a shop-wide drop that no single part explains keeps the shop and absorbs the alerts of its parts. Episodes of
one segment that overlap in time form one anomaly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sales_anomalies import baseline as b
from sales_anomalies.facts import ALL, DETECTION_START, Segment, ShopFacts

METRIC_LEVELS = {
    "sales": ("shop", "channel", "category", "cell"),
    "margin": ("shop", "channel", "category", "cell"),
    "returns": ("shop", "channel", "category"),
    "costs": ("shop", "channel"),
    "conversion": ("shop", "channel"),
}
SCALES = (1, 7)        # a 3-day scale was tried: more groups, more false alarms, no more found
# Smallest change worth a manager's attention, in PLN, for one day and for seven.
MATERIALITY = {"sales": (2000, 5000), "margin": (1000, 2500), "returns": (1000, 2500), "costs": (500, 1500),
               "conversion": (2000, 5000)}
DEFAULT_THRESHOLD = 4.0
CHILD_SHARE = 0.7          # a child carrying this share of its parent's change replaces the parent
CORE_Z = 1.0               # days of an episode that move with it (|daily z| at least this, same sign)
THRESHOLDS_FILE = Path(__file__).resolve().parent / "thresholds.json"


DENSE_LINES = 10          # a category or cell with at least this many order lines a day (training years) is dense


def group_key(group: str, metric: str, direction: str, scale: int) -> str:
    return f"{group}|{metric}|{direction}|{scale}"


def threshold_group(facts: ShopFacts, segment: Segment) -> str:
    """
    The shop and channels form their own groups; categories and cells are split into dense and sparse ones, so the
    few heavy-tailed small segments do not set the threshold for the busy ones.
    """
    if segment.level in ("shop", "channel"):
        return segment.level
    lines = facts.series(segment).loc[:DETECTION_START - pd.Timedelta(days=1), "lines"].mean()
    return f"{segment.level}-{'dense' if lines >= DENSE_LINES else 'sparse'}"


def load_thresholds(path: Path = THRESHOLDS_FILE) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))["thresholds"] if path.exists() else {}


# ---------------------------------------------------------------------------------------------------------------
# Expected values of every segment


def campaign_matrix(facts: ShopFacts) -> pd.DataFrame:
    """
    Per channel with campaigns: running or not (0/1). The budget per day was tried as a measure of strength: it
    explains little of a campaign's lift and a budget outside the history pushed a forecast to three times the
    real sales, so it stays out.
    """
    running = facts.campaigns.pivot(index="date", columns="channel", values="campaigns").reindex(facts.days,
                                                                                               fill_value=0)
    return running[[c for c in running.columns if running[c].sum() > 0]]


def _campaigns_for(segment: Segment, running: pd.DataFrame) -> pd.DataFrame | None:
    if segment.channel == ALL:
        return running
    return running[[segment.channel]] if segment.channel in running.columns else None


def _lagged_sales(series: pd.DataFrame) -> np.ndarray:
    """Log sales of the weeks before (days 1-7, 8-21, 22-49 back): the orders that returns come from."""
    columns = []
    for near, far in ((1, 7), (8, 21), (22, 49)):
        total = series["sales"].rolling(far - near + 1, min_periods=1).sum().shift(near).fillna(0)
        values = np.log1p(total.to_numpy())
        columns.append(values - values.mean())
    return np.column_stack(columns)


def _mixed_conversion(facts: ShopFacts, result: dict[str, pd.DataFrame],
                      shop: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    The shop mixes channels that convert differently: its expected orders are the sessions of each channel times
    that channel's expected conversion, so a campaign that brings many visitors who rarely buy is not a checkout
    problem. The dispersion is the one that gives the summed channel variance.
    """
    orders = pd.Series(0.0, index=facts.days)
    variance = pd.Series(0.0, index=facts.days)
    for channel in facts.channels:
        segment = Segment(channel=channel)
        sessions = facts.series(segment)["sessions"]
        p, phi = result[segment.key]["conversion"], result[segment.key]["phi_conversion"]
        orders += sessions * p
        variance += phi * sessions * p * (1 - p)
    sessions = shop["sessions"].where(shop["sessions"] > 0)
    p = orders / sessions
    phi = (variance / (sessions * p * (1 - p))).fillna(1.0)
    return p.fillna(0.0), phi


AMOUNTS = ("sales", "margin_sold", "return_cost")
MODELLED = ("lines", *AMOUNTS, *(f"{name}_w" for name in AMOUNTS))


def expectations(facts: ShopFacts, first: pd.Timestamp = DETECTION_START) -> dict[str, pd.DataFrame]:
    """
    Expected daily values of every segment (key -> frame): lines, sales, margin_sold, return_cost and their clipped
    versions (_w), and for the shop and channels costs, sessions and conversion; with dispersions (phi_<name>).
    """
    days = facts.days
    running = campaign_matrix(facts)
    result: dict[str, pd.DataFrame] = {}
    category_series: dict[str, pd.DataFrame] = {}
    # Channels before the shop (its conversion mixes theirs), categories before their cells.
    order = sorted(facts.segments(), key=lambda s: ("channel", "shop", "category", "cell").index(s.level))
    for segment in order:
        series = facts.series(segment)
        frame = pd.DataFrame(index=days)
        if segment.level == "cell":
            parent = Segment(category=segment.category)
            parent_series, parent_expected = category_series[segment.category], result[parent.key]
            x = b.design(days, _campaigns_for(segment, running), calendar=False)
            for name in MODELLED:
                fit = b.online_poisson(series[name], x, first, offset=parent_series[name].to_numpy(),
                                       offset_future=parent_expected[name].to_numpy())
                frame[name], frame[f"phi_{name}"], frame[f"k7_{name}"] = fit.expected, fit.dispersion, fit.inflation
        else:
            x = b.design(days, _campaigns_for(segment, running))
            # Returns follow the sales of the weeks before (a distributed lag of sales).
            with_recent = np.column_stack([x, _lagged_sales(series)])
            for name in MODELLED:
                fit = b.online_poisson(series[name], with_recent if name.startswith("return") else x, first)
                frame[name], frame[f"phi_{name}"], frame[f"k7_{name}"] = fit.expected, fit.dispersion, fit.inflation
            if segment.level in ("shop", "channel"):
                for name in ("costs", "sessions"):
                    fit = b.online_poisson(series[name], x, first)
                    frame[name], frame[f"phi_{name}"], frame[f"k7_{name}"] = fit.expected, fit.dispersion, fit.inflation
                if segment.level == "channel":
                    fit = b.online_binomial(series["orders"], series["sessions"], x, first)
                    frame["conversion"], frame["phi_conversion"] = fit.expected, fit.dispersion
                    frame["k7_conversion"] = fit.inflation
                else:
                    frame["conversion"], frame["phi_conversion"] = _mixed_conversion(facts, result, series)
                    inflations = [result[Segment(channel=c).key]["k7_conversion"] for c in facts.channels]
                    frame["k7_conversion"] = pd.concat(inflations, axis=1).median(axis=1)
            if segment.level == "category":
                category_series[segment.category] = series
        result[segment.key] = frame
    return result


# ---------------------------------------------------------------------------------------------------------------
# Scores


def _amount(series: pd.DataFrame, expected: pd.DataFrame, name: str, suffix: str) -> tuple[pd.Series, ...]:
    column = f"{name}{suffix}"
    return series[column], expected[column], expected[f"phi_{column}"] * expected[column]


def _observed_expected(metric: str, series: pd.DataFrame, expected: pd.DataFrame,
                       suffix: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Observed, expected and variance of a metric; suffix '_w' gives the clipped amounts used by the z-scores."""
    if metric == "sales":
        return _amount(series, expected, "sales", suffix)
    if metric == "returns":
        return _amount(series, expected, "return_cost", suffix)
    if metric == "margin":
        return _amount(series, expected, "margin_sold", suffix)
    if metric == "costs":
        return series["costs"], expected["costs"], expected["phi_costs"] * expected["costs"]
    p = expected["conversion"]  # conversion, counted in orders
    return series["orders"], series["sessions"] * p, expected["phi_conversion"] * series["sessions"] * p * (1 - p)


# The single item counted as a typical one before testing a direction: one order or one return is not an anomaly.
# Jumps are tested on real amounts, drops on clipped ones (see facts.py).
ROBUST_ITEMS = {
    ("sales", "up"): ("top_order_sales",),
    ("margin", "up"): ("top_order_margin",),
    ("margin", "down"): ("-bottom_order_margin_w",),   # the order that lost the most
    ("returns", "up"): ("top_return_cost",),
}


def _typical(series: pd.DataFrame, column: str) -> pd.Series:
    return b.reference(series[column], pd.Series(1.0, index=series.index)).fillna(0)


def _robust_shift(metric: str, direction: str, series: pd.DataFrame) -> pd.Series:
    """How much the day's largest items exceed typical ones (0 when the metric has none)."""
    shift = pd.Series(0.0, index=series.index)
    for item in ROBUST_ITEMS.get((metric, direction), ()):
        sign, column = (-1, item[1:]) if item.startswith("-") else (1, item)
        shift = shift + sign * (series[column] - _typical(series, column))
    return shift


# The series whose weekly inflation widens a metric's 7-day band (margin: the margin of the lines sold).
INFLATION_OF = {"sales": "sales", "margin": "margin_sold", "returns": "return_cost", "costs": "costs",
                "conversion": "conversion"}


def _scores(delta: pd.Series, variance: pd.Series, inflation: pd.Series) -> dict[int, pd.Series]:
    """z of the day and of the last days of each longer scale. Days run together (see baseline.weekly_inflation),
    so a sum of days spreads more than independent days."""
    result = {1: delta / np.sqrt(variance.where(variance > 0))}
    for scale in SCALES[1:]:
        spread = np.sqrt(variance.rolling(scale, min_periods=scale).sum().where(lambda v: v > 0)) * inflation
        result[scale] = delta.rolling(scale, min_periods=scale).sum() / spread
    return result


def _metric_frame(segment: Segment, metric: str, series: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    observed, mean, variance = _observed_expected(metric, series, expected, "")
    clipped = "_w" if metric in ("sales", "margin", "returns") else ""
    observed_w, mean_w, variance_w = _observed_expected(metric, series, expected, clipped)
    # Impact in PLN; a conversion change costs the typical sales of an order per order.
    value = b.reference(series["sales"], series["orders"]).fillna(0) if metric == "conversion" else 1.0
    frame = pd.DataFrame({"observed": observed, "expected": mean, "variance": variance})
    frame["delta"] = observed - mean
    frame["impact"] = frame["delta"] * value
    for scale in SCALES[1:]:
        frame[f"impact{scale}"] = frame["impact"].rolling(scale, min_periods=scale).sum()
    frame["delta_w"], frame["variance_w"] = observed_w - mean_w, variance_w
    name = INFLATION_OF[metric]
    frame["z"] = _scores(frame["delta"], variance, expected[f"k7_{name}"])[1]
    # Each direction keeps its own robust daily change, spread and weekly inflation, so an episode of any length
    # can be scored the same way (see episodes()).
    frame["d_up"] = frame["delta"] - _robust_shift(metric, "up", series)
    frame["var_up"], frame["k7_up"] = variance, expected[f"k7_{name}"]
    frame["d_down"] = frame["delta_w"] + _robust_shift(metric, "down", series)
    frame["var_down"] = variance_w
    frame["k7_down"] = expected[f"k7_{name}{clipped}"] if clipped else expected[f"k7_{name}"]
    for direction in ("up", "down"):
        for scale, z in _scores(frame[f"d_{direction}"], frame[f"var_{direction}"], frame[f"k7_{direction}"]).items():
            frame[score_column(direction, scale)] = z
    frame.insert(0, "metric", metric)
    frame.insert(0, "level", segment.level)
    frame.insert(0, "segment", segment.key)
    return frame


def score_column(direction: str, scale: int) -> str:
    return f"z{'' if scale == 1 else scale}_{direction}"


def impact_column(scale: int) -> str:
    return "impact" if scale == 1 else f"impact{scale}"


def run_scale(days: int) -> int:
    """The scale whose threshold a run of days must pass: the one-day threshold or the weekly one."""
    return 1 if days == 1 else SCALES[-1]


def scores(facts: ShopFacts, expected: dict[str, pd.DataFrame],
           first: pd.Timestamp = DETECTION_START) -> pd.DataFrame:
    """One row per day, segment and metric: observed, expected, variance, impact in PLN, daily and 7-day z."""
    frames = []
    for segment in facts.segments():
        series = facts.series(segment)
        group = threshold_group(facts, segment)
        for metric, levels in METRIC_LEVELS.items():
            if segment.level in levels:
                frame = _metric_frame(segment, metric, series, expected[segment.key])
                frame.insert(3, "group", group)
                frames.append(frame)
    table = pd.concat(frames)
    table.index.name = "date"
    table = table.reset_index()
    return table[table["date"] >= first].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
# Flags, episodes, the most specific segment, anomalies


def flags(table: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Flagged windows: one row per flag with segment, metric, direction, scale, window start and end, z."""
    rows = []
    for scale in SCALES:
        impact = table[impact_column(scale)]
        for direction, sign in (("up", 1), ("down", -1)):
            z = table[score_column(direction, scale)]
            limit = np.array([thresholds.get(group_key(group, metric, direction, scale), DEFAULT_THRESHOLD)
                              for group, metric in zip(table["group"], table["metric"], strict=True)])
            materiality = table["metric"].map(lambda m, s=scale: MATERIALITY[m][SCALES.index(s)]).to_numpy()
            hit = (sign * z >= limit) & (sign * impact >= materiality)
            flagged = table.loc[hit, ["date", "segment", "level", "metric"]].copy()
            flagged["direction"], flagged["scale"] = direction, scale
            flagged["z"] = z[hit].to_numpy()
            flagged["start"] = flagged["date"] - pd.Timedelta(days=scale - 1)
            flagged["end"] = flagged["date"]
            rows.append(flagged)
    return pd.concat(rows, ignore_index=True)


@dataclass
class Episode:
    segment: str
    level: str
    metric: str
    direction: str
    start: pd.Timestamp
    end: pd.Timestamp
    z_peak: float
    scales: list[int] = field(default_factory=list)
    observed: float = 0.0
    expected: float = 0.0
    impact: float = 0.0
    z: float = 0.0
    kept: bool = True
    absorbed_by: int | None = None


def _merge(windows: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, float, list[int]]]:
    windows = windows.sort_values("start")
    merged: list[list] = []
    for row in windows.itertuples():
        if merged and row.start <= merged[-1][1] + pd.Timedelta(days=1):
            merged[-1][1] = max(merged[-1][1], row.end)
            merged[-1][2] = max(merged[-1][2], abs(row.z))
            merged[-1][3].add(row.scale)
        else:
            merged.append([row.start, row.end, abs(row.z), {row.scale}])
    return [(start, end, peak, sorted(scales)) for start, end, peak, scales in merged]


def _runs(days: pd.DatetimeIndex, gap: int = 1) -> list[pd.DatetimeIndex]:
    """Split sorted days into runs; a pause of more than `gap` days starts a new run."""
    runs, current = [], [days[0]]
    for day in days[1:]:
        if (day - current[-1]).days > gap + 1:
            runs.append(pd.DatetimeIndex(current))
            current = [day]
        else:
            current.append(day)
    runs.append(pd.DatetimeIndex(current))
    return runs


def run_score(span: pd.DataFrame, direction: str) -> float:
    """
    z of a run of days: its summed robust change over its spread, widened by the weekly inflation if longer than a
    day, and for a run longer than a week also by sqrt(days / 7): over weeks the level itself wanders (a month
    that sells a few percent above its history), so a long sum spreads more than a weekly one.
    """
    spread = np.sqrt(span[f"var_{direction}"].sum())
    if len(span) > 1:
        spread *= float(span[f"k7_{direction}"].mean()) * np.sqrt(max(len(span) / 7, 1.0))
    return float(span[f"d_{direction}"].sum() / spread) if spread > 0 else 0.0


def episodes(table: pd.DataFrame, flagged: pd.DataFrame, thresholds: dict[str, float]) -> list[Episode]:
    """
    Merge the flagged windows of a segment, metric and direction; split them into runs of the days that move with
    them (|daily z| at least CORE_Z in the direction, a pause of at most one day); keep a run only if it passes on
    its own: its z the threshold of its scale (one day or longer) and its impact the materiality.
    """
    by_key = {key: frame.set_index("date") for key, frame in table.groupby(["segment", "metric"])}
    result = []
    groups = flagged.groupby(["segment", "level", "metric", "direction"])
    for (segment, level, metric, direction), windows in groups:
        sign = 1 if direction == "up" else -1
        days = by_key[(segment, metric)]
        group = days["group"].iloc[0]
        for start, end, peak, scales in _merge(windows):
            span = days.loc[start:end]
            core = span.index[sign * span[f"z_{direction}"] >= CORE_Z]
            for run in _runs(core) if len(core) else []:
                piece = days.loc[run.min():run.max()]
                scale = run_scale(len(piece))
                z = run_score(piece, direction)
                impact = float(piece["impact"].sum())
                limit = thresholds.get(group_key(group, metric, direction, scale), DEFAULT_THRESHOLD)
                if sign * z < limit or sign * impact < MATERIALITY[metric][SCALES.index(scale)]:
                    continue
                episode = Episode(segment, level, metric, direction, run.min(), run.max(), float(peak), scales)
                episode.observed = float(piece["observed"].sum())
                episode.expected = float(piece["expected"].sum())
                episode.impact, episode.z = impact, z
                result.append(episode)
    return result


def _overlap(a: Episode, b_: Episode) -> bool:
    return a.start <= b_.end and b_.start <= a.end


def most_specific(found: list[Episode], table: pd.DataFrame) -> list[Episode]:
    """
    Keep the most specific segment that carries each change (see the module docstring); returns every episode with
    `kept` and `absorbed_by` set.
    """
    impact = table.set_index(["segment", "metric", "date"])["impact"].sort_index()
    order = sorted(range(len(found)), key=lambda i: ("shop", "channel", "category", "cell").index(found[i].level))
    for i in order:
        parent = found[i]
        if not parent.kept:
            continue
        inside = Segment.from_key(parent.segment)
        children = [j for j in range(len(found)) if j != i and found[j].metric == parent.metric
                    and found[j].direction == parent.direction and found[j].kept
                    and found[j].segment != parent.segment
                    and inside.contains(Segment.from_key(found[j].segment)) and _overlap(parent, found[j])]
        if not children:
            continue
        carried = {j: float(impact.loc[(found[j].segment, parent.metric)].loc[parent.start:parent.end].sum())
                   for j in children}
        best = max(carried, key=lambda j: abs(carried[j]))
        if abs(parent.impact) > 0 and carried[best] / parent.impact >= CHILD_SHARE:
            parent.kept = False
            parent.absorbed_by = best
        else:
            for j in children:
                found[j].kept = False
                found[j].absorbed_by = i
    return found


@dataclass
class Anomaly:
    """Episodes of one segment that overlap in time; the main metric has the strongest 7-day or episode z."""

    anomaly_id: int
    segment: str
    level: str
    start: pd.Timestamp
    end: pd.Timestamp
    episodes: list[Episode]

    @property
    def main(self) -> Episode:
        return max(self.episodes, key=lambda e: abs(e.z))

    @property
    def metrics(self) -> list[str]:
        return sorted({e.metric for e in self.episodes})


def anomalies(found: list[Episode]) -> list[Anomaly]:
    kept = sorted((e for e in found if e.kept), key=lambda e: (e.segment, e.start))
    groups: list[list[Episode]] = []
    for episode in kept:
        if groups and groups[-1][0].segment == episode.segment and episode.start <= max(
                e.end for e in groups[-1]):
            groups[-1].append(episode)
        else:
            groups.append([episode])
    result = [Anomaly(0, g[0].segment, g[0].level, min(e.start for e in g), max(e.end for e in g), g)
              for g in groups]
    result.sort(key=lambda a: (a.start, a.segment))
    for number, anomaly in enumerate(result, start=1):
        anomaly.anomaly_id = number
    return result


@dataclass
class Detection:
    table: pd.DataFrame
    expected: dict[str, pd.DataFrame]
    flagged: pd.DataFrame
    episodes: list[Episode]
    anomalies: list[Anomaly]


def detect(facts: ShopFacts, thresholds: dict[str, float] | None = None,
           first: pd.Timestamp = DETECTION_START) -> Detection:
    thresholds = load_thresholds() if thresholds is None else thresholds
    expected = expectations(facts, first)
    table = scores(facts, expected, first)
    flagged = flags(table, thresholds)
    found = most_specific(episodes(table, flagged, thresholds), table)
    return Detection(table, expected, flagged, found, anomalies(found))
