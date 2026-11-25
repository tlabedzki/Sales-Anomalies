"""
Daily facts of every segment: the shop, a channel, a category, and a category in one channel (a cell).

Cells are summed up to the other levels, so a channel or the shop is exactly the sum of its cells. Sessions carry a
channel but no category: conversion (orders / sessions) exists for the shop and for each channel only.

Next to every amount there is a winsorized one (suffix _w): each order line or return clipped to the 1st and 99th
percentile of its category in the training years before DETECTION_START. A few very expensive lines make the daily
sales of a channel swing; clipped, the same day is steadier, so the check sees a real drop sooner. Drops are tested
on the clipped amounts (a drop is ordinary orders missing), jumps on the real ones (a fraud ring is a few very
expensive orders, which clipping would hide). Every amount the check reports (impact, causes) is in real PLN.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sales_anomalies.data import ShopData
from sales_anomalies.economics import LINE_VALUES, booked_returns, sold_lines

ALL = "*"
LEVELS = ("shop", "channel", "category", "cell")
DETECTION_START = pd.Timestamp("2023-01-01")
CLIPPED = ("sales_w", "margin_w", "return_cost_w")
FACT_VALUES = (*LINE_VALUES, "return_cost", *CLIPPED)
COUNT_COLUMNS = ("lines", "returns")
TOP_COLUMNS = ("top_order_sales", "top_order_margin", "top_return_cost", "bottom_order_margin_w",
               "top_return_cost_w")
CLIP_QUANTILES = (0.01, 0.99)
METRICS = ("sales", "margin", "returns", "costs", "conversion")


@dataclass(frozen=True)
class Segment:
    """A part of the shop: category and channel, each a value or ALL."""

    category: str = ALL
    channel: str = ALL

    @property
    def level(self) -> str:
        if self.category == ALL:
            return "shop" if self.channel == ALL else "channel"
        return "category" if self.channel == ALL else "cell"

    @property
    def key(self) -> str:
        return f"{self.category}|{self.channel}"

    @classmethod
    def from_key(cls, key: str) -> Segment:
        category, channel = key.split("|")
        return cls(category, channel)

    def contains(self, other: Segment) -> bool:
        """True when other is this segment or lies inside it."""
        return self.category in (ALL, other.category) and self.channel in (ALL, other.channel)

    def label(self) -> str:
        if self.level == "shop":
            return "whole shop"
        if self.level == "channel":
            return f"{self.channel} channel"
        if self.level == "category":
            return self.category
        return f"{self.category} · {self.channel}"


@dataclass
class ShopFacts:
    """Line and return tables plus the daily facts built from them."""

    lines: pd.DataFrame
    returns: pd.DataFrame
    cells: pd.DataFrame        # one row per day x category x channel (all combinations, zeros filled)
    tops: pd.DataFrame         # largest single order and return of every day and segment (clipped amounts)
    traffic: pd.DataFrame      # one row per day x channel x device: sessions, orders, checkouts
    campaigns: pd.DataFrame    # one row per day x channel: campaigns running
    days: pd.DatetimeIndex
    _series_cache: dict = field(default_factory=dict, repr=False)

    @property
    def categories(self) -> list[str]:
        return sorted(self.cells["category"].unique())

    @property
    def channels(self) -> list[str]:
        return sorted(self.cells["channel"].unique())

    def segments(self) -> list[Segment]:
        segments = [Segment()]
        segments += [Segment(channel=c) for c in self.channels]
        segments += [Segment(category=c) for c in self.categories]
        segments += [Segment(c, ch) for c in self.categories for ch in self.channels]
        return segments

    def series(self, segment: Segment) -> pd.DataFrame:
        """Daily facts of one segment, indexed by day (sums of its cells; traffic for the shop and channels)."""
        cached = self._series_cache.get(segment.key)
        if cached is not None:
            return cached.copy()
        mask = np.ones(len(self.cells), dtype=bool)
        if segment.category != ALL:
            mask &= self.cells["category"].to_numpy() == segment.category
        if segment.channel != ALL:
            mask &= self.cells["channel"].to_numpy() == segment.channel
        columns = [*COUNT_COLUMNS, *FACT_VALUES]
        frame = self.cells.loc[mask].groupby("date")[columns].sum().reindex(self.days, fill_value=0)
        if segment.category == ALL:
            traffic = self.traffic
            if segment.channel != ALL:
                traffic = traffic[traffic["channel"] == segment.channel]
            frame = frame.join(traffic.groupby("date")[["sessions", "orders"]].sum().reindex(self.days,
                                                                                               fill_value=0))
        tops = self.tops[(self.tops["category"] == segment.category) & (self.tops["channel"] == segment.channel)]
        frame = frame.join(tops.set_index("date")[list(TOP_COLUMNS)].reindex(self.days, fill_value=0))
        frame["costs"] = frame["shipping"] + frame["payment"] + frame["commission"]
        frame["margin_sold"] = frame["margin"]
        frame["margin"] = frame["margin"] - frame["return_cost"]
        frame["margin_sold_w"] = frame.pop("margin_w")
        frame.index.name = "date"
        self._series_cache[segment.key] = frame
        return frame.copy()


def campaign_calendar(campaigns: pd.DataFrame, days: pd.DatetimeIndex, channels: list[str]) -> pd.DataFrame:
    """Campaigns running per day and channel and their budget per day (campaign_data is planned in advance)."""
    rows = []
    for channel in channels:
        running = np.zeros(len(days), dtype=int)
        spend = np.zeros(len(days))
        for _, campaign in campaigns[campaigns["channel"] == channel].iterrows():
            on = (days >= campaign["start_date"]) & (days <= campaign["end_date"])
            running += on.astype(int)
            spend += on * campaign["total_cost"] / max(on.sum(), 1)
        rows.append(pd.DataFrame({"date": days, "channel": channel, "campaigns": running, "spend": spend}))
    return pd.concat(rows, ignore_index=True)


def clip(values: pd.Series, groups: pd.Series, training: pd.Series) -> pd.Series:
    """Clip every value to the CLIP_QUANTILES of its group, measured on the training rows only."""
    bounds = values[training].groupby(groups[training]).quantile(list(CLIP_QUANTILES)).unstack()
    low = groups.map(bounds[CLIP_QUANTILES[0]]).fillna(values.min())
    high = groups.map(bounds[CLIP_QUANTILES[1]]).fillna(values.max())
    return values.clip(lower=low, upper=high)


def largest_items(lines: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Per day and segment: the largest order of a customer who placed only that one order there that day (sales and
    margin, real PLN), the largest return (real and clipped) and the order that lost the most (clipped). In a small
    segment one order can be most of a day; detect.py counts it as a typical one before it tests a direction. A
    customer with several orders is left as it is: a stolen account buying again and again is the anomaly.
    """
    frames = []
    for by_category, by_channel in ((False, False), (False, True), (True, False), (True, True)):
        keys = ["date"] + (["category"] if by_category else []) + (["channel"] if by_channel else [])
        orders = lines.groupby([*keys, "order_id"]).agg(customer_id=("customer_id", "first"), sales=("sales", "sum"),
                                                         margin=("margin", "sum"), margin_w=("margin_w", "sum"))
        orders = orders.reset_index()
        repeat = orders.groupby([*keys, "customer_id"])["order_id"].transform("size") > 1
        single = orders[~repeat].groupby(keys)[["sales", "margin"]].max()
        single.columns = ["top_order_sales", "top_order_margin"]
        bottom = orders.groupby(keys)["margin_w"].min().rename("bottom_order_margin_w")
        back = returns.groupby(keys)[["return_cost", "return_cost_w"]].max()
        back.columns = ["top_return_cost", "top_return_cost_w"]
        frame = single.join(bottom, how="outer").join(back, how="outer").astype(float).fillna(0.0).reset_index()
        if not by_category:
            frame["category"] = ALL
        if not by_channel:
            frame["channel"] = ALL
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[["date", "category", "channel", *TOP_COLUMNS]]


def build(shop: ShopData) -> ShopFacts:
    """Line economics and daily facts of a shop."""
    lines = sold_lines(shop)
    returns = booked_returns(shop)
    start, end = shop.period
    days = pd.date_range(start, end, freq="D")
    categories = sorted(shop.products["category"].unique())
    channels = sorted(shop.orders["channel"].unique())

    training = lines["date"] < DETECTION_START
    lines["sales_w"] = clip(lines["sales"], lines["category"], training)
    lines["margin_w"] = clip(lines["margin"], lines["category"], training)
    returns["return_cost_w"] = clip(returns["return_cost"], returns["category"], returns["date"] < DETECTION_START)
    sold = lines.groupby(["date", "category", "channel"])[[*LINE_VALUES, "sales_w", "margin_w"]].sum()
    sold["lines"] = lines.groupby(["date", "category", "channel"]).size()
    booked = returns[returns["date"].between(start, end)].groupby(["date", "category", "channel"])
    back = pd.DataFrame({"returns": booked.size(), "return_cost": booked["return_cost"].sum(),
                         "return_cost_w": booked["return_cost_w"].sum()})
    grid = pd.MultiIndex.from_product([days, categories, channels], names=["date", "category", "channel"])
    cells = sold.join(back, how="outer").reindex(grid, fill_value=0).fillna(0).reset_index()
    cells[list(COUNT_COLUMNS)] = cells[list(COUNT_COLUMNS)].astype(int)

    sessions = shop.sessions.assign(date=shop.sessions["session_start"].dt.normalize())
    sessions = sessions[sessions["date"].between(start, end)]
    traffic = (sessions.groupby(["date", "channel", "device"])
               .agg(sessions=("session_id", "size"), orders=("order_id", "count"),
                    checkouts=("reached_checkout", "sum"))
               .reset_index())
    campaigns = campaign_calendar(shop.campaigns, days, channels)
    tops = largest_items(lines, returns[returns["date"].between(start, end)])
    return ShopFacts(lines=lines, returns=returns, cells=cells, tops=tops, traffic=traffic, campaigns=campaigns,
                     days=days)
