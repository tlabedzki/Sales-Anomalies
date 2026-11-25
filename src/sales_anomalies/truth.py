"""
The answer key: events that truly happened, in one shape for both sources.

    scenario      anomalies injected by scenarios.py, with a known cause and a known impact in PLN
    anomaly_data  events Data-Generator labelled itself: fraud rings (a stolen account places expensive orders for
                  2-3 days), shop outages (the shop is down for a day, most orders come back on the next two) and a
                  return spike (a defective batch of one product is returned 55% of the time)

Each event lists the most specific segments it touched (keys of facts.Segment), its days and the metrics it moves.
An alert matches an event when their days overlap and the alert's segment is one of the event's segments, lies
inside one or contains one (a shop-wide alert for a fraud ring in one cell still points at it).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from sales_anomalies.data import ShopData
from sales_anomalies.facts import Segment, ShopFacts

OUTAGE_RECOVERY_DAYS = 2  # Data-Generator moves the orders of an outage day to the next two days
CAUSE_OF_TYPE = {"fraud_ring": "single_account", "shop_outage": "outage", "return_spike": "returns"}


@dataclass
class TruthEvent:
    event_id: str
    source: str            # scenario or anomaly_data
    kind: str              # scenario type or anomaly_type
    cause: str             # cause label the decomposition should name (see decompose.CAUSES)
    start: pd.Timestamp
    end: pd.Timestamp
    segments: list[str]
    metrics: list[str]
    impact: float = float("nan")   # true change of the margin in PLN (nan when not known)
    description: str = ""
    details: dict = field(default_factory=dict)
    # What the event itself put into the data, per day and cell (columns date, category, channel, sales,
    # margin_sold, return_cost); None when it cannot be told apart (an outage moves the whole shop).
    contribution: pd.DataFrame | None = field(default=None, repr=False)

    def touches(self, segment: str) -> bool:
        alert = Segment.from_key(segment)
        return any(alert.contains(Segment.from_key(s)) or Segment.from_key(s).contains(alert)
                   for s in self.segments)


def dg_events(shop: ShopData, facts: ShopFacts) -> list[TruthEvent]:
    """The events of anomaly_data as truth events, with segments and windows read from the shop tables."""
    events = []
    lines = facts.lines
    for row in shop.anomalies.itertuples():
        start, end = pd.Timestamp(row.start_date), pd.Timestamp(row.end_date)
        event_id = f"dg-{row.anomaly_id}"
        if row.anomaly_type == "fraud_ring":
            ring = lines[lines["is_fraud"] & (lines["customer_id"] == row.customer_id)
                         & lines["date"].between(start, end)]
            if ring.empty:
                continue
            segments = sorted({Segment(c, ch).key for c, ch in zip(ring["category"], ring["channel"], strict=True)})
            contribution = ring.groupby(["date", "category", "channel"])[["sales", "margin"]].sum().reset_index()
            contribution = contribution.rename(columns={"margin": "margin_sold"}).assign(return_cost=0.0)
            events.append(TruthEvent(event_id, "anomaly_data", row.anomaly_type, CAUSE_OF_TYPE[row.anomaly_type],
                                     ring["date"].min(), ring["date"].max(), segments, ["sales", "margin"],
                                     float(ring["margin"].sum()), row.description,
                                     {"sales": float(ring["sales"].sum()), "orders": int(ring["order_id"].nunique()),
                                      "customer_id": int(row.customer_id)}, contribution))
        elif row.anomaly_type == "shop_outage":
            events.append(TruthEvent(event_id, "anomaly_data", row.anomaly_type, CAUSE_OF_TYPE[row.anomaly_type],
                                     start, end + pd.Timedelta(days=OUTAGE_RECOVERY_DAYS), [Segment().key],
                                     ["sales", "margin", "costs", "conversion"], description=row.description))
        elif row.anomaly_type == "return_spike":
            match = re.search(r"product (\d+)", row.description)
            if not match:
                continue
            product = int(match.group(1))
            ordered = lines[(lines["product_id"] == product) & lines["date"].between(start, end)]
            back = facts.returns[facts.returns["order_line_id"].isin(ordered["order_line_id"])]
            if back.empty:
                continue
            segments = sorted({Segment(c, ch).key for c, ch in zip(back["category"], back["channel"], strict=True)})
            contribution = back.groupby(["date", "category", "channel"])["return_cost"].sum().reset_index()
            contribution = contribution.assign(sales=0.0, margin_sold=0.0)
            events.append(TruthEvent(event_id, "anomaly_data", row.anomaly_type, CAUSE_OF_TYPE[row.anomaly_type],
                                     back["date"].min(), back["date"].max(), segments, ["returns"],
                                     -float(back["return_cost"].sum()), row.description,
                                     {"product_id": product, "returns": len(back)}, contribution))
    return events


def matches(event: TruthEvent, segment: str, start: pd.Timestamp, end: pd.Timestamp, tolerance: int = 1) -> bool:
    """Days overlap (with `tolerance` days of slack) and the segments are related."""
    slack = pd.Timedelta(days=tolerance)
    return start <= event.end + slack and event.start - slack <= end and event.touches(segment)
