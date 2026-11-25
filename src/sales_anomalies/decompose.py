"""
Why did it move? Every anomaly split into causes in PLN that add up to the change to the grosz.

The change is observed - expected over the anomaly's days, on its main metric. Standard bridges:

sales and margin (a price-volume bridge, as in a controller's margin walk):
    change = orders of one account (only when one customer with several orders made a large part of it)
           + volume          (lines - expected lines) x expected value per line
           + prices and mix  lines x change of (list price - purchase cost) per line
           + discount        - lines x change of discount per line
           + shipping, payment fees, commission   - lines x change of each cost per line (margin only)
    The expected value per line is the model's; its split into list price, discount and costs comes from the
    recent weeks (baseline.reference), the gap to the model goes to prices and mix.

volume, further split by LMDI (logarithmic mean Divisia index, Ang 2004), which adds up exactly:
    lines = sessions x conversion x lines per order (shop, channel) or x share of the channel's orders (category,
    cell); each factor gets volume x ln(actual / expected factor) / ln(actual / expected lines).

returns: (returns - expected returns) x cost per return, split by return reason, + change of the cost per return.
costs: shipping, payment fees and commission, each actual - expected (expected split by recent shares).
conversion: orders lost or gained per device (sessions x change of its conversion), in orders and in PLN.

Each part carries a cause from a fixed list (CAUSES). Volume gets one from evidence the code checks: the shop's
traffic collapsing is an outage, a channel with a campaign running is the campaign, a category whose competitor cut
prices is competitor prices. The main cause is the largest part that moves with the change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from sales_anomalies import baseline as b
from sales_anomalies.data import ShopData
from sales_anomalies.detect import Anomaly, Detection, Episode
from sales_anomalies.facts import ALL, Segment, ShopFacts

CAUSES = ("discount", "prices", "returns", "shipping", "payment_fees", "commission", "competitor_prices",
          "campaign", "conversion", "traffic", "demand", "outage", "single_account")
ACCOUNT_SHARE = 0.25        # one customer with at least this share of the change and two orders is named
OUTAGE_TRAFFIC = 0.3        # sessions below this share of the expected ones: the shop was down
COMPETITOR_DROP = 0.05      # competitor prices 5% lower against ours than in the weeks before
EPSILON = 1e-6


@dataclass
class Part:
    name: str            # e.g. volume_traffic, discount, returns_defective
    cause: str           # one of CAUSES
    amount: float        # PLN (orders for a conversion breakdown's own units are kept in `orders`)
    grosz: int = 0       # amount rounded so that the parts add up to the change exactly
    orders: float | None = None


@dataclass
class Breakdown:
    anomaly_id: int
    segment: str
    metric: str
    direction: str
    start: str
    end: str
    observed: float
    expected: float
    change_grosz: int
    parts: list[Part]
    main_cause: str
    signals: dict = field(default_factory=dict)

    @property
    def change(self) -> float:
        return self.change_grosz / 100

    def as_dict(self) -> dict:
        return {**asdict(self), "change": self.change}


# ---------------------------------------------------------------------------------------------------------------
# Rounding


def to_grosz(parts: list[Part], observed: float, expected: float) -> int:
    """
    Round the parts to grosze by the largest remainder method (Hamilton), so that they add up exactly to
    round(observed) - round(expected); returns that change in grosze.
    """
    total = int(round(observed * 100)) - int(round(expected * 100))
    raw = np.array([p.amount * 100 for p in parts], dtype=float)
    if len(parts) == 0:
        return total
    # Scale tiny float drift so the raw parts sum to the rounded total before rounding them.
    drift = total - raw.sum()
    raw = raw + drift * (np.abs(raw) / np.abs(raw).sum() if np.abs(raw).sum() > 0 else 1 / len(raw))
    floors = np.floor(raw).astype(np.int64)
    remainder = int(total - floors.sum())
    order = np.argsort(-(raw - floors), kind="stable")
    floors[order[:remainder]] += 1
    for part, value in zip(parts, floors, strict=True):
        part.grosz = int(value)
    return total


# ---------------------------------------------------------------------------------------------------------------
# Evidence


def _segment_mask(frame: pd.DataFrame, segment: Segment) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if segment.category != ALL:
        mask &= frame["category"] == segment.category
    if segment.channel != ALL:
        mask &= frame["channel"] == segment.channel
    return mask


def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, column: str = "date") -> pd.Series:
    return frame[column].between(start, end)


def competitor_signal(shop: ShopData, category: str, start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    """
    The competitor whose prices in the category fell most against ours: mean ratio competitor price / our list
    price over the anomaly's days against the 28 days before. Returned when the drop is at least COMPETITOR_DROP.
    """
    products = shop.products.loc[shop.products["category"] == category, ["product_id", "sales_price_gross"]]
    if products.empty:
        return None
    prices = shop.competitor_prices[shop.competitor_prices["product_id"].isin(products["product_id"])]
    ours = shop.price_history[shop.price_history["product_id"].isin(products["product_id"])]

    def level(day_from: pd.Timestamp, day_to: pd.Timestamp) -> pd.Series:
        middle = day_from + (day_to - day_from) / 2
        theirs = prices[(prices["valid_from"] <= middle) & (prices["valid_to"].fillna(pd.Timestamp.max) >= middle)]
        own = ours[(ours["valid_from"] <= middle) & (ours["valid_to"].fillna(pd.Timestamp.max) >= middle)]
        merged = theirs.merge(own[["product_id", "sales_price_gross"]], on="product_id")
        merged = merged[merged["sales_price_gross"] > 0]
        return (merged["price_gross"] / merged["sales_price_gross"]).groupby(merged["competitor_name"]).mean()

    now, before = level(start, end), level(start - pd.Timedelta(days=28), start - pd.Timedelta(days=1))
    change = (now / before - 1).dropna()
    if change.empty or change.min() > -COMPETITOR_DROP:
        return None
    name = str(change.idxmin())
    return {"competitor": name, "price_change": float(change.min()), "price_ratio": float(now[name])}


def running_campaigns(shop: ShopData, channel: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    c = shop.campaigns
    running = c[(c["channel"] == channel) & (c["start_date"] <= end) & (c["end_date"] >= start)]
    return sorted(running["campaign_name"].tolist())


# ---------------------------------------------------------------------------------------------------------------
# Bridges


@dataclass
class Window:
    segment: Segment
    start: pd.Timestamp
    end: pd.Timestamp
    series: pd.DataFrame        # the segment's daily facts over the window
    expected: pd.DataFrame      # the model's expected values over the window
    reference: pd.Series        # recent per-line (and per-return) values at the window's first day


def _window_data(facts: ShopFacts, detection: Detection, segment: Segment, start: pd.Timestamp,
                 end: pd.Timestamp) -> Window:
    series = facts.series(segment)
    reference = pd.Series({column: b.reference(series[column], series["lines"]).get(start, np.nan)
                           for column in ("list_net", "discount", "cogs", "shipping", "payment", "commission")})
    reference["return_cost_per_return"] = b.reference(series["return_cost"], series["returns"]).get(start, np.nan)
    return Window(segment, start, end, series.loc[start:end], detection.expected[segment.key].loc[start:end],
                  reference.fillna(0.0))


def _one_account(facts: ShopFacts, w: Window, metric: str, change: float) -> tuple[Part | None, pd.DataFrame]:
    """The customer who placed several orders and made at least ACCOUNT_SHARE of the change (same sign)."""
    lines = facts.lines[_window(facts.lines, w.start, w.end) & _segment_mask(facts.lines, w.segment)]
    value = "sales" if metric == "sales" else "margin"
    if lines.empty or change == 0:
        return None, lines
    per_customer = lines.groupby("customer_id").agg(amount=(value, "sum"), orders=("order_id", "nunique"))
    candidates = per_customer[per_customer["orders"] >= 2]
    if candidates.empty:
        return None, lines
    best = candidates["amount"].idxmax() if change > 0 else candidates["amount"].idxmin()
    amount = float(candidates.loc[best, "amount"])
    if amount / change < ACCOUNT_SHARE:
        return None, lines
    part = Part("single_account", "single_account", amount)
    return part, lines[lines["customer_id"] != best].assign(_account=int(best))


def _volume_parts(shop: ShopData, facts: ShopFacts, detection: Detection, w: Window, lines_actual: float,
                  lines_expected: float, volume: float) -> tuple[list[Part], dict]:
    """Split the volume part by LMDI into traffic, conversion and share (or lines per order)."""
    segment = w.segment
    channel_segment = Segment(channel=segment.channel)
    traffic_actual = facts.series(channel_segment).loc[w.start:w.end]
    traffic_expected = detection.expected[channel_segment.key].loc[w.start:w.end]
    sessions1, sessions0 = float(traffic_actual["sessions"].sum()), float(traffic_expected["sessions"].sum())
    orders1 = float(traffic_actual["orders"].sum())
    orders0 = float((traffic_expected["sessions"] * traffic_expected["conversion"]).sum())
    signals: dict = {"sessions": sessions1, "expected_sessions": sessions0, "orders": orders1,
                     "expected_orders": orders0}
    factors = {
        "traffic": (sessions1, sessions0),
        "conversion": (orders1 / max(sessions1, EPSILON), orders0 / max(sessions0, EPSILON)),
    }
    last = "share" if segment.category != ALL else "basket"
    factors[last] = (lines_actual / max(orders1, EPSILON), lines_expected / max(orders0, EPSILON))
    parts = []
    collapsed = next((name for name, (actual, _) in factors.items() if actual <= 0), None)
    if lines_actual <= 0 or collapsed:
        # LMDI needs every factor above zero; when one collapses (no sessions at all, no orders), the whole
        # volume belongs to the first factor that did.
        first = collapsed or next(iter(factors))
        parts = [Part(f"volume_{name}", "", volume if name == first else 0.0) for name in factors]
    else:
        log_total = np.log(lines_actual / lines_expected) if lines_expected > 0 else 0.0
        for name, (actual, expected) in factors.items():
            share = np.log(actual / expected) / log_total if abs(log_total) > 1e-12 and expected > 0 else \
                1 / len(factors)
            parts.append(Part(f"volume_{name}", "", volume * share))
    # Causes from the evidence.
    shop_traffic = facts.series(Segment()).loc[w.start:w.end, "sessions"].sum()
    shop_expected = detection.expected[Segment().key].loc[w.start:w.end, "sessions"].sum()
    outage = shop_traffic < OUTAGE_TRAFFIC * shop_expected
    campaigns = running_campaigns(shop, segment.channel, w.start, w.end) if segment.channel != ALL else []
    competitor = competitor_signal(shop, segment.category, w.start, w.end) if segment.category != ALL else None
    for part in parts:
        if part.name == "volume_traffic":
            part.cause = "outage" if outage else ("campaign" if campaigns else "traffic")
        elif part.name == "volume_conversion":
            part.cause = "outage" if outage else "conversion"
        elif part.name == "volume_share":
            part.cause = "competitor_prices" if competitor and volume < 0 else "demand"
        else:  # basket: lines per order
            part.cause = "demand"
    signals.update({"outage": bool(outage), "campaigns": campaigns, "competitor": competitor})
    return parts, signals


def _price_volume(shop: ShopData, facts: ShopFacts, detection: Detection, w: Window, metric: str) -> tuple[
        list[Part], float, float, dict]:
    observed = float(w.series["sales" if metric == "sales" else "margin_sold"].sum())
    expected = float(w.expected["sales" if metric == "sales" else "margin_sold"].sum())
    change = observed - expected
    account, lines = _one_account(facts, w, metric, change)
    parts: list[Part] = []
    signals: dict = {}
    if account:
        parts.append(account)
        signals["account"] = {"customer_id": int(lines["_account"].iloc[0]) if "_account" in lines else None}
    n1 = float(len(lines))
    n0 = float(w.expected["lines"].sum())
    per_line_expected = expected / n0 if n0 > 0 else 0.0
    sums = lines[["list_net", "discount", "cogs", "shipping", "payment", "commission", "sales", "margin"]].sum()
    actual = sums / n1 if n1 > 0 else sums * 0
    ref = w.reference
    if metric == "sales":
        ref_value = ref["list_net"] - ref["discount"]
        costs = ()
    else:
        ref_value = ref["list_net"] - ref["discount"] - ref["cogs"] - ref["shipping"] - ref["payment"] - \
            ref["commission"]
        costs = ("shipping", "payment", "commission")
    gap = per_line_expected - ref_value  # the model's season and mix against the recent weeks: prices and mix
    volume = (n1 - n0) * per_line_expected
    gross = (actual["list_net"] - (ref["list_net"] + gap)) - (
        (actual["cogs"] - ref["cogs"]) if metric == "margin" else 0.0)
    parts.append(Part("prices", "prices", n1 * gross))
    parts.append(Part("discount", "discount", -n1 * (actual["discount"] - ref["discount"])))
    for name, cause in zip(costs, ("shipping", "payment_fees", "commission"), strict=False):
        parts.append(Part(name, cause, -n1 * (actual[name] - ref[name])))
    volume_parts, volume_signals = _volume_parts(shop, facts, detection, w, n1, n0, volume)
    parts = parts[:1] + volume_parts + parts[1:] if account else volume_parts + parts
    signals.update(volume_signals)
    signals.update({"lines": n1, "expected_lines": n0,
                    "discount_rate": float(sums["discount"] / sums["list_net"]) if sums["list_net"] else 0.0,
                    "expected_discount_rate": float(ref["discount"] / ref["list_net"]) if ref["list_net"] else 0.0})
    return parts, observed, expected, signals


def _returns(facts: ShopFacts, w: Window) -> tuple[list[Part], float, float, dict]:
    observed = float(w.series["return_cost"].sum())
    expected = float(w.expected["return_cost"].sum())
    back = facts.returns[_window(facts.returns, w.start, w.end) & _segment_mask(facts.returns, w.segment)]
    cost0 = w.reference["return_cost_per_return"] or (observed / max(len(back), 1))
    r1, r0 = float(len(back)), expected / cost0 if cost0 else 0.0
    cost1 = observed / r1 if r1 else 0.0
    # Reason mix of the recent weeks for the segment.
    before = facts.returns[_window(facts.returns, w.start - pd.Timedelta(days=63), w.start - pd.Timedelta(days=8))
                           & _segment_mask(facts.returns, w.segment)]
    mix = before["reason"].value_counts(normalize=True) if len(before) else pd.Series(dtype=float)
    counts = back["reason"].value_counts()
    parts = []
    for reason in sorted(set(mix.index) | set(counts.index)):
        parts.append(Part(f"returns_{reason}", "returns", (counts.get(reason, 0) - r0 * mix.get(reason, 0.0)) * cost0))
    if mix.empty:  # no history: all extra returns in one part
        parts = [Part("returns_more", "returns", (r1 - r0) * cost0)]
    parts.append(Part("returns_cost_each", "returns", r1 * (cost1 - cost0)))
    top = back["product_id"].value_counts()
    signals = {"returns": r1, "expected_returns": r0, "cost_per_return": cost1, "expected_cost_per_return": cost0,
               "top_reason": str(counts.idxmax()) if len(counts) else None,
               "top_product": int(top.index[0]) if len(top) else None,
               "top_product_returns": int(top.iloc[0]) if len(top) else 0}
    return parts, observed, expected, signals


def _costs(shop: ShopData, facts: ShopFacts, w: Window) -> tuple[list[Part], float, float, dict]:
    observed = float(w.series["costs"].sum())
    expected = float(w.expected["costs"].sum())
    names = ("shipping", "payment", "commission")
    shares = w.reference[list(names)]
    shares = shares / shares.sum() if shares.sum() else pd.Series(1 / 3, index=list(names))
    parts = [Part(name, cause, float(w.series[name].sum()) - expected * float(shares[name]))
             for name, cause in zip(names, ("shipping", "payment_fees", "commission"), strict=True)]
    signals: dict = {}
    lines = facts.lines[_window(facts.lines, w.start, w.end) & _segment_mask(facts.lines, w.segment)]
    orders = lines["order_id"].unique()
    now = shop.shipments[shop.shipments["order_id"].isin(orders)]
    earlier_lines = facts.lines[_window(facts.lines, w.start - pd.Timedelta(days=35), w.start - pd.Timedelta(days=8))
                                & _segment_mask(facts.lines, w.segment)]
    before = shop.shipments[shop.shipments["order_id"].isin(earlier_lines["order_id"].unique())]
    per_parcel = now.groupby("delivery_method")["carrier_cost"].mean() / \
        before.groupby("delivery_method")["carrier_cost"].mean() - 1
    if len(per_parcel.dropna()):
        method = str(per_parcel.idxmax())
        signals["carrier"] = {"delivery_method": method, "cost_change": float(per_parcel.max()),
                              "parcels": int((now["delivery_method"] == method).sum())}
    return parts, observed, expected, signals


def _conversion(facts: ShopFacts, detection: Detection, w: Window) -> tuple[list[Part], float, float, dict]:
    traffic = facts.traffic[_window(facts.traffic, w.start, w.end)]
    before = facts.traffic[_window(facts.traffic, w.start - pd.Timedelta(days=35), w.start - pd.Timedelta(days=8))]
    if w.segment.channel != ALL:
        traffic = traffic[traffic["channel"] == w.segment.channel]
        before = before[before["channel"] == w.segment.channel]
    expected_p = w.expected["conversion"]
    sessions = w.series["sessions"]
    orders0 = float((sessions * expected_p).sum())
    orders1 = float(w.series["orders"].sum())
    value = float(b.reference(facts.series(w.segment)["sales"], facts.series(w.segment)["orders"]).get(w.start, 0))
    by_device = traffic.groupby("device")[["sessions", "orders"]].sum()
    past = before.groupby("device")[["sessions", "orders"]].sum()
    rate0 = (past["orders"] / past["sessions"]).reindex(by_device.index).fillna(0)
    expected_device = by_device["sessions"] * rate0
    if expected_device.sum() > 0:
        expected_device *= orders0 / expected_device.sum()  # the model's level, the recent split by device
    lost = by_device["orders"] - expected_device
    parts = [Part(f"conversion_{device}", "conversion", float(lost[device] * value), orders=float(lost[device]))
             for device in by_device.index]
    signals = {"orders": orders1, "expected_orders": orders0, "order_value": value,
               "sessions": float(sessions.sum()), "conversion": orders1 / max(float(sessions.sum()), 1),
               "expected_conversion": orders0 / max(float(sessions.sum()), 1)}
    return parts, orders1 * value, orders0 * value, signals


def main_cause(parts: list[Part], change: float) -> str:
    """The cause whose parts move most with the change (summed per cause)."""
    by_cause: dict[str, float] = {}
    for part in parts:
        by_cause[part.cause] = by_cause.get(part.cause, 0.0) + part.amount
    sign = 1 if change >= 0 else -1
    return max(by_cause, key=lambda cause: sign * by_cause[cause])


def explain(anomaly: Anomaly, shop: ShopData, facts: ShopFacts, detection: Detection,
            episode: Episode | None = None) -> Breakdown:
    """The breakdown of an anomaly's main episode (or of the episode given)."""
    episode = episode or anomaly.main
    segment = Segment.from_key(anomaly.segment)
    w = _window_data(facts, detection, segment, episode.start, episode.end)
    if episode.metric in ("sales", "margin"):
        parts, observed, expected, signals = _price_volume(shop, facts, detection, w, episode.metric)
    elif episode.metric == "returns":
        parts, observed, expected, signals = _returns(facts, w)
    elif episode.metric == "costs":
        parts, observed, expected, signals = _costs(shop, facts, w)
    else:
        parts, observed, expected, signals = _conversion(facts, detection, w)
    change = to_grosz(parts, observed, expected)
    return Breakdown(anomaly.anomaly_id, anomaly.segment, episode.metric, episode.direction,
                     episode.start.strftime("%Y-%m-%d"), episode.end.strftime("%Y-%m-%d"), observed, expected,
                     change, parts, main_cause(parts, observed - expected), signals)


def explain_all(anomalies: list[Anomaly], shop: ShopData, facts: ShopFacts, detection: Detection) -> list[Breakdown]:
    return [explain(a, shop, facts, detection) for a in anomalies]


