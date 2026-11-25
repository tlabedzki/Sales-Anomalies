"""
Injected anomalies with a known cause and a known size: the first answer key.

Each scenario changes the Data-Generator tables the way the real event would, and keeps them consistent (a lower
price also lowers the payment, its fee, the refund of a later return and the affiliate commission; a lost order
takes its shipment, payment and returns with it and leaves its session without an order). Its true impact is the
difference between the daily facts after and before the injection, in its segments and days, so it is exact.

    kind                 cause               what changes
    discount             discount            extra discount on one category in one channel
    price_error          prices              list price of the category's best sellers set too low
    return_wave          returns             a faulty batch: many lines of one category come back
    carrier_surcharge    shipping            a carrier charges more for one delivery method
    competitor_undercut  competitor_prices   one competitor cuts prices in a category; our orders there drop
    campaign_stop        campaign            a running campaign stops bringing visitors
    checkout_failure     conversion          checkout fails on mobile phones; visits stay, orders do not
    seo_drop             traffic             organic search visits fall
    commission_error     commission          the affiliate network charges a higher commission

Scenarios do not overlap in time, avoid November and December (peak season), the days around the shop's own
outages and a three-month negative period (2024-03-01 to 2024-05-31) left without any injection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sales_anomalies.data import ShopData
from sales_anomalies.facts import ALL, Segment, ShopFacts
from sales_anomalies.truth import TruthEvent

SCENARIO_SEED = 2026
NEGATIVE_PERIOD = (pd.Timestamp("2024-03-01"), pd.Timestamp("2024-05-31"))
PERIOD = (pd.Timestamp("2023-01-08"), pd.Timestamp("2024-12-31"))
SLOT_DAYS = 14
# The column whose relative change says how big a scenario is (its own segment and days).
PRIMARY = {"discount": "margin_sold", "price_error": "margin_sold", "return_wave": "return_cost",
           "carrier_surcharge": "costs", "competitor_undercut": "sales", "campaign_stop": "sales",
           "checkout_failure": "orders", "seo_drop": "sales", "commission_error": "costs"}
KINDS = ("discount", "price_error", "return_wave", "carrier_surcharge", "competitor_undercut", "campaign_stop",
         "checkout_failure", "seo_drop", "commission_error")
CAUSE = {"discount": "discount", "price_error": "prices", "return_wave": "returns", "carrier_surcharge": "shipping",
         "competitor_undercut": "competitor_prices", "campaign_stop": "campaign", "checkout_failure": "conversion",
         "seo_drop": "traffic", "commission_error": "commission"}
METRICS = {"discount": ["sales", "margin"], "price_error": ["sales", "margin"], "return_wave": ["returns"],
           "carrier_surcharge": ["costs", "margin"],
           # visitors still come and compare, then buy elsewhere: conversion falls too
           "competitor_undercut": ["sales", "margin", "conversion"],
           "campaign_stop": ["sales", "margin"], "checkout_failure": ["conversion", "sales", "margin"],
           "seo_drop": ["sales", "margin"], "commission_error": ["costs", "margin"]}


@dataclass
class Scenario:
    scenario_id: str
    kind: str
    start: pd.Timestamp
    days: int
    size: float                    # share or rate of the change (see each injector)
    category: str = ALL
    channel: str = ALL
    detail: str = ""               # delivery method, competitor, campaign or device
    effect: tuple[pd.Timestamp, pd.Timestamp] | None = None   # days the facts move (returns come back later)

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(days=self.days - 1)

    @property
    def segment(self) -> Segment:
        return Segment(self.category, self.channel)

    def describe(self) -> str:
        where = self.segment.label()
        texts = {
            "discount": f"extra {self.size:.0%} discount in {where}",
            "price_error": f"best sellers of {where} priced {self.size:.0%} too low",
            "return_wave": f"{self.size:.0%} of {where} lines come back (faulty batch)",
            "carrier_surcharge": f"{self.detail} charges {self.size:.0%} more",
            "competitor_undercut": f"{self.detail} cuts {where} prices; {self.size:.0%} of our orders there lost",
            "campaign_stop": f"campaign {self.detail} stops: {self.size:.0%} of its visits lost",
            "checkout_failure": f"checkout fails on {self.detail}: {self.size:.0%} of its orders lost",
            "seo_drop": f"organic search visits fall by {self.size:.0%}",
            "commission_error": f"affiliate commission {1 + self.size:.1f}x higher",
        }
        return texts[self.kind]


# ---------------------------------------------------------------------------------------------------------------
# Plan


def _blocked_days(shop: ShopData) -> pd.DatetimeIndex:
    days = pd.date_range(*PERIOD, freq="D")
    blocked = (days.month >= 11) | ((days >= NEGATIVE_PERIOD[0] - pd.Timedelta(days=SLOT_DAYS))
                                    & (days <= NEGATIVE_PERIOD[1]))
    for row in shop.anomalies[shop.anomalies["anomaly_type"] == "shop_outage"].itertuples():
        blocked |= (days >= row.start_date - pd.Timedelta(days=SLOT_DAYS)) & (days <= row.end_date + pd.Timedelta(
            days=5))
    return days[blocked]


def slots(shop: ShopData) -> list[pd.Timestamp]:
    """Start days at least SLOT_DAYS apart whose slot lies outside the blocked days."""
    blocked = set(_blocked_days(shop))
    result, day = [], PERIOD[0]
    while day + pd.Timedelta(days=SLOT_DAYS) <= PERIOD[1]:
        slot = pd.date_range(day, day + pd.Timedelta(days=SLOT_DAYS - 1), freq="D")
        if not any(d in blocked for d in slot):
            result.append(day)
            day += pd.Timedelta(days=SLOT_DAYS)
        else:
            day += pd.Timedelta(days=1)
    return result


def plan(shop: ShopData, seed: int = SCENARIO_SEED) -> list[Scenario]:
    """One scenario per slot, kinds in a shuffled cycle, segment and size drawn at random (seeded)."""
    rng = np.random.default_rng(seed)
    starts = slots(shop)
    kinds = [KINDS[i % len(KINDS)] for i in range(len(starts))]
    rng.shuffle(kinds)
    lines = shop.orders.merge(shop.products[["product_id", "category"]], on="product_id")
    lines = lines[~lines["order_status"].isin(["cancelled", "payment_failed"])]
    scenarios = []
    for number, (start, kind) in enumerate(zip(starts, kinds, strict=True), start=1):
        before = lines[lines["order_date"].between(start - pd.Timedelta(days=56), start - pd.Timedelta(days=1))]
        per_day = before.groupby(["category", "channel"]).size() / 56
        by_category = before.groupby("category").size() / 56
        scenario = Scenario(f"s{number:02d}", kind, start, 1, 0.0)
        if kind == "campaign_stop" and _running_campaign(shop, start, None) is None:
            kind = "seo_drop"  # no campaign to stop in this slot
            scenario.kind = kind
        if kind == "discount":
            cells = per_day[per_day >= 4].index if (per_day >= 4).any() else per_day.nlargest(5).index
            scenario.category, scenario.channel = cells[rng.integers(len(cells))]
            scenario.days, scenario.size = int(rng.integers(3, 10)), float(rng.choice([0.15, 0.25, 0.35, 0.5]))
        elif kind == "price_error":
            names = _enough(by_category, 15)
            scenario.category = names[rng.integers(len(names))]
            scenario.days, scenario.size = int(rng.integers(2, 6)), float(rng.choice([0.3, 0.45, 0.6]))
        elif kind == "return_wave":
            names = _enough(by_category, 8)
            scenario.category = names[rng.integers(len(names))]
            scenario.days, scenario.size = int(rng.integers(3, 7)), float(rng.choice([0.25, 0.4, 0.6]))
        elif kind == "carrier_surcharge":
            scenario.detail = str(rng.choice(["InPost parcel locker", "InPost parcel locker", "DPD", "DHL"]))
            scenario.days, scenario.size = int(rng.integers(5, 12)), float(rng.choice([1.0, 2.0, 3.0]))
        elif kind == "competitor_undercut":
            names = _enough(by_category, 8)
            scenario.category = names[rng.integers(len(names))]
            scenario.detail = str(rng.choice(["Frugalia", "Mercatoria", "Fulmina"]))
            scenario.days, scenario.size = int(rng.integers(6, 12)), float(rng.choice([0.35, 0.5, 0.7]))
        elif kind == "campaign_stop":
            campaign = _running_campaign(shop, start, rng)
            scenario.channel, scenario.detail = campaign["channel"], campaign["campaign_name"]
            last = min(campaign["end_date"], start + pd.Timedelta(days=SLOT_DAYS - 4))
            scenario.start = max(start, campaign["start_date"])
            scenario.days = max(1, min(int(rng.integers(3, 7)), int((last - scenario.start).days) + 1))
            scenario.size = float(rng.choice([0.7, 0.85, 1.0]))
        elif kind == "checkout_failure":
            scenario.detail = "mobile"
            scenario.days, scenario.size = int(rng.integers(1, 4)), float(rng.choice([0.5, 0.7, 0.9, 1.0]))
        elif kind == "seo_drop":
            scenario.channel = "organic_search"
            scenario.days, scenario.size = int(rng.integers(5, 12)), float(rng.choice([0.3, 0.45, 0.6, 0.75]))
        elif kind == "commission_error":
            scenario.channel = "affiliate"
            scenario.days, scenario.size = int(rng.integers(5, 12)), float(rng.choice([1.0, 2.0, 3.0]))
        scenarios.append(scenario)
    return scenarios


def _enough(per_day: pd.Series, least: float) -> pd.Index:
    """Segments with at least `least` order lines a day; the five busiest when none has that many."""
    chosen = per_day[per_day >= least].index
    return chosen if len(chosen) else per_day.nlargest(5).index


def _running_campaign(shop: ShopData, start: pd.Timestamp, rng: np.random.Generator | None) -> pd.Series | None:
    """A campaign running in the slot (None when there is none); without rng only says whether there is one."""
    campaigns = shop.campaigns[shop.campaigns["channel"].isin(["social", "affiliate", "paid_search"])]
    window_end = start + pd.Timedelta(days=SLOT_DAYS - 4)
    running = campaigns[(campaigns["start_date"] <= window_end - pd.Timedelta(days=2))
                        & (campaigns["end_date"] >= start + pd.Timedelta(days=2))]
    if running.empty:
        return None
    return running.iloc[int(rng.integers(len(running)))] if rng is not None else running.iloc[0]


# ---------------------------------------------------------------------------------------------------------------
# Injection helpers


def _sold(orders: pd.DataFrame) -> pd.Series:
    return ~orders["order_status"].isin(["cancelled", "payment_failed"])


def _categories(shop: ShopData) -> pd.Series:
    return shop.orders["product_id"].map(shop.products.set_index("product_id")["category"])


def _reprice(shop: ShopData, mask: pd.Series, new_price: pd.Series | None = None,
             new_discount: pd.Series | None = None) -> None:
    """Change the price or discount of order lines and carry it to payments, fees, refunds and commission."""
    orders = shop.orders
    old_net = orders.loc[mask, "unit_price_gross"] * (1 - orders.loc[mask, "discount_rate"])
    if new_price is not None:
        orders.loc[mask, "unit_price_gross"] = new_price
    if new_discount is not None:
        orders.loc[mask, "discount_rate"] = new_discount
    new_net = orders.loc[mask, "unit_price_gross"] * (1 - orders.loc[mask, "discount_rate"])
    factor = (new_net / old_net.where(old_net > 0)).fillna(1.0)
    orders.loc[mask, "affiliate_commission"] = orders.loc[mask, "affiliate_commission"] * factor
    change = ((new_net - old_net) * orders.loc[mask, "quantity"]).groupby(orders.loc[mask, "order_id"]).sum()
    payments = shop.payments
    hit = payments["order_id"].isin(change.index)
    payments.loc[hit, "payment_amount"] = payments.loc[hit, "payment_amount"] + payments.loc[hit, "order_id"].map(
        change)
    payments.loc[hit, "payment_fee"] = (payments.loc[hit, "payment_amount"] * payments.loc[hit, "fee_rate"]
                                        + payments.loc[hit, "fixed_fee"]).round(2)
    line_factor = pd.Series(factor.to_numpy(), index=orders.loc[mask, "order_line_id"].to_numpy())
    returns = shop.returns
    back = returns["order_line_id"].isin(line_factor.index)
    returns.loc[back, "refund_amount"] = (returns.loc[back, "refund_amount"]
                                          * returns.loc[back, "order_line_id"].map(line_factor)).round(2)


def remove_orders(shop: ShopData, order_ids: pd.Index, keep_sessions: bool = True) -> None:
    """Take orders out of the shop with their shipments, payments and returns; their sessions end without an order
    (keep_sessions) or are removed too."""
    ids = set(order_ids)
    shop.orders = shop.orders[~shop.orders["order_id"].isin(ids)].reset_index(drop=True)
    shop.shipments = shop.shipments[~shop.shipments["order_id"].isin(ids)].reset_index(drop=True)
    shop.payments = shop.payments[~shop.payments["order_id"].isin(ids)].reset_index(drop=True)
    shop.returns = shop.returns[~shop.returns["order_id"].isin(ids)].reset_index(drop=True)
    converted = shop.sessions["order_id"].isin(ids)
    if keep_sessions:
        shop.sessions.loc[converted, "order_id"] = pd.NA
    else:
        shop.sessions = shop.sessions[~converted].reset_index(drop=True)


def _window(frame: pd.DataFrame, column: str, scenario: Scenario) -> pd.Series:
    return frame[column].between(scenario.start, scenario.end)


# ---------------------------------------------------------------------------------------------------------------
# Injectors


def _discount(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    o = shop.orders
    mask = _sold(o) & _window(o, "order_date", s) & (_categories(shop) == s.category) & (o["channel"] == s.channel)
    _reprice(shop, mask, new_discount=(o.loc[mask, "discount_rate"] + s.size).clip(upper=0.9))


def _price_error(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    o = shop.orders
    in_category = _sold(o) & (_categories(shop) == s.category)
    recent = o[in_category & o["order_date"].between(s.start - pd.Timedelta(days=56), s.start - pd.Timedelta(days=1))]
    sales = (recent["quantity"] * recent["unit_price_gross"]).groupby(recent["product_id"]).sum().sort_values(
        ascending=False)
    best = sales.index[: max(1, int((sales.cumsum() <= 0.6 * sales.sum()).sum()))]
    mask = in_category & _window(o, "order_date", s) & o["product_id"].isin(best)
    _reprice(shop, mask, new_price=(o.loc[mask, "unit_price_gross"] * (1 - s.size)).round(2))


def _return_wave(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    o = shop.orders
    delivered = (o["order_status"] == "delivered") & _window(o, "order_date", s) & (_categories(shop) == s.category)
    already = shop.returns["order_line_id"]
    candidates = o[delivered & ~o["order_line_id"].isin(already)]
    chosen = candidates[rng.random(len(candidates)) < s.size]
    if chosen.empty:
        return
    cost = chosen["product_id"].map(shop.products.set_index("product_id")["net_purchase_price"])
    new = pd.DataFrame({
        "return_id": np.arange(len(chosen)) + int(shop.returns["return_id"].max()) + 1,
        "order_id": chosen["order_id"].to_numpy(),
        "order_line_id": chosen["order_line_id"].to_numpy(),
        "product_id": chosen["product_id"].to_numpy(),
        "return_date": (chosen["order_date"] + pd.to_timedelta(rng.integers(2, 8, len(chosen)), unit="D")).to_numpy(),
        "returned_quantity": chosen["quantity"].to_numpy(),
        "return_reason": "defective",
        "refund_amount": (chosen["quantity"] * chosen["unit_price_gross"] * (1 - chosen["discount_rate"])).round(2)
        .to_numpy(),
        "return_status": "refunded",
        "return_shipping_cost": 8.5,
        "handling_cost": 8.0,
        "written_off_quantity": chosen["quantity"].to_numpy(),
        "write_off_cost": (chosen["quantity"] * cost).round(2).to_numpy(),
    })
    shop.returns = pd.concat([shop.returns, new.astype(shop.returns.dtypes.to_dict())], ignore_index=True)
    s.effect = (pd.Timestamp(new["return_date"].min()), pd.Timestamp(new["return_date"].max()))


def _carrier_surcharge(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    o = shop.orders
    orders = o.loc[_window(o, "order_date", s), "order_id"].unique()
    hit = shop.shipments["order_id"].isin(orders) & (shop.shipments["delivery_method"] == s.detail)
    shop.shipments.loc[hit, "carrier_cost"] = (shop.shipments.loc[hit, "carrier_cost"] * (1 + s.size)).round(2)


def _competitor_undercut(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    products = shop.products.loc[shop.products["category"] == s.category, "product_id"]
    prices = shop.competitor_prices
    valid_to = prices["valid_to"].fillna(pd.Timestamp.max)
    covering = (prices["competitor_name"] == s.detail) & prices["product_id"].isin(products) & \
        (prices["valid_from"] <= s.end) & (valid_to >= s.start)
    old = prices[covering]
    pieces = []
    for low, high, factor in ((None, s.start - pd.Timedelta(days=1), 1.0),   # before the cut
                              (s.start, s.end, 0.8),                          # 20% lower while it lasts
                              (s.end + pd.Timedelta(days=1), None, 1.0)):     # back after it
        piece = old.copy()
        start = piece["valid_from"] if low is None else piece["valid_from"].clip(lower=low)
        end = piece["valid_to"] if high is None else piece["valid_to"].fillna(pd.Timestamp.max).clip(upper=high)
        piece["valid_from"], piece["valid_to"] = start, end.where(end != pd.Timestamp.max)
        piece = piece[piece["valid_from"] <= piece["valid_to"].fillna(pd.Timestamp.max)]
        piece["price_gross"] = (piece["price_gross"] * factor).round(2)
        pieces.append(piece)
    shop.competitor_prices = pd.concat([prices[~covering], *pieces], ignore_index=True).sort_values(
        ["product_id", "competitor_id", "valid_from"], ignore_index=True)
    o = shop.orders
    order_categories = pd.DataFrame({"order_id": o["order_id"], "category": _categories(shop)})
    only_here = order_categories.groupby("order_id")["category"].agg(lambda c: set(c) == {s.category})
    candidates = o.loc[_window(o, "order_date", s) & o["order_id"].map(only_here), "order_id"].unique()
    lost = candidates[rng.random(len(candidates)) < s.size]
    remove_orders(shop, pd.Index(lost))


def _campaign_stop(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    campaign_id = shop.campaigns.loc[shop.campaigns["campaign_name"] == s.detail, "campaign_id"].iloc[0]
    sessions = shop.sessions
    days = sessions["session_start"].dt.normalize()
    # campaign_id is a nullable integer: compare with NA filled, or NA rows would leak into the mask.
    mask = (sessions["campaign_id"] == campaign_id).fillna(False).astype(bool) & days.between(s.start, s.end)
    lost = mask & pd.Series(rng.random(len(sessions)) < s.size, index=sessions.index)
    orders = sessions.loc[lost, "order_id"].dropna()
    shop.sessions = sessions[~lost].reset_index(drop=True)
    remove_orders(shop, pd.Index(orders.astype(int)), keep_sessions=False)


def _checkout_failure(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    o = shop.orders
    mask = _window(o, "order_date", s) & (o["device"] == s.detail)
    candidates = o.loc[mask, "order_id"].unique()
    remove_orders(shop, pd.Index(candidates[rng.random(len(candidates)) < s.size]), keep_sessions=True)


def _seo_drop(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    sessions = shop.sessions
    mask = (sessions["channel"] == "organic_search") & sessions["session_start"].dt.normalize().between(s.start, s.end)
    lost = mask & pd.Series(rng.random(len(sessions)) < s.size, index=sessions.index)
    orders = sessions.loc[lost, "order_id"].dropna()
    shop.sessions = sessions[~lost].reset_index(drop=True)
    remove_orders(shop, pd.Index(orders.astype(int)), keep_sessions=False)


def _commission_error(shop: ShopData, s: Scenario, rng: np.random.Generator) -> None:
    o = shop.orders
    mask = _window(o, "order_date", s) & (o["channel"] == "affiliate")
    o.loc[mask, "affiliate_commission"] = (o.loc[mask, "affiliate_commission"] * (1 + s.size)).round(2)


INJECTORS = {"discount": _discount, "price_error": _price_error, "return_wave": _return_wave,
             "carrier_surcharge": _carrier_surcharge, "competitor_undercut": _competitor_undercut,
             "campaign_stop": _campaign_stop, "checkout_failure": _checkout_failure, "seo_drop": _seo_drop,
             "commission_error": _commission_error}


def inject(shop: ShopData, scenarios: list[Scenario], seed: int = SCENARIO_SEED) -> ShopData:
    """A copy of the shop with every scenario applied (the input stays unchanged)."""
    changed = shop.copy()
    rng = np.random.default_rng(seed + 1)
    for scenario in scenarios:
        INJECTORS[scenario.kind](changed, scenario, rng)
    return changed


# ---------------------------------------------------------------------------------------------------------------
# Truth


def _effect_window(s: Scenario) -> tuple[pd.Timestamp, pd.Timestamp]:
    return s.effect if s.effect is not None else (s.start, s.end)


def _sum(facts: ShopFacts, segment: Segment, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return facts.series(segment).loc[start:end].sum()


def truth(scenarios: list[Scenario], before: ShopFacts, after: ShopFacts) -> list[TruthEvent]:
    """Truth events with the exact impact of each scenario (facts after minus facts before)."""
    events = []
    for s in scenarios:
        start, end = _effect_window(s)
        changed, original = _sum(after, s.segment, start, end), _sum(before, s.segment, start, end)
        names = ("sales", "margin_sold", "margin", "return_cost", "costs")
        impact = {name: float(changed[name] - original[name]) for name in names}
        if s.segment.level in ("shop", "channel"):
            impact["orders"] = float(changed["orders"] - original["orders"])
            impact["sessions"] = float(changed["sessions"] - original["sessions"])
        base = {name: float(original[name]) for name in names}
        if s.segment.level in ("shop", "channel"):
            base["orders"] = float(original["orders"])
        column = PRIMARY[s.kind]
        relative = impact[column] / base[column] if base.get(column) else float("nan")
        events.append(TruthEvent(s.scenario_id, "scenario", s.kind, CAUSE[s.kind], start, end, _cells(s, after),
                                 METRICS[s.kind], impact["margin"], s.describe(),  # margin after returns
                                 {"impact": impact, "base": base, "scenario": as_record(s), "primary": column,
                                  "relative": relative}))
    return events


def _cells(s: Scenario, facts: ShopFacts) -> list[str]:
    """The scenario's segment and every cell inside it (a channel alert can then point at a category event)."""
    inside = [Segment(c, ch).key for c in facts.categories for ch in facts.channels
              if s.segment.contains(Segment(c, ch))]
    return [s.segment.key] + ([] if s.segment.level == "shop" else inside)


def as_record(s: Scenario) -> dict:
    values = asdict(s)
    values["start"] = s.start.strftime("%Y-%m-%d")
    values["effect"] = [d.strftime("%Y-%m-%d") for d in _effect_window(s)]
    return values
