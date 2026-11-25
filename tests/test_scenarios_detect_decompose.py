"""The chain on the synthetic shop: inject, detect, locate, break down; every part checked against the truth."""
import numpy as np
import pandas as pd
import pytest

from sales_anomalies import decompose, detect, evaluate, facts, scenarios
from sales_anomalies.decompose import Part, to_grosz
from sales_anomalies.facts import Segment

SCENARIOS = [
    scenarios.Scenario("t1", "discount", pd.Timestamp("2023-03-06"), 6, 0.4, "Clothing", "direct"),
    scenarios.Scenario("t2", "competitor_undercut", pd.Timestamp("2023-06-05"), 6, 0.8, "Clothing", detail="Frugalia"),
    scenarios.Scenario("t3", "campaign_stop", pd.Timestamp("2023-08-02"), 4, 1.0, channel="social",
                       detail="Social 2023-08-01"),
    scenarios.Scenario("t4", "checkout_failure", pd.Timestamp("2023-10-10"), 5, 1.0, detail="mobile"),
]


@pytest.fixture(scope="module")
def injected(shop):
    return scenarios.inject(shop, SCENARIOS)


@pytest.fixture(scope="module")
def injected_facts(injected):
    return facts.build(injected)


@pytest.fixture(scope="module")
def injected_detection(injected_facts):
    return detect.detect(injected_facts, thresholds={})


@pytest.fixture(scope="module")
def events(shop_facts, injected_facts):
    return scenarios.truth(SCENARIOS, shop_facts, injected_facts)


def test_injection_keeps_the_input_and_the_tables_consistent(shop, injected):
    assert len(shop.orders) > len(injected.orders)              # orders were lost in the copy only
    orders = set(injected.orders["order_id"])
    assert set(injected.payments["order_id"]) <= orders
    assert set(injected.shipments["order_id"]) <= orders
    assert set(injected.returns["order_line_id"]) <= set(injected.orders["order_line_id"])
    assert set(injected.sessions["order_id"].dropna().astype(int)) <= orders


def test_discount_lowers_the_price_payment_and_refund_of_its_lines_only(shop, injected):
    s = SCENARIOS[0]
    category = injected.orders["product_id"].map(injected.products.set_index("product_id")["category"])
    inside = injected.orders["order_date"].between(s.start, s.end) & (category == "Clothing") & \
        (injected.orders["channel"] == "direct") & (injected.orders["order_status"] == "delivered")
    before = shop.orders.set_index("order_line_id")["discount_rate"]
    after = injected.orders.set_index("order_line_id")["discount_rate"]
    lines = injected.orders.loc[inside, "order_line_id"]
    assert np.allclose(after[lines], (before[lines] + 0.4).clip(upper=0.9))
    outside = injected.orders.loc[~inside & injected.orders["order_line_id"].isin(before.index), "order_line_id"]
    assert np.allclose(after[outside], before[outside])
    paid = injected.payments.set_index("order_id")["payment_amount"]
    original = shop.payments.set_index("order_id")["payment_amount"]
    order = injected.orders.loc[inside, "order_id"].iloc[0]
    assert paid[order] < original[order]


def test_campaign_stop_removes_only_that_campaigns_visits(shop, injected):
    s = SCENARIOS[2]
    day = lambda frame: frame["session_start"].dt.normalize().between(s.start, s.end)  # noqa: E731
    direct_before = (day(shop.sessions) & (shop.sessions["channel"] == "direct")).sum()
    direct_after = (day(injected.sessions) & (injected.sessions["channel"] == "direct")).sum()
    assert direct_after == direct_before
    social_after = (day(injected.sessions) & (injected.sessions["channel"] == "social")).sum()
    assert social_after < (day(shop.sessions) & (shop.sessions["channel"] == "social")).sum()


def test_truth_impact_is_the_difference_of_the_facts(events, shop_facts, injected_facts):
    competitor = next(e for e in events if e.event_id == "t2")
    segment = Segment(category="Clothing")
    expected = (injected_facts.series(segment).loc[competitor.start:competitor.end, "sales"].sum()
                - shop_facts.series(segment).loc[competitor.start:competitor.end, "sales"].sum())
    assert competitor.details["impact"]["sales"] == pytest.approx(expected)
    assert competitor.details["impact"]["sales"] < 0
    assert "Clothing|direct" in competitor.segments                # the cells inside the category are listed


def test_clean_shop_raises_no_strong_alarm(detection):
    assert all(abs(e.z) < 6.5 for e in detection.episodes)


def test_injected_anomalies_are_found_and_explained(injected_detection, events, shop_facts, injected_facts):
    grades = evaluate.grade(injected_detection.anomalies, events, detect.DETECTION_START,
                            injected_facts.days[-1], (shop_facts, injected_facts))
    found = grades.events.set_index("event_id")["found"]
    assert found["t2"] and found["t3"] and found["t4"]


def test_breakdowns_add_up_to_the_grosz_and_name_the_cause(shop, injected, injected_facts, injected_detection,
                                                          events):
    breakdowns = decompose.explain_all(injected_detection.anomalies, injected, injected_facts, injected_detection)
    assert breakdowns
    for b in breakdowns:
        assert sum(p.grosz for p in b.parts) == b.change_grosz
        assert b.change_grosz == round(b.observed * 100) - round(b.expected * 100)
    by_window = {(b.segment, b.metric): b for b in breakdowns}
    checkout = [b for b in breakdowns if b.metric == "conversion" and b.start >= "2023-10-10"
                and b.end <= "2023-10-14"]
    assert checkout and checkout[0].main_cause == "conversion"
    assert any(b.main_cause == "campaign" for (segment, _), b in by_window.items() if segment == "*|social")


def test_largest_remainder_rounding_is_exact():
    rng = np.random.default_rng(11)
    for _ in range(200):
        amounts = rng.normal(0, 5000, rng.integers(1, 9))
        parts = [Part(f"p{i}", "prices", float(a)) for i, a in enumerate(amounts)]
        expected = float(rng.uniform(1000, 9000))
        observed = expected + float(amounts.sum())
        total = to_grosz(parts, observed, expected)
        assert sum(p.grosz for p in parts) == total
        assert all(abs(p.grosz - p.amount * 100) <= 1.5 for p in parts)


def test_most_specific_keeps_the_cell_that_carries_its_category():
    days = pd.date_range("2023-05-01", "2023-05-03")
    rows = []
    for segment, impact in (("Clothing|*", -1000.0), ("Clothing|direct", -900.0), ("Clothing|social", -100.0)):
        for day in days:
            rows.append({"segment": segment, "metric": "sales", "date": day, "impact": impact / 3})
    table = pd.DataFrame(rows)
    found = [detect.Episode(s, Segment.from_key(s).level, "sales", "down", days[0], days[-1], 5.0, [7],
                            impact=i) for s, i in (("Clothing|*", -1000.0), ("Clothing|direct", -900.0))]
    detect.most_specific(found, table)
    assert [e.kept for e in found] == [False, True]
    spread = [detect.Episode(s, Segment.from_key(s).level, "sales", "down", days[0], days[-1], 5.0, [7], impact=i)
              for s, i in (("*|*", -1000.0), ("Clothing|*", -300.0))]
    rows = [{"segment": "*|*", "metric": "sales", "date": d, "impact": -1000 / 3} for d in days] + \
        [{"segment": "Clothing|*", "metric": "sales", "date": d, "impact": -100.0} for d in days]
    detect.most_specific(spread, pd.DataFrame(rows))
    assert [e.kept for e in spread] == [True, False]
