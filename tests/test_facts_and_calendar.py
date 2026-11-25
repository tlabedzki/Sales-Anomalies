import numpy as np
import pandas as pd
import pytest

from sales_anomalies import calendar
from sales_anomalies.data import load
from sales_anomalies.facts import DETECTION_START, Segment, clip, largest_items


def test_black_friday_and_easter():
    assert calendar.black_friday(2023) == pd.Timestamp("2023-11-24")
    assert calendar.black_friday(2024) == pd.Timestamp("2024-11-29")
    assert calendar.easter_sunday(2024) == pd.Timestamp("2024-03-31")
    names = calendar.events(pd.date_range("2024-11-25", "2024-11-29"))
    assert list(names) == ["black_week"] * 4 + ["black_friday"]


def test_segments_add_up_to_the_shop(shop_facts):
    shop = shop_facts.series(Segment())
    cells = sum(shop_facts.series(Segment(c, ch))["sales"] for c in shop_facts.categories
                for ch in shop_facts.channels)
    assert np.allclose(shop["sales"], cells)
    by_channel = sum(shop_facts.series(Segment(channel=ch))["sessions"] for ch in shop_facts.channels)
    assert np.array_equal(shop["sessions"].to_numpy(), by_channel.to_numpy())
    assert shop["margin"].sum() == pytest.approx(shop["margin_sold"].sum() - shop["return_cost"].sum())


def test_series_is_a_copy(shop_facts):
    first = shop_facts.series(Segment())
    first["sales"] = 0.0
    assert shop_facts.series(Segment())["sales"].sum() > 0


def test_clip_uses_training_rows_only():
    values = pd.Series([1.0] * 98 + [50.0, 1000.0])
    groups = pd.Series(["a"] * 100)
    training = pd.Series([True] * 99 + [False])  # the 1000 comes after the training years
    clipped = clip(values, groups, training)
    assert clipped.iloc[-1] < 1000
    assert clipped.iloc[-1] == pytest.approx(values[training].quantile(0.99))


def test_one_order_of_a_repeat_customer_is_not_neutralized():
    day = pd.Timestamp("2024-01-02")
    lines = pd.DataFrame({"date": [day] * 3, "category": ["Toys"] * 3, "channel": ["direct"] * 3,
                          "order_id": [1, 2, 3], "customer_id": [9, 9, 5], "sales": [500.0, 400.0, 100.0],
                          "margin": [50.0, 40.0, 10.0], "margin_w": [50.0, 40.0, 10.0]})
    returns = pd.DataFrame(columns=["date", "category", "channel", "return_cost", "return_cost_w"])
    tops = largest_items(lines, returns)
    cell = tops[(tops["category"] == "Toys") & (tops["channel"] == "direct")].iloc[0]
    assert cell["top_order_sales"] == 100.0   # customer 9 ordered twice: left as it is


def test_load_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="get_data.py"):
        load(42, root=tmp_path)


def test_detection_starts_after_two_training_years():
    assert pd.Timestamp("2023-01-01") == DETECTION_START
