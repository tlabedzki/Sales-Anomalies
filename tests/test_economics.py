"""Line economics on a shop small enough to check by hand."""
import pandas as pd
import pytest

from sales_anomalies.data import TABLE_COLUMNS, ShopData
from sales_anomalies.economics import booked_returns, sold_lines


def tiny_shop() -> ShopData:
    """
    order 1: two lines (246 + 123 gross, the second 50% off), carrier 20, customer paid 12.30 gross delivery,
             payment fee 6; line 12 returned and refunded (61.50), 1 unit written off, label 8.50, handling 8
    order 2: cancelled (not sold)
    """
    products = pd.DataFrame({"product_id": [1, 2], "product_name": ["A", "B"], "category": ["Toys", "Toys"],
                             "net_purchase_price": [100.0, 40.0], "sales_price_gross": [246.0, 123.0]})
    orders = pd.DataFrame({
        "order_id": [1, 1, 2], "order_line_id": [11, 12, 21], "order_date": pd.to_datetime(["2024-01-02"] * 3),
        "product_id": [1, 2, 1], "quantity": [1, 1, 1], "unit_price_gross": [246.0, 123.0, 246.0],
        "customer_id": [7, 7, 8], "payment_method": ["BLIK"] * 3, "delivery_method": ["DPD"] * 3,
        "discount_rate": [0.0, 0.5, 0.0], "is_fraud": [False] * 3, "channel": ["direct"] * 3,
        "device": ["mobile"] * 3, "campaign_id": pd.array([pd.NA] * 3, dtype="Int64"), "session_id": [1, 1, 2],
        "order_status": ["delivered", "delivered", "cancelled"], "affiliate_commission": [0.0, 0.0, 0.0]})
    shipments = pd.DataFrame({"shipment_id": [1], "order_id": [1], "delivery_method": ["DPD"],
                              "shipping_cost": [12.3], "carrier_cost": [20.0]})
    payments = pd.DataFrame({"payment_id": [1, 2], "order_id": [1, 2], "payment_amount": [320.8, 0.0],
                             "fee_rate": [0.0, 0.0], "fixed_fee": [6.0, 0.0], "payment_fee": [6.0, 0.0]})
    returns = pd.DataFrame({"return_id": [1], "order_id": [1], "order_line_id": [12], "product_id": [2],
                            "return_date": pd.to_datetime(["2024-01-09"]), "returned_quantity": [1],
                            "return_reason": ["defective"], "refund_amount": [61.5], "return_status": ["refunded"],
                            "return_shipping_cost": [8.5], "handling_cost": [8.0], "written_off_quantity": [1],
                            "write_off_cost": [40.0]})
    empty = {name: pd.DataFrame(columns=columns) for name, columns in TABLE_COLUMNS.items()}
    return ShopData(products=products, orders=orders, shipments=shipments, returns=returns, payments=payments,
                    sessions=empty["session_data"], campaigns=empty["campaign_data"],
                    competitor_prices=empty["competitor_prices"], price_history=empty["price_history"],
                    anomalies=empty["anomaly_data"])


def test_sold_lines_split_order_costs_by_sales():
    lines = sold_lines(tiny_shop()).set_index("order_line_id")
    assert list(lines.index) == [11, 12]                      # the cancelled order is not sold
    assert lines.loc[11, "sales"] == pytest.approx(200.0)      # 246 / 1.23
    assert lines.loc[12, "discount"] == pytest.approx(50.0)    # 123 / 1.23 * 50%
    assert lines.loc[12, "sales"] == pytest.approx(50.0)
    # shipping = carrier 20 - delivery fee 10 net = 10, split 200:50; payment fee 6 split the same way
    assert lines.loc[11, "shipping"] == pytest.approx(8.0)
    assert lines.loc[12, "shipping"] == pytest.approx(2.0)
    assert lines.loc[11, "payment"] == pytest.approx(4.8)
    assert lines.loc[11, "margin"] == pytest.approx(200 - 100 - 8 - 4.8)
    assert lines["shipping"].sum() == pytest.approx(10.0)


def test_return_cost_counts_refund_back_on_sale_label_and_handling():
    back = booked_returns(tiny_shop()).iloc[0]
    assert back["refund"] == pytest.approx(50.0)               # 61.50 / 1.23
    assert back["cogs_back"] == pytest.approx(0.0)             # the only unit was written off
    assert back["return_cost"] == pytest.approx(50.0 + 8.5 + 8.0)
    assert back["date"] == pd.Timestamp("2024-01-09")
