"""
What every sold order line and every return is worth to the shop, in PLN without VAT (SQL in DuckDB).

A sold line belongs to an order that was not cancelled and whose payment did not fail. Its contribution margin is

    sales (list price after discount) - purchase cost - shipping - payment fee - affiliate commission

where shipping is what the carrier charges the shop minus the delivery fee the customer paid, and the shipping and
the payment fee of an order are split over its lines by their sales. A return is booked on its return day:

    return cost = refund - purchase cost of the units that go back on sale + return label + handling

so the margin of a day is the margin of the lines sold that day minus the cost of the returns booked that day.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from sales_anomalies.data import VAT_RATE, ShopData

LINE_VALUES = ("list_net", "discount", "sales", "cogs", "shipping", "payment", "commission", "margin")
RETURN_VALUES = ("refund", "cogs_back", "return_label", "handling", "return_cost")

LINES_SQL = f"""
WITH sold AS (
    SELECT o.*, p.category, p.net_purchase_price
    FROM orders o JOIN products p USING (product_id)
    WHERE o.order_status NOT IN ('cancelled', 'payment_failed')
),
priced AS (
    SELECT *,
        quantity * unit_price_gross / (1 + {VAT_RATE}) AS list_net,
        quantity * unit_price_gross / (1 + {VAT_RATE}) * discount_rate AS discount,
        quantity * unit_price_gross / (1 + {VAT_RATE}) * (1 - discount_rate) AS sales,
        quantity * net_purchase_price AS cogs
    FROM sold
),
shares AS (
    SELECT *,
        CASE WHEN SUM(sales) OVER (PARTITION BY order_id) > 0
             THEN sales / SUM(sales) OVER (PARTITION BY order_id)
             ELSE 1.0 / COUNT(*) OVER (PARTITION BY order_id) END AS order_share
    FROM priced
),
order_shipping AS (
    SELECT order_id, SUM(carrier_cost - shipping_cost / (1 + {VAT_RATE})) AS shipping
    FROM shipments GROUP BY order_id
),
order_payment AS (
    SELECT order_id, SUM(payment_fee) AS payment FROM payments GROUP BY order_id
)
SELECT
    s.order_line_id, s.order_id, s.order_date AS date, s.product_id, s.customer_id, s.category, s.channel,
    s.device, s.delivery_method, s.campaign_id, s.is_fraud, s.quantity,
    s.list_net, s.discount, s.sales, s.cogs,
    COALESCE(sh.shipping, 0) * s.order_share AS shipping,
    COALESCE(pa.payment, 0) * s.order_share AS payment,
    s.affiliate_commission AS commission
FROM shares s
LEFT JOIN order_shipping sh USING (order_id)
LEFT JOIN order_payment pa USING (order_id)
ORDER BY s.order_line_id
"""

RETURNS_SQL = f"""
SELECT
    r.return_id, r.order_line_id, r.order_id, r.return_date AS date, r.product_id, o.customer_id, p.category,
    o.channel, r.return_reason AS reason, r.returned_quantity,
    CASE WHEN r.return_status = 'refunded' THEN r.refund_amount / (1 + {VAT_RATE}) ELSE 0 END AS refund,
    CASE WHEN r.return_status = 'refunded'
         THEN (r.returned_quantity - r.written_off_quantity) * p.net_purchase_price ELSE 0 END AS cogs_back,
    r.return_shipping_cost AS return_label,
    r.handling_cost AS handling
FROM returns r
JOIN orders o USING (order_line_id)
JOIN products p ON p.product_id = r.product_id
ORDER BY r.return_id
"""


def _query(sql: str, **frames: pd.DataFrame) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        for name, frame in frames.items():
            connection.register(name, frame)
        return connection.execute(sql).df()
    finally:
        connection.close()


def sold_lines(shop: ShopData) -> pd.DataFrame:
    """One row per sold order line with its money in PLN without VAT and its contribution margin."""
    lines = _query(LINES_SQL, orders=shop.orders, products=shop.products, shipments=shop.shipments,
                   payments=shop.payments)
    lines["margin"] = lines["sales"] - lines["cogs"] - lines["shipping"] - lines["payment"] - lines["commission"]
    lines["date"] = pd.to_datetime(lines["date"])
    return lines


def booked_returns(shop: ShopData) -> pd.DataFrame:
    """One row per return, booked on its return day, with its cost to the shop."""
    returns = _query(RETURNS_SQL, returns=shop.returns, orders=shop.orders, products=shop.products)
    returns["return_cost"] = returns["refund"] - returns["cogs_back"] + returns["return_label"] + returns["handling"]
    returns["date"] = pd.to_datetime(returns["date"])
    return returns
