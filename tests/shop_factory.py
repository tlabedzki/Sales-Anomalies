"""
A small synthetic shop in the Data-Generator layout for the tests (no data files): three years, two categories,
two channels, about 30 orders a day, a social campaign every two months, one competitor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sales_anomalies.data import TABLE_COLUMNS, ShopData

START, END = "2021-01-01", "2023-12-31"
CATEGORIES = {"Clothing": (100, 200), "Electronics": (500, 1500)}
CHANNELS = {"direct": 0.6, "social": 0.4}


def make_shop(seed: int = 1, orders_per_day: float = 30.0) -> ShopData:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D")
    products = []
    for number, (category, (low, high)) in enumerate(
            [(c, r) for c, r in CATEGORIES.items() for _ in range(20)]):
        gross = float(rng.integers(low, high) // 10 * 10 + 9)
        products.append({"product_id": 200001 + number, "product_name": f"{category} {number}",
                         "category": category, "net_purchase_price": round(gross / 1.23 * 0.6, 2),
                         "sales_price_gross": gross})
    products = pd.DataFrame(products)
    clothing = products[products["category"] == "Clothing"]["product_id"].to_numpy()
    electronics = products[products["category"] == "Electronics"]["product_id"].to_numpy()
    price = products.set_index("product_id")["sales_price_gross"]

    campaigns = []
    for number, start in enumerate(pd.date_range("2021-02-01", END, freq="2MS"), start=1):
        campaigns.append({"campaign_id": number, "campaign_name": f"Social {start:%Y-%m-%d}", "channel": "social",
                          "start_date": start, "end_date": start + pd.Timedelta(days=6), "total_cost": 1000.0})
    campaigns = pd.DataFrame(campaigns)
    running = np.zeros(len(days), dtype=bool)
    running_id = np.zeros(len(days), dtype=int)
    for row in campaigns.itertuples():
        on = (days >= row.start_date) & (days <= row.end_date)
        running |= on
        running_id[on] = row.campaign_id

    weekday = np.array([1.1, 1.0, 1.0, 1.0, 0.95, 0.9, 1.05])
    growth = 1 + 0.15 * (days.year - 2021)
    rows, sessions = [], []
    order_id, line_id, session_id = 100001, 1000001, 1
    for index, day in enumerate(days):
        social_lift = 1.3 if running[index] else 1.0
        for channel, share in CHANNELS.items():
            lam = orders_per_day * share * weekday[day.dayofweek] * growth[index] * (
                social_lift if channel == "social" else 1.0)
            for _ in range(rng.poisson(lam)):
                category_pick = rng.random() < 0.7
                product = int(rng.choice(clothing if category_pick else electronics))
                device = "mobile" if rng.random() < 0.6 else "desktop"
                cancelled = rng.random() < 0.04
                rows.append({"order_id": order_id, "order_line_id": line_id, "order_date": day,
                             "product_id": product, "quantity": 1, "unit_price_gross": float(price[product]),
                             "customer_id": int(rng.integers(1, 5000)), "payment_method": "BLIK",
                             "delivery_method": "InPost parcel locker" if rng.random() < 0.7 else "DPD",
                             "discount_rate": 0.1 if rng.random() < 0.2 else 0.0, "is_fraud": False,
                             "channel": channel, "device": device,
                             "campaign_id": int(running_id[index]) if channel == "social" and running[index]
                             else pd.NA,
                             "session_id": session_id, "order_status": "cancelled" if cancelled else "delivered",
                             "affiliate_commission": 0.0})
                sessions.append({"session_id": session_id, "session_start": day + pd.Timedelta(hours=12),
                                 "channel": channel, "device": device,
                                 "campaign_id": rows[-1]["campaign_id"], "reached_checkout": True,
                                 "order_id": order_id})
                session_id += 1
                order_id += 1
                line_id += 1
            for _ in range(rng.poisson(lam * 4)):
                sessions.append({"session_id": session_id, "session_start": day + pd.Timedelta(hours=10),
                                 "channel": channel, "device": "mobile" if rng.random() < 0.6 else "desktop",
                                 "campaign_id": int(running_id[index]) if channel == "social" and running[index]
                                 else pd.NA,
                                 "reached_checkout": False, "order_id": pd.NA})
                session_id += 1
    orders = pd.DataFrame(rows)
    orders["campaign_id"] = orders["campaign_id"].astype("Int64")
    sessions = pd.DataFrame(sessions)
    sessions["campaign_id"] = sessions["campaign_id"].astype("Int64")
    sessions["order_id"] = sessions["order_id"].astype("Int64")

    sold = orders[orders["order_status"] == "delivered"]
    net_after = sold["unit_price_gross"] * (1 - sold["discount_rate"])
    shipments = pd.DataFrame({"shipment_id": np.arange(len(sold)) + 1, "order_id": sold["order_id"].to_numpy(),
                              "delivery_method": sold["delivery_method"].to_numpy(),
                              "shipping_cost": np.where(net_after.to_numpy() >= 150, 0.0, 12.3),
                              "carrier_cost": np.where(sold["delivery_method"] == "DPD", 14.0, 10.0)})
    payment_amount = (orders["unit_price_gross"] * (1 - orders["discount_rate"])).to_numpy()
    payments = pd.DataFrame({"payment_id": np.arange(len(orders)) + 1, "order_id": orders["order_id"].to_numpy(),
                             "payment_amount": payment_amount.round(2), "fee_rate": 0.012, "fixed_fee": 0.3})
    payments["payment_fee"] = (payments["payment_amount"] * payments["fee_rate"] + payments["fixed_fee"]).round(2)

    back = sold[rng.random(len(sold)) < 0.1]
    cost = back["product_id"].map(products.set_index("product_id")["net_purchase_price"])
    returns = pd.DataFrame({
        "return_id": np.arange(len(back)) + 1, "order_id": back["order_id"].to_numpy(),
        "order_line_id": back["order_line_id"].to_numpy(), "product_id": back["product_id"].to_numpy(),
        "return_date": (back["order_date"] + pd.Timedelta(days=5)).to_numpy(), "returned_quantity": 1,
        "return_reason": np.where(rng.random(len(back)) < 0.7, "wrong_size", "defective"),
        "refund_amount": (back["unit_price_gross"] * (1 - back["discount_rate"])).round(2).to_numpy(),
        "return_status": "refunded", "return_shipping_cost": 8.5, "handling_cost": 8.0, "written_off_quantity": 0,
        "write_off_cost": 0.0})
    returns["write_off_cost"] = returns["write_off_cost"] * cost.to_numpy()

    competitor_prices = pd.DataFrame({"product_id": products["product_id"], "competitor_id": 1,
                                      "competitor_name": "Frugalia", "valid_from": pd.Timestamp(START),
                                      "valid_to": pd.NaT, "price_gross": products["sales_price_gross"] * 0.95,
                                      "is_promotion": False})
    price_history = pd.DataFrame({"product_id": products["product_id"], "valid_from": pd.Timestamp(START),
                                  "valid_to": pd.NaT, "sales_price_gross": products["sales_price_gross"]})
    anomalies = pd.DataFrame(columns=TABLE_COLUMNS["anomaly_data"])
    return ShopData(products=products, orders=orders, shipments=shipments, returns=returns, payments=payments,
                    sessions=sessions, campaigns=campaigns, competitor_prices=competitor_prices,
                    price_history=price_history, anomalies=anomalies)
