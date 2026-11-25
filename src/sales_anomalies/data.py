"""The shop tables made by Data-Generator (scripts/get_data.py): where they live, which columns, how to load them."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DATA_GENERATOR_TAG = "v1.7.1"
REPORT_SEED = 42
CALIBRATION_SEED = 7
SEEDS = (REPORT_SEED, CALIBRATION_SEED)
VAT_RATE = 0.23

# Lumo about ten times the default run: four years (the first two only teach the model), 1,060,000 order lines,
# about 730 a day, the size of a mid-sized online shop.
GENERATOR_SETTINGS = {
    "start_year": 2021,
    "end_year": 2024,
    "order_data_rows": 1_060_000,
    "customer_data_rows": 400_000,
    "product_data_rows": 50_000,
    "data_language": "en",
}

TABLE_COLUMNS = {
    "product_data": ["product_id", "product_name", "category", "net_purchase_price", "sales_price_gross"],
    "order_data": ["order_id", "order_line_id", "order_date", "product_id", "quantity", "unit_price_gross",
                   "customer_id", "payment_method", "delivery_method", "discount_rate", "is_fraud", "channel",
                   "device", "campaign_id", "session_id", "order_status", "affiliate_commission"],
    "shipment_data": ["shipment_id", "order_id", "delivery_method", "shipping_cost", "carrier_cost"],
    "return_data": ["return_id", "order_id", "order_line_id", "product_id", "return_date", "returned_quantity",
                    "return_reason", "refund_amount", "return_status", "return_shipping_cost", "handling_cost",
                    "written_off_quantity", "write_off_cost"],
    "payment_data": ["payment_id", "order_id", "payment_amount", "fee_rate", "fixed_fee", "payment_fee"],
    "session_data": ["session_id", "session_start", "channel", "device", "campaign_id", "reached_checkout",
                     "order_id"],
    "campaign_data": ["campaign_id", "campaign_name", "channel", "start_date", "end_date", "total_cost"],
    "competitor_prices": ["product_id", "competitor_id", "competitor_name", "valid_from", "valid_to", "price_gross",
                          "is_promotion"],
    "price_history": ["product_id", "valid_from", "valid_to", "sales_price_gross"],
    "anomaly_data": ["anomaly_id", "anomaly_type", "start_date", "end_date", "table_name", "affected_rows",
                     "customer_id", "description"],
}

# ShopData attribute -> Data-Generator table.
ATTRIBUTE_TABLES = {
    "products": "product_data", "orders": "order_data", "shipments": "shipment_data", "returns": "return_data",
    "payments": "payment_data", "sessions": "session_data", "campaigns": "campaign_data",
    "competitor_prices": "competitor_prices", "price_history": "price_history", "anomalies": "anomaly_data",
}


@dataclass
class ShopData:
    """The shop tables in the Data-Generator layout (only the columns in TABLE_COLUMNS)."""

    products: pd.DataFrame
    orders: pd.DataFrame
    shipments: pd.DataFrame
    returns: pd.DataFrame
    payments: pd.DataFrame
    sessions: pd.DataFrame
    campaigns: pd.DataFrame
    competitor_prices: pd.DataFrame
    price_history: pd.DataFrame
    anomalies: pd.DataFrame
    source: dict = field(default_factory=dict)

    def copy(self) -> ShopData:
        tables = {f.name: getattr(self, f.name).copy() for f in fields(self) if f.name != "source"}
        return replace(self, **tables, source=dict(self.source))

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.orders["order_date"].min(), self.orders["order_date"].max()


def seed_dir(seed: int, root: Path = RAW_DIR) -> Path:
    return root / f"seed-{seed}"


def missing_data_message(directory: Path) -> str:
    return f"No shop data in {directory}. Run: python scripts/get_data.py"


def load(seed: int = REPORT_SEED, root: Path = RAW_DIR) -> ShopData:
    """Read the tables of one seed; a missing file or column raises an error that says what to run."""
    import pyarrow.parquet as pq

    directory = seed_dir(seed, root)
    tables = {}
    for attribute, table in ATTRIBUTE_TABLES.items():
        path = directory / f"{table}.parquet"
        if not path.exists():
            raise FileNotFoundError(missing_data_message(directory))
        missing = [c for c in TABLE_COLUMNS[table] if c not in pq.read_schema(path).names]
        if missing:
            raise ValueError(f"{table} has no column {', '.join(missing)}. Run: python scripts/get_data.py")
        tables[attribute] = pd.read_parquet(path, columns=TABLE_COLUMNS[table])
    source_path = directory / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    return ShopData(**tables, source=source)
