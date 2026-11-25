"""
Choose the alert thresholds on the second shop (seed 7, no injected scenario) and save them to
src/sales_anomalies/thresholds.json, which main.py then uses unchanged on seed 42.

    python scripts/calibrate.py           # about 4 minutes
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sales_anomalies import calibrate, detect, facts, truth  # noqa: E402
from sales_anomalies.data import CALIBRATION_SEED, load  # noqa: E402


def main() -> None:
    started = time.time()
    shop = load(CALIBRATION_SEED)
    shop_facts = facts.build(shop)
    table = detect.scores(shop_facts, detect.expectations(shop_facts))
    groups = calibrate.calibrate(table, truth.dg_events(shop, shop_facts))
    calibrate.save(groups, CALIBRATION_SEED)
    false = sum(v["false"] for v in groups.values()) // len(detect.SCALES)
    print(f"Saved {len(groups)} thresholds (seed {CALIBRATION_SEED}, {false} false episodes left, "
          f"{time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
