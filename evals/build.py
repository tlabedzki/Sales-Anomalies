"""
Build evals/cases.json: sheets of facts from the anomalies of the last run (one per metric and cause, the biggest
first) plus hand-made edge cases (tiny change, a loss bigger than the expected value, an outage, many causes of
about the same size). Rerun after python main.py when the anomalies change.

    python -m evals.build
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sales_anomalies import explain, report  # noqa: E402
from sales_anomalies.decompose import Breakdown, Part  # noqa: E402

CASES = ROOT / "evals" / "cases.json"
FROM_RUN = 14


def edge_cases() -> list[Breakdown]:
    return [
        Breakdown(9001, "Office|affiliate", "margin", "down", "2024-04-03", "2024-04-03", 412.0, 1650.0, -123800,
                  [Part("discount", "discount", -1100.0, grosz=-110000), Part("prices", "prices", -138.0,
                                                                                 grosz=-13800)],
                  "discount", {"lines": 4.0, "expected_lines": 5.0, "discount_rate": 0.41,
                               "expected_discount_rate": 0.06}),
        Breakdown(9002, "*|*", "sales", "down", "2024-08-30", "2024-08-30", 31250.5, 512980.25, -48172975,
                  [Part("volume_traffic", "outage", -455300.0, grosz=-45530000),
                   Part("volume_conversion", "outage", -24000.0, grosz=-2400000),
                   Part("prices", "prices", -2429.75, grosz=-242975)], "outage",
                  {"sessions": 380.0, "expected_sessions": 4710.0, "lines": 61.0, "expected_lines": 790.0}),
        Breakdown(9003, "Clothing|*", "returns", "up", "2023-07-15", "2023-07-28", 98120.0, 41200.0, 5692000,
                  [Part("returns_defective", "returns", 50100.0, grosz=5010000),
                   Part("returns_wrong_size", "returns", 4100.0, grosz=410000),
                   Part("returns_cost_each", "returns", 2720.0, grosz=272000)], "returns",
                  {"returns": 1210.0, "expected_returns": 520.0, "top_reason": "defective"}),
        Breakdown(9004, "*|social", "margin", "down", "2024-02-01", "2024-02-05", 15020.0, 23880.0, -886000,
                  [Part("volume_traffic", "campaign", -3100.0, grosz=-310000),
                   Part("discount", "discount", -2900.0, grosz=-290000),
                   Part("prices", "prices", -2860.0, grosz=-286000)], "campaign",
                  {"sessions": 9100.0, "expected_sessions": 12800.0, "campaigns": ["Social 2024-01-31"],
                   "lines": 410.0, "expected_lines": 530.0}),
        Breakdown(9005, "Electronics|direct", "sales", "up", "2023-05-10", "2023-05-12", 94500.0, 41000.0, 5350000,
                  [Part("single_account", "single_account", 47800.0, grosz=4780000),
                   Part("prices", "prices", 5700.0, grosz=570000)], "single_account",
                  {"account": {"customer_id": 123}, "lines": 61.0, "expected_lines": 58.0}),
    ]


def main() -> None:
    r = report.load()
    chosen, seen = [], set()
    for b in sorted(r.breakdowns.values(), key=lambda b: -abs(b.change)):
        key = (b.metric, b.main_cause)
        if key not in seen:
            chosen.append(b)
            seen.add(key)
    for b in sorted(r.breakdowns.values(), key=lambda b: -abs(b.change)):
        if len(chosen) >= FROM_RUN:
            break
        if b not in chosen:
            chosen.append(b)
    cases = []
    for number, b in enumerate([*chosen[:FROM_RUN], *edge_cases()], start=1):
        cases.append({"id": f"c{number:02d}", "anomaly": explain.anomaly_key(b), "main_cause": b.main_cause,
                      "direction": "down" if b.change < 0 else "up",
                      "sheets": {lang: explain.facts(b, lang) for lang in ("pl", "en")}})
    CASES.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(cases)} cases to {CASES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
