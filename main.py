"""
The daily anomaly check of the fictional shop Lumo over 2023-2024, graded against the answer key.

    python main.py              # seed 42 with the injected scenarios and the negative control; writes output/
    python main.py --no-control # skip the run without scenarios (about half the time)

Run python scripts/get_data.py first.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sales_anomalies import notes, pipeline  # noqa: E402
from sales_anomalies.data import REPORT_SEED, missing_data_message, seed_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=REPORT_SEED)
    parser.add_argument("--no-control", action="store_true", help="skip the negative control run")
    args = parser.parse_args()
    if not (seed_dir(args.seed) / "order_data.parquet").exists():
        print(missing_data_message(seed_dir(args.seed)), file=sys.stderr)
        return 1

    started = time.time()
    run = pipeline.run(args.seed, control=not args.no_control)
    pipeline.save(run)
    written = notes.notes(run.breakdowns)
    s = run.summary
    print(f"Lumo, seed {s['seed']}, {s['detection_start']} to {s['period'][1]} ({time.time() - started:.0f} s)")
    print(f"  anomalies {s['anomalies']}: {s['hits']} real, {s['false_alarms']} false "
          f"({s['false_per_month']:.2f} a month), precision {s['precision']:.0%}")
    print(f"  recall: scenarios {s['recall_scenarios']:.0%}, anomaly_data {s['recall_anomaly_data']:.0%}")
    print(f"  negative period: {s['negative_period']['false']} false alarms in "
          f"{s['negative_period']['months']:.0f} months")
    if "control" in s:
        print(f"  without scenarios: {s['control']['false_alarms']} false alarms "
              f"({s['control']['false_per_month']:.2f} a month)")
    print(f"  breakdowns exact to the grosz: {s['breakdowns_exact_to_grosz']} of {s['breakdowns']}")
    print(f"  main cause right: {s['main_cause_right']} of {s['main_cause_graded']}")
    print(f"  notes for the manager: {len(written)} (Polish and English)")
    if s["breakdowns_exact_to_grosz"] != s["breakdowns"]:
        print("Some breakdowns do not add up to the change", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
