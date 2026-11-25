"""
Alert thresholds chosen on a second shop (seed 7) without injected anomalies, then used unchanged on seed 42.

For every group (level and density, metric, direction, scale) the threshold is the lowest z on a grid at which the
group raises at most BUDGET false episodes over the two detection years. A false episode is one that no event of
seed 7's own anomaly_data explains, by the same test the grading uses (evaluate.explains). The budget is the
manager's attention: how many false alarms a month the shop accepts. Choosing the thresholds on another seed keeps
the scores on seed 42 honest: nothing is tuned on the data it is graded on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sales_anomalies.detect import SCALES, THRESHOLDS_FILE, episodes, flags, group_key
from sales_anomalies.evaluate import explains
from sales_anomalies.truth import TruthEvent

# Never below 4 standard deviations: a single shop or channel series is too little to read its tail from one seed,
# and with about 100,000 tests a year a z of 4 still passes by chance a few times a year.
GRID = np.round(np.arange(4.0, 20.01, 0.25), 2)
BUDGET = 0


def _false(table: pd.DataFrame, flagged: pd.DataFrame, thresholds: dict[str, float],
           events: list[TruthEvent]) -> int:
    found = episodes(table, flagged, thresholds)
    return sum(not any(explains(e, episode) for e in events) for episode in found)


def calibrate(table: pd.DataFrame, events: list[TruthEvent], budget: int = BUDGET) -> dict[str, dict]:
    """
    Threshold and false-episode count of every group. Each scale is tuned with the other one switched off, on
    the same episodes and the same evidence test as the grading (evaluate.explains).
    """
    result = {}
    floor = {}
    for (group, metric), rows in table.groupby(["group", "metric"]):
        for direction in ("up", "down"):
            keys = {scale: group_key(group, metric, direction, scale) for scale in SCALES}
            floor = {key: float(GRID[0]) for key in keys.values()}
            candidates = flags(rows, floor)
            candidates = candidates[candidates["direction"] == direction]
            sign = 1 if direction == "up" else -1
            for scale in SCALES:
                windows = candidates[candidates["scale"] == scale]
                other = {k: float("inf") for s_, k in keys.items() if s_ != scale}
                chosen, count = float(GRID[-1]), 0
                for limit in GRID:
                    subset = windows[sign * windows["z"] >= limit]
                    count = _false(rows, subset, {**other, keys[scale]: float(limit)}, events)
                    if count <= budget:
                        chosen = float(limit)
                        break
                result[keys[scale]] = {"threshold": chosen, "false": count}
    return result


def save(groups: dict[str, dict], seed: int, path: Path = THRESHOLDS_FILE) -> None:
    payload = {
        "seed": seed, "budget_per_group": BUDGET,
        "thresholds": {k: v["threshold"] for k, v in sorted(groups.items())},
        "false_episodes": {k: v["false"] for k, v in sorted(groups.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
