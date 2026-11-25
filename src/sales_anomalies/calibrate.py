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

from sales_anomalies.detect import SCALES, THRESHOLDS_FILE, by_segment, episodes, flags, group_key
from sales_anomalies.evaluate import explains
from sales_anomalies.truth import TruthEvent

# Never below 4 standard deviations: a single shop or channel series is too little to read its tail from one seed,
# and with about 100,000 tests a year a z of 4 still passes by chance a few times a year.
GRID = np.round(np.arange(4.0, 20.01, 0.25), 2)
BUDGET = 0


def _false(table: pd.DataFrame, flagged: pd.DataFrame, thresholds: dict[str, float],
           events: list[TruthEvent], frames: dict) -> int:
    found = episodes(table, flagged, thresholds, frames)
    return sum(not any(explains(e, episode) for e in events) for episode in found)


def calibrate(table: pd.DataFrame, events: list[TruthEvent], budget: int = BUDGET) -> dict[str, dict]:
    """
    Threshold and false-episode count of every group, on the same episodes and the same evidence test as the
    grading (evaluate.explains). Each scale is tuned first with the other one switched off; then both are on
    together and raised in turn (the weekly one first) until the group keeps within the budget, since a one-day
    flag can grow into a longer run that the weekly threshold judges.
    """
    result = {}
    for (group, metric), rows in table.groupby(["group", "metric"]):
        frames = by_segment(rows)
        for direction in ("up", "down"):
            keys = {scale: group_key(group, metric, direction, scale) for scale in SCALES}
            candidates = flags(rows, {key: float(GRID[0]) for key in keys.values()})
            candidates = candidates[candidates["direction"] == direction]
            sign = 1 if direction == "up" else -1

            def false_with(limits: dict[int, float], candidates=candidates, sign=sign, keys=keys, frames=frames,
                           rows=rows) -> int:
                keep = pd.Series(False, index=candidates.index)
                for scale, limit in limits.items():
                    keep |= (candidates["scale"] == scale) & (sign * candidates["z"] >= limit)
                return _false(rows, candidates[keep], {keys[s]: v for s, v in limits.items()}, events, frames)

            chosen = {}
            for scale in SCALES:
                chosen[scale] = float(GRID[-1])
                for limit in GRID:
                    off = {s: float("inf") for s in SCALES if s != scale}
                    if false_with({**off, scale: float(limit)}) <= budget:
                        chosen[scale] = float(limit)
                        break
            count = false_with(chosen)
            turn = 0
            while count > budget and min(chosen.values()) < GRID[-1]:
                scale = SCALES[-1 - turn % len(SCALES)]
                chosen[scale] = min(float(GRID[-1]), chosen[scale] + 0.25)
                count = false_with(chosen)
                turn += 1
            for scale in SCALES:
                result[keys[scale]] = {"threshold": chosen[scale], "false": count}
    return result


def save(groups: dict[str, dict], seed: int, path: Path = THRESHOLDS_FILE) -> None:
    payload = {
        "seed": seed, "budget_per_group": BUDGET,
        "thresholds": {k: v["threshold"] for k, v in sorted(groups.items())},
        "false_episodes": {k: v["false"] for k, v in sorted(groups.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
