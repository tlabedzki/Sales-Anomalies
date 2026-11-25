"""
One run of the daily check over the whole period, with its grading against the answer key.

    shop (Data-Generator) -> scenarios injected -> daily facts -> expected values -> scores -> anomalies
    -> breakdown of each anomaly into causes -> grading against the scenarios and anomaly_data

The negative control repeats the check on the same shop without any scenario: everything it raises that
anomaly_data does not explain is a false alarm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sales_anomalies import decompose, detect, evaluate, facts, scenarios, truth
from sales_anomalies.data import REPORT_SEED, ROOT, ShopData, load

OUTPUT_DIR = ROOT / "output"


@dataclass
class Run:
    seed: int
    shop: ShopData                   # after the scenarios
    before: facts.ShopFacts          # facts without the scenarios
    after: facts.ShopFacts           # facts with them
    plan: list[scenarios.Scenario]
    events: list[truth.TruthEvent]
    detection: detect.Detection
    grades: evaluate.Grades
    breakdowns: list[decompose.Breakdown]
    control: evaluate.Grades | None = None
    summary: dict = field(default_factory=dict)


def cause_accuracy(grades: evaluate.Grades, breakdowns: list[decompose.Breakdown],
                   events: list[truth.TruthEvent]) -> pd.DataFrame:
    """Per anomaly that is a hit: the main cause found and the causes of the events it matched."""
    by_id = {e.event_id: e for e in events}
    main = {b.anomaly_id: b.main_cause for b in breakdowns}
    rows = []
    for row in grades.anomalies[grades.anomalies["verdict"] != "false"].itertuples():
        causes = sorted({by_id[i].cause for i in row.events.split(",") if i})
        rows.append({"anomaly_id": row.anomaly_id, "found": main[row.anomaly_id], "truth": ",".join(causes),
                     "right": main[row.anomaly_id] in causes, "source": row.verdict})
    return pd.DataFrame(rows, columns=["anomaly_id", "found", "truth", "right", "source"])


def run(seed: int = REPORT_SEED, control: bool = True) -> Run:
    shop = load(seed)
    plan = scenarios.plan(shop)
    changed = scenarios.inject(shop, plan)
    before, after = facts.build(shop), facts.build(changed)
    events = scenarios.truth(plan, before, after) + truth.dg_events(changed, after)
    detection = detect.detect(after)
    end = after.days[-1]
    grades = evaluate.grade(detection.anomalies, events, detect.DETECTION_START, end, (before, after))
    breakdowns = decompose.explain_all(detection.anomalies, changed, after, detection)
    result = Run(seed, changed, before, after, plan, events, detection, grades, breakdowns)
    if control:
        clean = detect.detect(before)
        result.control = evaluate.grade(clean.anomalies, truth.dg_events(shop, before), detect.DETECTION_START, end)
    result.summary = summarize(result)
    return result


def _recall(events: pd.DataFrame) -> dict:
    by_kind = events.groupby("kind")["found"].agg(["sum", "size"])
    return {kind: {"found": int(row["sum"]), "events": int(row["size"])} for kind, row in by_kind.iterrows()}


def summarize(r: Run) -> dict:
    g = r.grades
    accuracy = cause_accuracy(g, r.breakdowns, r.events)
    exact = sum(sum(p.grosz for p in b.parts) == b.change_grosz for b in r.breakdowns)
    scenario_events = g.events[g.events["source"] == "scenario"]
    dg_events = g.events[g.events["source"] == "anomaly_data"]
    summary = {
        "seed": r.seed,
        "period": [r.after.days[0].strftime("%Y-%m-%d"), r.after.days[-1].strftime("%Y-%m-%d")],
        "detection_start": detect.DETECTION_START.strftime("%Y-%m-%d"),
        "months": round(g.months, 2),
        "anomalies": len(g.anomalies),
        "hits": int((g.anomalies["verdict"] != "false").sum()),
        "false_alarms": g.false_alarms(),
        "false_per_month": round(g.false_per_month(), 3),
        "precision": round(g.precision(), 4),
        "recall_scenarios": round(g.recall("scenario"), 4),
        "scenarios_found": int(scenario_events["found"].sum()),
        "scenarios_total": len(scenario_events),
        "segments": len(r.after.segments()),
        "recall_anomaly_data": round(g.recall("anomaly_data"), 4),
        "scenarios": _recall(scenario_events),
        "anomaly_data": _recall(dg_events),
        "negative_period": g.negative_period(),
        "breakdowns": len(r.breakdowns),
        "breakdowns_exact_to_grosz": exact,
        "main_cause_right": int(accuracy["right"].sum()),
        "main_cause_graded": len(accuracy),
        "main_cause_right_scenarios": int(accuracy.loc[accuracy["source"] == "scenario", "right"].sum()),
        "main_cause_graded_scenarios": int((accuracy["source"] == "scenario").sum()),
    }
    if r.control is not None:
        summary["control"] = {"anomalies": len(r.control.anomalies), "false_alarms": r.control.false_alarms(),
                              "false_per_month": round(r.control.false_per_month(), 3)}
    return summary


def save(r: Run, directory: Path = OUTPUT_DIR) -> None:
    """Results for the app and the README: anomalies, breakdowns, truth events, the scores of each segment."""
    directory.mkdir(parents=True, exist_ok=True)
    r.grades.anomalies.to_csv(directory / "anomalies.csv", index=False)
    r.grades.events.to_csv(directory / "events.csv", index=False)
    cause_accuracy(r.grades, r.breakdowns, r.events).to_csv(directory / "causes.csv", index=False)
    (directory / "breakdowns.json").write_text(
        json.dumps([b.as_dict() for b in r.breakdowns], indent=1, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(r.summary, indent=2, default=str) + "\n", encoding="utf-8")
    r.detection.table.to_parquet(directory / "scores.parquet", index=False)
    plan = pd.DataFrame([{**scenarios.as_record(s), "description": s.describe()} for s in r.plan])
    plan.to_csv(directory / "scenarios.csv", index=False)
