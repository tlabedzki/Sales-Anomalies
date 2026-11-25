"""
Results for the app and the README: read output/ (written by main.py) and the saved notes, and shape them into rows,
titles and numbers. One place, so the app, the charts and the README say the same.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from sales_anomalies import explain, i18n
from sales_anomalies.decompose import Breakdown, Part
from sales_anomalies.pipeline import OUTPUT_DIR

VERDICTS = {"scenario": "injected", "anomaly_data": "labelled by the generator", "false": "false alarm"}
KIND_LABELS = {
    "discount": "Extra discount in one category and channel", "price_error": "Best sellers priced too low",
    "return_wave": "Faulty batch comes back", "carrier_surcharge": "Carrier charges more",
    "competitor_undercut": "Competitor cuts prices", "campaign_stop": "Running campaign stops",
    "checkout_failure": "Checkout fails on phones", "seo_drop": "Organic search visits fall",
    "commission_error": "Affiliate charges more commission", "fraud_ring": "Fraud ring (stolen account)",
    "shop_outage": "Shop down for a day", "return_spike": "Faulty batch of one product",
}


def breakdown_from_dict(data: dict) -> Breakdown:
    values = {k: v for k, v in data.items() if k != "change"}
    values["parts"] = [Part(**part) for part in data["parts"]]
    return Breakdown(**values)


@dataclass
class Results:
    summary: dict
    anomalies: pd.DataFrame
    events: pd.DataFrame
    causes: pd.DataFrame
    breakdowns: dict[int, Breakdown]
    directory: Path

    @cached_property
    def scores(self) -> pd.DataFrame:
        return pd.read_parquet(self.directory / "scores.parquet")

    @cached_property
    def notes(self) -> dict:
        return explain.describe(list(self.breakdowns.values()), live=False)

    def note(self, anomaly_id: int, lang: str = "en") -> dict:
        return self.notes[f"{explain.anomaly_key(self.breakdowns[anomaly_id])}|{lang}"]

    def row(self, anomaly_id: int) -> pd.Series:
        return self.anomalies.set_index("anomaly_id").loc[anomaly_id]

    def events_of(self, anomaly_id: int) -> pd.DataFrame:
        ids = str(self.row(anomaly_id)["events"] or "")
        return self.events[self.events["event_id"].isin([i for i in ids.split(",") if i])]


def missing_message(directory: Path = OUTPUT_DIR) -> str:
    return f"No results in {directory}. Run: python scripts/get_data.py, then python main.py"


def load(directory: Path = OUTPUT_DIR) -> Results:
    if not (directory / "summary.json").exists():
        raise FileNotFoundError(missing_message(directory))
    anomalies = pd.read_csv(directory / "anomalies.csv", parse_dates=["start", "end"]).fillna({"events": ""})
    events = pd.read_csv(directory / "events.csv", parse_dates=["start", "end"]).fillna({"anomalies": ""})
    breakdowns = {d["anomaly_id"]: breakdown_from_dict(d)
                  for d in json.loads((directory / "breakdowns.json").read_text(encoding="utf-8"))}
    causes = pd.read_csv(directory / "causes.csv")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return Results(summary, anomalies, events, causes, breakdowns, directory)


# ---------------------------------------------------------------------------------------------------------------
# Rows and labels (English: the app)


def hurts(metric: str, change: float) -> bool:
    """A change that costs the shop money: less sales, margin or conversion, more returns or costs."""
    return change > 0 if metric in ("returns", "costs") else change < 0


def change_percent(b: Breakdown) -> float:
    return b.change / b.expected if b.expected else float("nan")


def anomaly_table(r: Results) -> pd.DataFrame:
    """One row per anomaly for the app's table, newest first."""
    rows = []
    for a in r.anomalies.itertuples():
        b = r.breakdowns[a.anomaly_id]
        rows.append({
            "anomaly_id": a.anomaly_id, "days": i18n.period(a.start, a.end, "en"),
            "segment": i18n.segment(a.segment, "en"), "metric": i18n.METRICS["en"][b.metric],
            "change": b.change, "change_percent": change_percent(b),
            "cause": i18n.CAUSES["en"][b.main_cause], "verdict": VERDICTS[a.verdict], "start": a.start,
            "hurts": hurts(b.metric, b.change),
        })
    return pd.DataFrame(rows).sort_values("start", ascending=False).reset_index(drop=True)


def default_anomaly(r: Results) -> int:
    """The anomaly the page opens with: the biggest real one (by |change|) whose main cause was named right."""
    right = set(r.causes.loc[r.causes["right"], "anomaly_id"])
    candidates = [b for i, b in r.breakdowns.items() if i in right] or list(r.breakdowns.values())
    return max(candidates, key=lambda b: abs(b.change)).anomaly_id


def truth_line(r: Results, anomaly_id: int) -> str:
    """What the answer key says about an anomaly, in one sentence (English)."""
    row = r.row(anomaly_id)
    if row["verdict"] == "false":
        return "Answer key: no injected scenario or labelled event explains it, so it counts as a false alarm."
    events = r.events_of(anomaly_id)
    first = events.iloc[0]
    what = first["description"] if first["source"] == "scenario" else KIND_LABELS.get(first["kind"], first["kind"])
    found = r.breakdowns[anomaly_id].main_cause
    verdict = "named right" if found == first["cause"] else f"named {i18n.CAUSES['en'][found]}, truth is " \
                                                             f"{i18n.CAUSES['en'][first['cause']]}"
    more = f" (+{len(events) - 1} more)" if len(events) > 1 else ""
    source = "Injected" if first["source"] == "scenario" else "Labelled by Data-Generator"
    return f"Answer key: {source.lower()}: {what}{more}. Main cause {verdict}."


def recall_rows(r: Results) -> pd.DataFrame:
    """Found / events per kind of truth event, scenarios first."""
    rows = []
    for source in ("scenario", "anomaly_data"):
        events = r.events[r.events["source"] == source]
        for kind, frame in events.groupby("kind", sort=False):
            rows.append({"source": source, "kind": kind, "label": KIND_LABELS.get(kind, kind),
                         "found": int(frame["found"].sum()), "events": len(frame)})
    table = pd.DataFrame(rows)
    return table.sort_values(["source", "found", "events"], ascending=[False, False, True]).reset_index(drop=True)


def series_window(r: Results, anomaly_id: int, days: int = 28) -> pd.DataFrame:
    """The anomaly's segment and main metric, observed against expected, from `days` before to `days` after."""
    b = r.breakdowns[anomaly_id]
    start, end = pd.Timestamp(b.start), pd.Timestamp(b.end)
    s = r.scores
    frame = s[(s["segment"] == b.segment) & (s["metric"] == b.metric)
              & s["date"].between(start - pd.Timedelta(days=days), end + pd.Timedelta(days=days))]
    return frame[["date", "observed", "expected"]].assign(inside=frame["date"].between(start, end))


# ---------------------------------------------------------------------------------------------------------------
# Titles: the takeaway first


def cause_total(b: Breakdown, cause: str) -> float:
    """Net PLN of one cause: all its parts (the same amount the note names)."""
    return sum(p.grosz for p in b.parts if p.cause == cause) / 100


def bridge_title(b: Breakdown, lang: str = "en") -> tuple[str, str]:
    total = cause_total(b, b.main_cause)
    share = total / b.change if b.change else float("nan")
    if lang == "pl":
        return (f"Główna przyczyna: {i18n.CAUSES['pl'][b.main_cause]}, {i18n.money(total, 'pl', sign=True)}, "
                f"czyli {i18n.percent(share, 'pl', sign=False)} zmiany",
                "Od wartości oczekiwanej do rzeczywistej. Przyczyna może mieć kilka części (konwersja: po jednej na "
                "urządzenie); w miedzi największa część głównej przyczyny. Części sumują się co do grosza.")
    return (f"Main cause: {i18n.CAUSES['en'][b.main_cause]}, {i18n.money(total, 'en', sign=True)}, "
            f"{i18n.percent(share, 'en', sign=False)} of the change",
            "From the expected value to the actual one. A cause can have several parts (conversion: one per "
            "device); the largest part of the main cause is in copper. The parts add up to the grosz.")
