"""
Grading the alerts against the answer key.

An anomaly is a hit when it matches a truth event (truth.matches: days overlap, segments related) on a metric the
event moves. For a scenario, whose exact effect is known, the event must also explain at least EXPLAINED_SHARE of
the alert: its true change in the alert's segment and days, on the alert's metric, is at least that share of what
the alert saw (same sign). A channel alert during a category's return wave counts only if the wave made it. The
same test holds for the events of anomaly_data with a record of what they added (a fraud ring's order lines, the
returns of a faulty batch); with hundreds of fraud rings, one is almost always running somewhere, and a loose match
would credit it with any alert nearby.

Precision is the share of anomalies that are hits: of everything the check raised, how much was real. Recall
(sensitivity) is the share of truth events that at least one anomaly found. The negative period has no injected
scenario, so every anomaly there that anomaly_data does not explain is a false alarm; the same holds for the whole
run on the data without any scenario (the negative control).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sales_anomalies.detect import Anomaly, Episode
from sales_anomalies.facts import Segment, ShopFacts
from sales_anomalies.scenarios import NEGATIVE_PERIOD
from sales_anomalies.truth import TruthEvent, matches

EXPLAINED_SHARE = 0.3
COLUMN_OF_METRIC = {"sales": "sales", "margin": "margin_sold", "returns": "return_cost", "costs": "costs",
                    "conversion": "orders"}


def true_change(episode: Episode, before: ShopFacts, after: ShopFacts) -> float:
    """What the injected scenarios changed in the episode's segment, days and metric (PLN, or orders)."""
    segment, column = Segment.from_key(episode.segment), COLUMN_OF_METRIC[episode.metric]
    changed = after.series(segment).loc[episode.start:episode.end, column].sum()
    original = before.series(segment).loc[episode.start:episode.end, column].sum()
    return float(changed - original)


def contributed(episode: Episode, event: TruthEvent) -> float | None:
    """What a labelled event (a fraud ring's lines, a faulty batch's returns) put into the episode's segment, days
    and metric; None when the event has no such record."""
    if event.contribution is None or episode.metric == "conversion" or episode.metric == "costs":
        return None
    frame = event.contribution
    segment = Segment.from_key(episode.segment)
    inside = frame["date"].between(episode.start, episode.end)
    if segment.category != "*":
        inside &= frame["category"] == segment.category
    if segment.channel != "*":
        inside &= frame["channel"] == segment.channel
    return float(frame.loc[inside, COLUMN_OF_METRIC[episode.metric]].sum())


def explains(event: TruthEvent, episode: Episode, facts: tuple[ShopFacts, ShopFacts] | None = None) -> bool:
    """The event moves the episode's metric, touches its segment and days, and made at least EXPLAINED_SHARE of
    what the episode saw (when that can be measured)."""
    if episode.metric not in event.metrics or not matches(event, episode.segment, episode.start, episode.end):
        return False
    seen = episode.observed - episode.expected
    if event.kind == "shop_outage":
        # The outage day explains a drop that includes it, the two days after explain the orders coming back.
        outage_day = event.start
        if episode.direction == "down":
            return episode.start <= outage_day <= episode.end
        return episode.start <= event.end and episode.end > outage_day
    if event.source == "scenario" and facts is not None:
        known = true_change(episode, *facts)
    else:
        known = contributed(episode, event)
        if known is None:  # an outage: the whole shop moves, nothing to weigh
            return True
    return bool(seen) and known / seen >= EXPLAINED_SHARE


def _matching(anomaly: Anomaly, event: TruthEvent, facts: tuple[ShopFacts, ShopFacts] | None) -> bool:
    return any(explains(event, episode, facts) for episode in anomaly.episodes)


@dataclass
class Grades:
    anomalies: pd.DataFrame   # one row per anomaly: verdict and matched events
    events: pd.DataFrame      # one row per truth event: found or not, by which anomalies
    months: float

    def precision(self) -> float:
        return float((self.anomalies["verdict"] != "false").mean()) if len(self.anomalies) else float("nan")

    def recall(self, source: str | None = None, kind: str | None = None) -> float:
        events = self.events
        if source:
            events = events[events["source"] == source]
        if kind:
            events = events[events["kind"] == kind]
        return float(events["found"].mean()) if len(events) else float("nan")

    def false_alarms(self) -> int:
        return int((self.anomalies["verdict"] == "false").sum())

    def false_per_month(self) -> float:
        return self.false_alarms() / self.months

    def negative_period(self) -> dict[str, float]:
        start, end = NEGATIVE_PERIOD
        inside = self.anomalies[(self.anomalies["start"] <= end) & (self.anomalies["end"] >= start)]
        months = (end - start).days / 30.44
        false = int((inside["verdict"] == "false").sum())
        return {"anomalies": len(inside), "false": false, "false_per_month": false / months, "months": months}


def grade(anomalies: list[Anomaly], events: list[TruthEvent], start: pd.Timestamp, end: pd.Timestamp,
          facts: tuple[ShopFacts, ShopFacts] | None = None) -> Grades:
    """Grade the anomalies; `facts` (before, after the scenarios) enables the explained-share test."""
    events = [e for e in events if e.end >= start and e.start <= end]
    rows = []
    found: dict[str, list[int]] = {e.event_id: [] for e in events}
    for anomaly in anomalies:
        hits = [e for e in events if _matching(anomaly, e, facts)]
        for event in hits:
            found[event.event_id].append(anomaly.anomaly_id)
        sources = {e.source for e in hits}
        verdict = "scenario" if "scenario" in sources else ("anomaly_data" if hits else "false")
        main = anomaly.main
        rows.append({"anomaly_id": anomaly.anomaly_id, "segment": anomaly.segment, "level": anomaly.level,
                     "start": anomaly.start, "end": anomaly.end, "metrics": ",".join(anomaly.metrics),
                     "main_metric": main.metric, "direction": main.direction, "z": main.z, "impact": main.impact,
                     "verdict": verdict, "events": ",".join(e.event_id for e in hits)})
    event_rows = [{"event_id": e.event_id, "source": e.source, "kind": e.kind, "cause": e.cause, "start": e.start,
                   "end": e.end, "segments": ",".join(e.segments), "metrics": ",".join(e.metrics),
                   "impact": e.impact, "relative_change": e.details.get("relative", float("nan")),
                   "description": e.description, "found": bool(found[e.event_id]),
                   "anomalies": ",".join(str(i) for i in found[e.event_id])} for e in events]
    months = (end - start).days / 30.44
    return Grades(pd.DataFrame(rows), pd.DataFrame(event_rows), months)
