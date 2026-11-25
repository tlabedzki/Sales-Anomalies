"""
The wording of every <!-- generated:... --> block of README.md, in Polish and English, from the numbers of one run.
Only numbers computed by the code enter here; scripts/make_charts.py writes the blocks into README.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from sales_anomalies import i18n

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "img" / "demo.json"


def picture(alt: str) -> str:
    return "\n".join([
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" srcset="docs/img/app-demo-dark.svg">',
        f'  <img alt="{alt}" src="docs/img/app-demo.svg">',
        "</picture>",
    ])


def _p(value: float, lang: str) -> str:
    return i18n.percent(value, lang, sign=False)


def _n(value: float, lang: str, decimals: int = 0) -> str:
    return i18n.number(value, lang, decimals)


def plural_pl(count: int, one: str, few: str, many: str) -> str:
    """Polish noun after a number: 1 alarm, 2-4 alarmy, 5+ alarmów (12-14 take many)."""
    if count == 1:
        return one
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return few
    return many


def alarms_pl(count: int) -> str:
    return plural_pl(count, "fałszywy alarm", "fałszywe alarmy", "fałszywych alarmów")


def _verdict_rows(n: dict, lang: str) -> list[str]:
    rows = n["recall"]
    out = []
    for row in rows.itertuples():
        source = ("wstrzyknięte" if row.source == "scenario" else "oznaczone przez generator") if lang == "pl" else (
            "injected" if row.source == "scenario" else "labelled by the generator")
        found = f"{row.found} z {row.events}" if lang == "pl" else f"{row.found} of {row.events}"
        out.append(f"| {row.label} | {source} | {found} |")
    return out


def blocks(n: dict) -> dict[str, str]:
    r, lang = n["r"], n["lang"]
    s = r.summary
    pl = lang == "pl"
    note = r.note(n["featured"], lang)
    notes = n["notes"]
    demo = json.loads(DEMO.read_text(encoding="utf-8")) if DEMO.exists() else {}
    alt = demo.get(f"alt_{lang}", "")
    recall = n["recall"]
    dg = recall[recall["source"] == "anomaly_data"].set_index("kind")
    outages = dg.loc["shop_outage"] if "shop_outage" in dg.index else None
    outages_found, outages_events = (outages.found, outages.events) if outages is not None else (0, 0)
    fraud = dg.loc["fraud_ring"] if "fraud_ring" in dg.index else None
    fraud_events, fraud_found = (fraud.events, fraud.found) if fraud is not None else (0, 0)
    scen = recall[recall["source"] == "scenario"]
    full = scen[scen["found"] == scen["events"]]
    none = scen[scen["found"] == 0]
    most = scen[(scen["found"] * 2 > scen["events"]) & (scen["found"] < scen["events"])]
    most_pl = "w większości: " + ", ".join(f"{r.label.lower()} ({r.found} z {r.events})" for r in most.itertuples())
    most_en = "most of: " + ", ".join(f"{r.label.lower()} ({r.found} of {r.events})" for r in most.itertuples())
    control = s.get("control", {})
    neg = s["negative_period"]
    sizes = n["sizes"]
    found_small = float(sizes.loc[sizes["found"], "signal"].min()) if sizes["found"].any() else 0.0
    missed_big = float(sizes.loc[~sizes["found"], "signal"].max()) if (~sizes["found"]).any() else 0.0
    missed_median = float(sizes.loc[~sizes["found"], "signal"].median()) if (~sizes["found"]).any() else 0.0
    found_median = float(sizes.loc[sizes["found"], "signal"].median()) if sizes["found"].any() else 0.0

    if pl:
        result = (f"**Wynik.** W dwóch latach dziennych danych wykrywanie zgłosiło {s['anomalies']} anomalii; "
                  f"{s['hits']} z nich to zdarzenia z klucza odpowiedzi ({_p(s['precision'], lang)}), przy "
                  f"{_n(s['false_per_month'], lang, 2)} fałszywego alarmu na miesiąc. Znalazło {s['scenarios_found']} "
                  f"z {s['scenarios_total']} wstrzykniętych scenariuszy, każdy rozkład zgadza się ze zmianą co do "
                  f"grosza, a główna przyczyna jest trafna w {s['main_cause_right']} z {s['main_cause_graded']} "
                  f"prawdziwych anomalii.")
        summary = (
            f"- **Trafione:** {', '.join(x.lower() for x in full['label'])} (wszystkie przypadki); "
            f"{most_pl}; awarie sklepu z `anomaly_data` ({outages_found} z {outages_events}). Każde z "
            f"{s['breakdowns']} rozbić sumuje się do zmiany co do grosza.\n"
            f"- **Chybione:** {', '.join(x.lower() for x in none['label'])} (0 trafień): ich zmiany mają "
            f"medianę {_n(missed_median, lang, 1)} odchylenia dziennego szumu i giną w nim. Z {fraud_events} oszustw z "
            f"`anomaly_data` wykrywanie znalazło {fraud_found}.")
        months = round(neg["months"])
        control_false = control.get("false_alarms", 0)
        detection = (f"![{n['titles']['timeline'][0]}](docs/img/timeline.png)\n\n"
                     f"Okres bez wstrzyknięć ({months} {plural_pl(months, 'miesiąc', 'miesiące', 'miesięcy')}): "
                     f"{neg['false']} {alarms_pl(neg['false'])}. Ten sam sklep bez żadnego scenariusza: "
                     f"{control.get('anomalies', 0)} anomalii, w tym {control_false} {alarms_pl(control_false)} "
                     f"({_n(control.get('false_per_month', 0), lang, 2)} na miesiąc).")
        recall_text = (f"![{n['titles']['recall'][0]}](docs/img/recall.png)\n\n"
                       f"![{n['titles']['sizes'][0]}](docs/img/sizes.png)\n\n"
                       f"Wykryte scenariusze mają medianę sygnału {_n(found_median, lang, 1)} odchylenia szumu, "
                       f"przeoczone {_n(missed_median, lang, 1)}; najsłabszy wykryty {_n(found_small, lang, 1)}, "
                       f"najsilniejszy przeoczony {_n(missed_big, lang, 1)}.")
        bridge = (f"![{n['titles']['bridge'][0]}](docs/img/bridge.png)\n\n"
                  f"Notatka do tej anomalii: „{note}”")
        notes_text = (f"Notatki powstały dla wszystkich {s['anomalies']} anomalii, po polsku i po angielsku: "
                      f"{notes['notes']} {plural_pl(notes['notes'], 'zdanie', 'zdania', 'zdań')}. "
                      f"Kontrolę liczb przeszło {notes['passed']} z {notes['notes']}; liczb "
                      f"spoza arkusza faktów: {notes['invented']}.")
        verdicts = (f"Wykrywanie zgłosiło {s['anomalies']} anomalii, z czego {s['false_alarms']} to fałszywe alarmy. "
                    f"Kontrole: okres bez wstrzyknięć {neg['false']} "
                    f"{alarms_pl(neg['false'])}, "
                    f"sklep bez scenariuszy "
                    f"{_n(control.get('false_per_month', 0), lang, 2)} fałszywego alarmu na miesiąc.\n\n"
                    "| Zdarzenie | Źródło | Znalezione |\n|---|---|---|\n" + "\n".join(_verdict_rows(n, lang)))
    else:
        result = (f"**Result.** Over two years of daily data the check raised {s['anomalies']} anomalies; "
                  f"{s['hits']} of them are events of the answer key ({_p(s['precision'], lang)}), with "
                  f"{_n(s['false_per_month'], lang, 2)} false alarms a month. It found {s['scenarios_found']} of "
                  f"{s['scenarios_total']} injected scenarios, every breakdown adds up exactly to the change, and "
                  f"the main cause is right in {s['main_cause_right']} of {s['main_cause_graded']} real anomalies.")
        summary = (
            f"- **What worked:** {', '.join(x.lower() for x in full['label'])} (all found); {most_en}; shop outages of "
            f"`anomaly_data` ({outages_found} of {outages_events}). Each of the {s['breakdowns']} breakdowns adds "
            f"up exactly to its change.\n"
            f"- **What missed:** {', '.join(x.lower() for x in none['label'])} (none found): their changes have "
            f"a median of {_n(missed_median, lang, 1)} deviations of the daily noise and vanish in it. Of the "
            f"{fraud_events} fraud rings in "
            f"`anomaly_data` the check found {fraud_found}.")
        detection = (f"![{n['titles']['timeline'][0]}](docs/img/timeline.png)\n\n"
                     f"In the period without injections ({neg['months']:.0f} months) there were {neg['false']} false "
                     f"alarms. The same shop without any scenario gave {control.get('anomalies', 0)} anomalies, "
                     f"{control.get('false_alarms', 0)} of them false ({_n(control.get('false_per_month', 0), lang, 2)}"
                     f" a month).")
        recall_text = (f"![{n['titles']['recall'][0]}](docs/img/recall.png)\n\n"
                       f"![{n['titles']['sizes'][0]}](docs/img/sizes.png)\n\n"
                       f"Scenarios found have a median signal of {_n(found_median, lang, 1)} noise deviations, those "
                       f"missed {_n(missed_median, lang, 1)}; the weakest found {_n(found_small, lang, 1)}, the "
                       f"strongest missed {_n(missed_big, lang, 1)}.")
        bridge = (f"![{n['titles']['bridge'][0]}](docs/img/bridge.png)\n\n"
                  f"The note on this anomaly: “{note}”")
        notes_text = (f"Every one of the {s['anomalies']} anomalies has a note in Polish and in English: "
                      f"{notes['notes']} sentences. {notes['passed']} of {notes['notes']} pass the number check; "
                      f"numbers outside the sheet of facts: {notes['invented']}.")
        verdicts = (f"The check raised {s['anomalies']} anomalies, {s['false_alarms']} of them false alarms. Controls: "
                    f"{neg['false']} false alarms in the period without injections, "
                    f"{_n(control.get('false_per_month', 0), lang, 2)} a month on the shop without scenarios.\n\n"
                    "| Event | Source | Found |\n|---|---|---|\n" + "\n".join(_verdict_rows(n, lang)))
    texts = {"result": result, "summary": summary, "detection": detection, "recall": recall_text,
             "bridge": bridge, "notes": notes_text, "verdicts": verdicts}
    if alt:
        texts["hero"] = picture(alt)
    return texts
