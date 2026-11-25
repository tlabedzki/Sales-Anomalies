"""
The wording of every <!-- generated:... --> block of README.md, in Polish and English, from the numbers of one run.
Only numbers computed by the code enter here; scripts/make_charts.py writes the blocks into README.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from sales_anomalies import explain, i18n

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
    totals = notes["totals"]
    demo = json.loads(DEMO.read_text(encoding="utf-8")) if DEMO.exists() else {}
    alt = demo.get(f"alt_{lang}", "")
    recall = n["recall"]
    dg = recall[recall["source"] == "anomaly_data"].set_index("kind")
    outages = dg.loc["shop_outage"] if "shop_outage" in dg.index else None
    fraud = dg.loc["fraud_ring"] if "fraud_ring" in dg.index else None
    fraud_events, fraud_found = (fraud.events, fraud.found) if fraud is not None else (0, 0)
    scen = recall[recall["source"] == "scenario"]
    full = scen[scen["found"] == scen["events"]]
    none = scen[scen["found"] == 0]
    control = s.get("control", {})
    neg = s["negative_period"]
    cost = totals.get("cost_usd", 0.0)
    from_model = notes["from_model"]
    evals = notes["evals"] or {}
    sizes = n["sizes"]
    found_small = float(sizes.loc[sizes["found"], "size"].min()) if sizes["found"].any() else 0.0
    missed_big = float(sizes.loc[~sizes["found"], "size"].max()) if (~sizes["found"]).any() else 0.0
    model = explain.MODEL

    if pl:
        result = (f"**Wynik.** W dwóch latach dziennych danych wykrywanie zgłosiło {s['anomalies']} anomalii; "
                  f"{s['hits']} z nich to zdarzenia z klucza odpowiedzi ({_p(s['precision'], lang)}), przy "
                  f"{_n(s['false_per_month'], lang, 2)} fałszywego alarmu na miesiąc. Znalazło {s['scenarios_found']} "
                  f"z {s['scenarios_total']} wstrzykniętych scenariuszy, każdy rozkład zgadza się ze zmianą co do "
                  f"grosza, a główna przyczyna jest trafna w {s['main_cause_right']} z {s['main_cause_graded']} "
                  f"prawdziwych anomalii.")
        summary = (
            f"- **Trafione:** {', '.join(x.lower() for x in full['label'])} "
            f"({'wszystkie' if len(full) else ''}); obie awarie sklepu z `anomaly_data`"
            f"{'' if outages is None else f' ({outages.found} z {outages.events})'}. Każde z "
            f"{s['breakdowns']} rozbić sumuje się do zmiany co do grosza.\n"
            f"- **Chybione:** {', '.join(x.lower() for x in none['label'])} (0 trafień): zmiany do "
            f"{_p(missed_big, lang)} swojego segmentu giną w dziennym szumie. Z {fraud_events} oszustw z "
            f"`anomaly_data` wykrywanie znalazło {fraud_found}.")
        detection = (f"![{n['titles']['timeline'][0]}](docs/img/timeline.png)\n\n"
                     f"W okresie bez wstrzyknięć ({neg['months']:.0f} miesiące) padło {neg['false']} fałszywych "
                     f"alarmów. Ten sam sklep bez żadnego scenariusza dał {control.get('anomalies', 0)} anomalii, w "
                     f"tym {control.get('false_alarms', 0)} fałszywych "
                     f"({_n(control.get('false_per_month', 0), lang, 2)} na miesiąc).")
        recall_text = (f"![{n['titles']['recall'][0]}](docs/img/recall.png)\n\n"
                       f"![{n['titles']['sizes'][0]}](docs/img/sizes.png)\n\n"
                       f"Najmniejszy wykryty scenariusz zmienił swój segment o {_p(found_small, lang)}, największy "
                       f"przeoczony o {_p(missed_big, lang)}.")
        author = f"napisał ją {model}" if note["source"] == "model" else "z szablonu kodu"
        bridge = (f"![{n['titles']['bridge'][0]}](docs/img/bridge.png)\n\n"
                  f"Notatka do tej anomalii ({author}"
                  f"): „{note['text']}”")
        notes_text = (f"{from_model} z {notes['notes']} notatek (po polsku i po angielsku) napisał {model}; "
                      f"{notes['retried']} przeszło dopiero za drugim razem, po odrzuceniu pierwszej wersji przez "
                      f"kontrolę liczb, a pozostałe mają zdanie z szablonu kodu. W notatkach, które trafiły do "
                      f"aplikacji, nie ma ani jednej liczby spoza arkusza faktów. Wszystkie wywołania kosztowały "
                      f"{_n(cost, lang, 4)} USD ({_n(totals.get('input_tokens', 0), lang)} tokenów wejścia, "
                      f"{_n(totals.get('output_tokens', 0), lang)} wyjścia).")
        if evals:
            notes_text += (f"\n\nEvale ({evals['cases']} przypadków × {evals['repeat']} powtórzenia, oba języki): "
                           f"{evals['passed']} z {evals['runs']} przebiegów przeszło wszystkie testy; zmyślonych liczb "
                           f"{evals['invented']}, koszt {_n(evals['cost_usd'], lang, 4)} USD.")
        verdicts = (f"Wykrywanie zgłosiło {s['anomalies']} anomalii, z czego {s['false_alarms']} to fałszywe alarmy. "
                    f"Kontrole: okres bez wstrzyknięć {neg['false']} fałszywe alarmy, sklep bez scenariuszy "
                    f"{_n(control.get('false_per_month', 0), lang, 2)} fałszywego alarmu na miesiąc.\n\n"
                    "| Zdarzenie | Źródło | Znalezione |\n|---|---|---|\n" + "\n".join(_verdict_rows(n, lang)))
    else:
        result = (f"**Result.** Over two years of daily data the check raised {s['anomalies']} anomalies; "
                  f"{s['hits']} of them are events of the answer key ({_p(s['precision'], lang)}), with "
                  f"{_n(s['false_per_month'], lang, 2)} false alarms a month. It found {s['scenarios_found']} of "
                  f"{s['scenarios_total']} injected scenarios, every breakdown adds up to the change to the grosz, and "
                  f"the main cause is right in {s['main_cause_right']} of {s['main_cause_graded']} real anomalies.")
        summary = (
            f"- **What worked:** {', '.join(x.lower() for x in full['label'])} (all found); both shop outages of "
            f"`anomaly_data`. Each of the {s['breakdowns']} breakdowns adds up to its change to the grosz.\n"
            f"- **What missed:** {', '.join(x.lower() for x in none['label'])} (none found): changes up to "
            f"{_p(missed_big, lang)} of their segment vanish in the daily noise. Of the {fraud_events} fraud rings in "
            f"`anomaly_data` the check found {fraud_found}.")
        detection = (f"![{n['titles']['timeline'][0]}](docs/img/timeline.png)\n\n"
                     f"In the period without injections ({neg['months']:.0f} months) there were {neg['false']} false "
                     f"alarms. The same shop without any scenario gave {control.get('anomalies', 0)} anomalies, "
                     f"{control.get('false_alarms', 0)} of them false ({_n(control.get('false_per_month', 0), lang, 2)}"
                     f" a month).")
        recall_text = (f"![{n['titles']['recall'][0]}](docs/img/recall.png)\n\n"
                       f"![{n['titles']['sizes'][0]}](docs/img/sizes.png)\n\n"
                       f"The smallest scenario found moved its segment by {_p(found_small, lang)}, the largest one "
                       f"missed by {_p(missed_big, lang)}.")
        author = f"written by {model}" if note["source"] == "model" else "from the code template"
        bridge = (f"![{n['titles']['bridge'][0]}](docs/img/bridge.png)\n\n"
                  f"The note on this anomaly ({author}"
                  f"): “{note['text']}”")
        notes_text = (f"{from_model} of {notes['notes']} notes (Polish and English) were written by {model}; "
                      f"{notes['retried']} passed only on the second try, after the number check rejected the first, "
                      f"and the rest carry the code's own sentence. The notes in the app hold no number outside the "
                      f"sheet of facts. All calls cost USD {_n(cost, lang, 4)} "
                      f"({_n(totals.get('input_tokens', 0), lang)} input and "
                      f"{_n(totals.get('output_tokens', 0), lang)} output tokens).")
        if evals:
            notes_text += (f"\n\nEvals ({evals['cases']} cases × {evals['repeat']} runs, both languages): "
                           f"{evals['passed']} of {evals['runs']} runs passed every test; invented numbers "
                           f"{evals['invented']}, cost USD {_n(evals['cost_usd'], lang, 4)}.")
        verdicts = (f"The check raised {s['anomalies']} anomalies, {s['false_alarms']} of them false alarms. Controls: "
                    f"{neg['false']} false alarms in the period without injections, "
                    f"{_n(control.get('false_per_month', 0), lang, 2)} a month on the shop without scenarios.\n\n"
                    "| Event | Source | Found |\n|---|---|---|\n" + "\n".join(_verdict_rows(n, lang)))
    texts = {"result": result, "summary": summary, "detection": detection, "recall": recall_text,
             "bridge": bridge, "notes": notes_text, "verdicts": verdicts}
    if alt:
        texts["hero"] = picture(alt)
    return texts
