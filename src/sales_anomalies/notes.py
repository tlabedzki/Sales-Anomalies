"""
Code computes and code writes: one or two sentences for the shop manager about each anomaly, built from numbers
the breakdown has already computed.

1. facts(): the breakdown as a small sheet of ready-made strings (segment, days, change in PLN and %, the main
   causes with their amounts, the evidence, a suggested check). Every number a note may use is in it.
2. note(): the sentence from the sheet (a fixed pattern per language). When the main cause is the metric itself
   (conversion, returns) it names the largest part instead, so it never reads "conversion, mainly conversion".
3. check(): every number in the note is one of the sheet's numbers with the same sign, the note has at most two
   sentences and MAX_WORDS words. The tests run it on the notes of the synthetic shop.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from sales_anomalies import i18n
from sales_anomalies.decompose import Breakdown

MAX_WORDS = 60
TOP_PARTS = 3

# ---------------------------------------------------------------------------------------------------------------
# Facts


def anomaly_key(breakdown: Breakdown) -> str:
    return f"{breakdown.segment}|{breakdown.metric}|{breakdown.start}|{breakdown.end}"


def _evidence(b: Breakdown, lang: str) -> list[str]:
    s, pl = b.signals, lang == "pl"
    items: list[str] = []
    if b.metric in ("sales", "margin"):
        if s.get("expected_lines"):
            items.append((f"linie zamówień: {i18n.number(s['lines'], lang)} zamiast oczekiwanych "
                          f"{i18n.number(s['expected_lines'], lang)}") if pl else
                         (f"order lines: {i18n.number(s['lines'], lang)} instead of the expected "
                          f"{i18n.number(s['expected_lines'], lang)}"))
        if b.main_cause == "discount":
            items.append((f"średni rabat {i18n.percent(s['discount_rate'], lang, sign=False)} zamiast "
                          f"{i18n.percent(s['expected_discount_rate'], lang, sign=False)}") if pl else
                         (f"average discount {i18n.percent(s['discount_rate'], lang, sign=False)} instead of "
                          f"{i18n.percent(s['expected_discount_rate'], lang, sign=False)}"))
        if b.main_cause in ("traffic", "campaign", "outage") and s.get("expected_sessions"):
            items.append((f"sesje: {i18n.number(s['sessions'], lang)} zamiast oczekiwanych "
                          f"{i18n.number(s['expected_sessions'], lang)}") if pl else
                         (f"sessions: {i18n.number(s['sessions'], lang)} instead of the expected "
                          f"{i18n.number(s['expected_sessions'], lang)}"))
        if b.main_cause == "campaign" and s.get("campaigns"):
            items.append(("trwała kampania " if pl else "campaign running: ") + ", ".join(s["campaigns"]))
        competitor = s.get("competitor")
        if b.main_cause == "competitor_prices" and competitor:
            items.append((f"{competitor['competitor']}: ceny względem naszych "
                          f"{i18n.percent(competitor['price_change'], lang)} wobec 4 poprzednich tygodni") if pl else
                         (f"{competitor['competitor']}: prices against ours "
                          f"{i18n.percent(competitor['price_change'], lang)} on the 4 weeks before"))
        account = s.get("account")
        if b.main_cause == "single_account" and account:
            part = next(p for p in b.parts if p.cause == "single_account")
            items.append((f"jedno konto klienta: {i18n.money(part.grosz / 100, lang)} w kilku zamówieniach") if pl
                         else f"one customer account: {i18n.money(part.grosz / 100, lang)} in several orders")
    elif b.metric == "returns":
        items.append((f"zwroty: {i18n.number(s['returns'], lang)} zamiast oczekiwanych "
                      f"{i18n.number(s['expected_returns'], lang)}") if pl else
                     (f"returns: {i18n.number(s['returns'], lang)} instead of the expected "
                      f"{i18n.number(s['expected_returns'], lang)}"))
        if s.get("top_reason"):
            reason = i18n.REASONS_PL.get(s["top_reason"], s["top_reason"]) if pl else s["top_reason"].replace("_", " ")
            items.append(f"najczęstszy powód: {reason}" if pl else f"most common reason: {reason}")
    elif b.metric == "costs":
        carrier = s.get("carrier")
        if carrier and b.main_cause == "shipping":
            items.append((f"{carrier['delivery_method']}: koszt paczki {i18n.percent(carrier['cost_change'], lang)}")
                         if pl else
                         f"{carrier['delivery_method']}: cost per parcel {i18n.percent(carrier['cost_change'], lang)}")
    elif b.metric == "conversion":
        items.append((f"zamówienia: {i18n.number(s['orders'], lang)} zamiast oczekiwanych "
                      f"{i18n.number(s['expected_orders'], lang)}") if pl else
                     (f"orders: {i18n.number(s['orders'], lang)} instead of the expected "
                      f"{i18n.number(s['expected_orders'], lang)}"))
        worst = min((p for p in b.parts), key=lambda p: p.amount * (1 if b.change < 0 else -1), default=None)
        if worst is not None:
            device = worst.name.removeprefix("conversion_")
            items.append(("najwięcej na urządzeniach: " + i18n.DEVICES_PL.get(device, device)) if pl else
                         f"mostly on {device}")
    return items


def facts(b: Breakdown, lang: str) -> dict:
    """The sheet of ready-made strings a note is written from (all numbers formatted for the language)."""
    start, end = pd.Timestamp(b.start), pd.Timestamp(b.end)
    expected = b.expected
    change = b.change
    sign = 1 if change >= 0 else -1
    # A cause is the net sum of all its parts (conversion: one part per device), the same amount the app's bridge
    # names; the causes that move with the change, largest first.
    by_cause: dict[str, int] = {}
    for part in b.parts:
        by_cause[part.cause] = by_cause.get(part.cause, 0) + part.grosz
    causes = sorted(((c, g) for c, g in by_cause.items() if sign * g > 0), key=lambda item: -abs(item[1]))[:TOP_PARTS]
    sheet = {
        "segment": i18n.segment(b.segment, lang),
        "main_part": main_part(b, lang),
        "metric": i18n.METRICS[lang][b.metric],
        "days": i18n.period(start, end, lang),
        "length": i18n.days((end - start).days + 1, lang),
        "direction": i18n.DIRECTIONS[lang][b.direction],
        "actual": i18n.money(b.observed, lang),
        "expected": i18n.money(expected, lang),
        "change": i18n.money(change, lang, sign=True),
        "change_percent": i18n.percent(change / expected, lang) if expected else "",
        "main_cause": i18n.CAUSES[lang][b.main_cause],
        "causes": [{"cause": i18n.CAUSES[lang][cause], "amount": i18n.money(grosz / 100, lang, sign=True)}
                   for cause, grosz in causes],
        "evidence": _evidence(b, lang),
        "suggested_check": i18n.SUGGESTIONS[lang][b.main_cause],
    }
    if SAME_AS_METRIC.get(b.metric) == b.main_cause:
        sheet["same_as_metric"] = True
    if b.metric == "conversion":
        sheet["metric_note"] = ("zmiana w złotych to sprzedaż utraconych albo dodatkowych zamówień" if lang == "pl"
                                else "the change in PLN is the sales of the orders lost or gained")
    return sheet


# Polish sentences need the genitive: "zamówienia z telefonów", "zwroty z powodu wady".
DEVICES_GENITIVE_PL = {"mobile": "telefonów", "desktop": "komputerów", "tablet": "tabletów"}
REASONS_GENITIVE_PL = {"wrong_size": "złego rozmiaru", "changed_mind": "zmiany zdania",
                       "not_as_described": "niezgodności z opisem", "defective": "wady", "damaged": "uszkodzenia",
                       "wrong_item": "pomyłki w zamówieniu"}


def part_phrase(name: str, lang: str) -> str:
    """A part of a cause in words that fit a sentence ("orders on mobile", "returns marked defective")."""
    pl = lang == "pl"
    if name.startswith("conversion_"):
        device = name.removeprefix("conversion_")
        return f"zamówienia z {DEVICES_GENITIVE_PL.get(device, device)}" if pl else f"orders on {device}"
    if name == "returns_cost_each":
        return "wyższy koszt jednego zwrotu" if pl else "a higher cost per return"
    if name.startswith("returns_"):
        reason = name.removeprefix("returns_")
        return (f"zwroty z powodu {REASONS_GENITIVE_PL.get(reason, reason)}" if pl
                else f"returns marked {reason.replace('_', ' ')}")
    from sales_anomalies.figures import part_label

    return part_label(name, lang).lower()


def main_part(b: Breakdown, lang: str) -> dict:
    """The largest part of the main cause, in words and PLN."""
    parts = [p for p in b.parts if p.cause == b.main_cause] or list(b.parts)
    part = max(parts, key=lambda p: abs(p.grosz))
    return {"part": part_phrase(part.name, lang), "amount": i18n.money(part.grosz / 100, lang, sign=True)}


# Metrics whose main cause has the same name: the note names the largest part instead of "conversion, mainly from
# conversion".
SAME_AS_METRIC = {"conversion": "conversion", "returns": "returns"}


# ---------------------------------------------------------------------------------------------------------------
# Checks


NUMBER = re.compile(r"([+\u2212]\s?(?:PLN\s)?)?(\d+(?:[ \u00a0\u202f.,]\d+)*)")


def signed_numbers(text: str, lang: str) -> set[tuple[str, float]]:
    """
    Every number written in a text with its sign: ('+', 12345.0), ('\u2212', 4321.0) or ('', 2024.0) when it has
    none. '12 345' and '12,345' are 12345, '3,5' (pl) and '3.5' (en) are 3.5. A hyphen or dash separates (a range,
    a date: 2024-07-01 gives 2024, 7 and 1); only + and the minus sign U+2212 count as signs.
    """
    values = set()
    for sign, token in NUMBER.findall(text.replace("\u2013", " | ").replace("-", " | ")):
        compact = re.sub(r"[ \u00a0\u202f]", "", token)
        # Polish: space for thousands, comma for decimals; English: comma for thousands, dot for decimals.
        compact = compact.replace(".", "").replace(",", ".") if lang == "pl" else compact.replace(",", "")
        try:
            values.add((sign.strip()[:1], round(float(compact), 2)))
        except ValueError:
            continue
    return values


def numbers(text: str, lang: str) -> set[float]:
    """The values of every number in a text, without signs."""
    return {value for _, value in signed_numbers(text, lang)}


def sheet_numbers(sheet: dict, lang: str) -> set[float]:
    return numbers(json.dumps(sheet, ensure_ascii=False), lang)


def invented_numbers(text: str, sheet: dict, lang: str) -> list[float]:
    """
    Numbers of the note that the sheet does not have: a value the sheet never shows, or a sign the sheet does not
    give it (a drop of \u22124 321 written as +4 321). A number written without a sign may be any of the sheet's.
    """
    known = signed_numbers(json.dumps(sheet, ensure_ascii=False), lang)
    values = {value for _, value in known}
    wrong = []
    for sign, value in signed_numbers(text, lang):
        if value not in values or (sign and (sign, value) not in known and ("", value) not in known):
            wrong.append(-value if sign == "\u2212" else value)
    return sorted(wrong)


def sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def check(text: str, sheet: dict, lang: str) -> dict:
    """Code's verdict on a note: invented numbers, sentence and word counts."""
    invented = invented_numbers(text, sheet, lang)
    words = len(text.split())
    problems = []
    if invented:
        problems.append("numbers not in the sheet: " + ", ".join(f"{v:g}" for v in invented))
    if sentences(text) > 2:
        problems.append(f"{sentences(text)} sentences, at most 2")
    if words > MAX_WORDS:
        problems.append(f"{words} words, at most {MAX_WORDS}")
    if not text.strip():
        problems.append("empty")
    return {"ok": not problems, "invented": invented, "words": words, "sentences": sentences(text),
            "problems": problems}


def template(sheet: dict, lang: str) -> str:
    """The note: days, metric and part of the shop, the change against expected, the main cause, the check."""
    main = sheet["causes"][0] if sheet["causes"] else {"cause": sheet["main_cause"], "amount": sheet["change"]}
    lead = "przez "  # "głównie przez rabaty"; a part reads on its own: "głównie zamówienia z telefonów"
    if sheet.get("same_as_metric"):  # not "conversion ... mainly from conversion": the largest part instead
        main = {"cause": sheet["main_part"]["part"], "amount": sheet["main_part"]["amount"]}
        lead = ""
    if lang == "pl":
        return (f"{sheet['days']}: {sheet['metric']} ({sheet['segment']}) {sheet['change']} "
                f"({sheet['change_percent']}) względem oczekiwań, głównie {lead}{main['cause']} "
                f"({main['amount']}); warto {sheet['suggested_check']}.")
    return (f"{sheet['days']}: {sheet['metric']} ({sheet['segment']}) {sheet['change']} "
            f"({sheet['change_percent']}) against what was expected, mainly {main['cause']} "
            f"({main['amount']}); {sheet['suggested_check']}.")



def note(b: Breakdown, lang: str) -> str:
    """The note on one anomaly in Polish or English."""
    return template(facts(b, lang), lang)


def notes(breakdowns: list[Breakdown], langs: tuple[str, ...] = ("pl", "en")) -> dict[str, str]:
    """Notes of every breakdown and language, keyed by '<anomaly key>|<lang>'."""
    return {f"{anomaly_key(b)}|{lang}": note(b, lang) for b in breakdowns for lang in langs}
