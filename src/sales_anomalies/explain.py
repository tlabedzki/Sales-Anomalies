"""
Code computes, AI explains: one or two sentences for the shop manager, written by a language model from numbers
the code has already computed, and checked by code before anyone sees them.

1. facts(): the breakdown as a small sheet of ready-made strings (segment, days, change in PLN and %, the main
   causes with their amounts, the evidence, a suggested check). Every number the text may use is in it.
2. The model (Claude Haiku 5.5, structured JSON output) rewrites the sheet as a note. It is told to copy numbers
   exactly and to add nothing.
3. check(): every number in the note must be one of the sheet's numbers (same value, any spacing), the note has at
   most two sentences and MAX_WORDS words. A note that fails gets one retry with the list of problems; if it fails
   again, the code's own sentence (template()) is used and marked as such.
4. Notes are saved with their token counts and cost in data/explanations.json, so the app and the README work
   without a key. With ANTHROPIC_API_KEY set (in the environment or in a .env file in the project) missing notes are
   written live.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sales_anomalies import i18n
from sales_anomalies.data import ROOT
from sales_anomalies.decompose import Breakdown

MODEL = "claude-haiku-5-5"
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}   # USD, Claude Haiku 5.5, prompts up to 100K tokens
PROMPT_VERSION = 1
MAX_WORDS = 50
MAX_TOKENS = 2048
CACHE_FILE = ROOT / "data" / "explanations.json"
ENV_FILE = ROOT / ".env"
TOP_PARTS = 3

SYSTEM = """You write a short note for the manager of an online shop about one unusual period that code has \
already found and measured. The user message is a JSON sheet of facts; every number in it was computed by code.

Write the note in {language}: one or two sentences, at most {max_words} words. Say what moved (metric, part of the \
shop, days, change in PLN and in percent against what was expected), name the main cause with its amount, and end \
with the suggested check.

Rules:
- Use only numbers that appear in the sheet and copy each exactly as written, with its sign and unit.
- Do not round, add up, subtract or convert numbers, and do not work out new percentages or totals.
- Do not add causes, guesses or advice that are not in the sheet.
- Name the part of the shop, the metric and the causes as the sheet does. When "same_as_metric" is true, the main
  cause is the metric itself: name the main part ("main_part") with its amount instead of repeating the metric.
- No greeting, no list, no quotation marks.{language_rules}

Return JSON with one field, "text"."""
LANGUAGE = {"pl": ("Polish", "\n- In Polish, do not put a comma before \"i\"."), "en": ("English", "")}

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
          "additionalProperties": False}


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
    """The sheet of ready-made strings the model writes from (all numbers formatted for the language)."""
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


def facts_hash(sheet: dict) -> str:
    return hashlib.sha256(json.dumps(sheet, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


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
    """The code's own note, used when the model is not available or its note fails the check twice."""
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


# ---------------------------------------------------------------------------------------------------------------
# Model


def api_key() -> str | None:
    """ANTHROPIC_API_KEY from the environment or from the project's .env file (never printed or stored)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def cost(usage: dict) -> float:
    return (usage["input_tokens"] * PRICE_PER_MTOK["input"] + usage["output_tokens"] * PRICE_PER_MTOK["output"]) / 1e6


@dataclass
class Writer:
    """Calls the model; one instance per run keeps the client and adds up the usage."""

    client: object
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_environment(cls) -> Writer | None:
        key = api_key()
        if not key:
            return None
        import anthropic

        return cls(anthropic.Anthropic(api_key=key, max_retries=3))

    def write(self, sheet: dict, lang: str, feedback: str | None = None) -> tuple[str, dict]:
        language, rules = LANGUAGE[lang]
        system = SYSTEM.format(language=language, max_words=MAX_WORDS, language_rules=rules)
        content = json.dumps(sheet, ensure_ascii=False, indent=1)
        if feedback:
            content += f"\n\nYour previous note was rejected by the check: {feedback}. Write it again."
        response = self.client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        self.calls += 1
        self.input_tokens += usage["input_tokens"]
        self.output_tokens += usage["output_tokens"]
        if response.stop_reason == "refusal":
            return "", usage
        text = next((block.text for block in response.content if block.type == "text"), "{}")
        try:
            return str(json.loads(text).get("text", "")).strip(), usage
        except json.JSONDecodeError:
            return "", usage


def note(writer: Writer | None, sheet: dict, lang: str) -> dict:
    """A checked note: the model's (with one retry) or the template's."""
    attempts, usage = [], {"input_tokens": 0, "output_tokens": 0}
    if writer is not None:
        feedback = None
        for _ in range(2):
            text, used = writer.write(sheet, lang, feedback)
            usage = {k: usage[k] + used[k] for k in usage}
            verdict = check(text, sheet, lang)
            attempts.append({"text": text, "check": verdict})
            if verdict["ok"]:
                return {"text": text, "source": "model", "check": verdict, "attempts": attempts, "usage": usage,
                        "cost_usd": cost(usage)}
            feedback = "; ".join(verdict["problems"])
    text = template(sheet, lang)
    return {"text": text, "source": "template", "check": check(text, sheet, lang), "attempts": attempts,
            "usage": usage, "cost_usd": cost(usage)}


# ---------------------------------------------------------------------------------------------------------------
# Cache


def load_cache(path: Path = CACHE_FILE) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"model": MODEL, "prompt_version": PROMPT_VERSION, "items": {}}


def save_cache(cache: dict, path: Path = CACHE_FILE) -> None:
    items = cache["items"]
    cache["totals"] = {
        "notes": len(items),
        "from_model": sum(1 for v in items.values() if v["source"] == "model"),
        "calls": sum(len(v["attempts"]) for v in items.values()),
        "input_tokens": sum(v["usage"]["input_tokens"] for v in items.values()),
        "output_tokens": sum(v["usage"]["output_tokens"] for v in items.values()),
        "cost_usd": round(sum(v["cost_usd"] for v in items.values()), 6),
        "retried": sum(1 for v in items.values() if len(v["attempts"]) > 1),
    }
    cache["items"] = dict(sorted(items.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def describe(breakdowns: list[Breakdown], langs: tuple[str, ...] = ("pl", "en"), live: bool | None = None,
             path: Path = CACHE_FILE) -> dict[str, dict]:
    """
    Notes for every breakdown and language, keyed by '<anomaly key>|<lang>'. Saved notes are reused when the facts
    and the prompt are the same; missing ones are written by the model when a key is available (live=None means:
    when there is a key), otherwise by the template (not saved).
    """
    cache = load_cache(path)
    writer = Writer.from_environment() if live is not False else None
    result, changed = {}, False
    for b in breakdowns:
        for lang in langs:
            sheet = facts(b, lang)
            key = f"{anomaly_key(b)}|{lang}"
            saved = cache["items"].get(key)
            if saved and saved["facts_hash"] == facts_hash(sheet) and saved["prompt_version"] == PROMPT_VERSION:
                result[key] = saved
                continue
            item = note(writer, sheet, lang)
            item.update({"facts": sheet, "facts_hash": facts_hash(sheet), "prompt_version": PROMPT_VERSION,
                         "model": MODEL if item["source"] == "model" else None, "lang": lang})
            result[key] = item
            if item["attempts"]:  # the model was asked: keep the outcome (also a template after two misses)
                cache["items"][key] = item
                changed = True
    if changed:
        save_cache(cache, path)
    return result
