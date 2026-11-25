"""
Evals of the notes: every case is a sheet of facts; the model writes a note and code grades it.

    python -m evals.run                 # every case once, both languages (during work: only the cases changed)
    python -m evals.run --repeat 3      # the full run before handing over
    python -m evals.run --cases c03,c07 --lang pl

Graders (all code, no second model):
    numbers     every number in the note is in the sheet (the same check as the app)
    length      at most two sentences and explain.MAX_WORDS words
    cause       the main cause is named (a stem of its label, so Polish inflection passes)
    segment     the part of the shop is named
    change      the change in PLN is there with its value
    direction   no word for the opposite direction (a drop is not called a rise)

A run is the model's first note, before the retry the app would allow: it measures the prompt, not the safety net.
The results, with token counts and cost, go to evals/results.json (committed: the README quotes them).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sales_anomalies import explain  # noqa: E402

CASES = ROOT / "evals" / "cases.json"
RESULTS = ROOT / "evals" / "results.json"
STEMS = {
    "pl": {"discount": ["rabat"], "prices": ["cen", "asortyment"], "returns": ["zwrot"], "shipping": ["wysył"],
           "payment_fees": ["opłat", "płatnoś"], "commission": ["prowizj"], "competitor_prices": ["konkuren"],
           "campaign": ["kampani"], "conversion": ["konwersj"], "traffic": ["ruch"], "demand": ["popyt"],
           "outage": ["awari"], "single_account": ["kont"]},
    "en": {"discount": ["discount"], "prices": ["price", "mix"], "returns": ["return"], "shipping": ["shipping"],
           "payment_fees": ["payment fee"], "commission": ["commission"], "competitor_prices": ["competitor"],
           "campaign": ["campaign"], "conversion": ["conversion"], "traffic": ["traffic"], "demand": ["demand"],
           "outage": ["outage"], "single_account": ["account"]},
}
OPPOSITE = {"pl": {"down": ["wzrost", "wzrosł", "wzrosła", "wzrosły", "wyższ"],
                   "up": ["spad", "niższ", "zmalał"]},
            "en": {"down": ["rise", "rose", "increase", "higher", "grew", "jump"],
                   "up": ["drop", "fell", "fall", "decline", "lower", "decrease"]}}


def grade(text: str, case: dict, lang: str) -> dict:
    sheet = case["sheets"][lang]
    lowered = text.lower()
    verdict = explain.check(text, sheet, lang)
    segment_words = [w for w in sheet["segment"].lower().replace("·", " ").split() if len(w) > 3]
    change_value = next(iter(explain.numbers(sheet["change"], lang)), None)
    grades = {
        "numbers": not verdict["invented"],
        "length": verdict["sentences"] <= 2 and verdict["words"] <= explain.MAX_WORDS,
        "cause": any(stem in lowered for stem in STEMS[lang][case["main_cause"]]),
        "segment": any(word[:5] in lowered for word in segment_words) if segment_words else True,
        "change": change_value in explain.numbers(text, lang),
        "direction": not any(word in lowered for word in OPPOSITE[lang][case["direction"]]),
    }
    return {"grades": grades, "passed": all(grades.values()), "invented": verdict["invented"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--lang", nargs="+", default=["pl", "en"], choices=["pl", "en"])
    args = parser.parse_args()
    writer = explain.Writer.from_environment()
    if writer is None:
        print("No ANTHROPIC_API_KEY (environment or .env): the evals need the model.", file=sys.stderr)
        return 1
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    runs, started = [], time.time()
    for case in cases:
        for lang in args.lang:
            for attempt in range(args.repeat):
                text, usage = writer.write(case["sheets"][lang], lang)
                result = grade(text, case, lang)
                runs.append({"case": case["id"], "lang": lang, "attempt": attempt + 1, "text": text, **result,
                             "usage": usage})
                mark = "pass" if result["passed"] else "FAIL " + ",".join(k for k, v in result["grades"].items()
                                                                           if not v)
                print(f"{case['id']} {lang} #{attempt + 1}: {mark}")
    cost = explain.cost({"input_tokens": writer.input_tokens, "output_tokens": writer.output_tokens})
    graders = list(runs[0]["grades"]) if runs else []
    summary = {
        "model": explain.MODEL, "prompt_version": explain.PROMPT_VERSION, "cases": len(cases),
        "languages": args.lang, "repeat": args.repeat, "runs": len(runs),
        "passed": sum(r["passed"] for r in runs),
        "invented": sum(len(r["invented"]) for r in runs),
        "by_grader": {g: sum(r["grades"][g] for r in runs) for g in graders},
        "input_tokens": writer.input_tokens, "output_tokens": writer.output_tokens, "cost_usd": round(cost, 6),
        "seconds": round(time.time() - started),
        "runs_detail": runs,
    }
    if not args.cases:  # only a run over every case replaces the saved results
        RESULTS.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{summary['passed']} of {summary['runs']} runs passed; invented numbers {summary['invented']}; "
          f"cost {cost:.4f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
